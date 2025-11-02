"""Rename images in a folder using BLIP captions driven by a YAML config file.

This script generates a caption for each image with a BLIP image captioning
model from Hugging Face and renames the file accordingly. Filenames are
normalised (lowercase ASCII, underscores) and truncated to a configurable
length, with automatic collision handling. Behaviour is configured via a YAML
file so common settings can be reused.

Usage:
    # Update BLIP/blip_caption_renamer_config.yaml, then run:
    python blip_caption_renamer.py

Minimal config example:

    directory: ./images
    dry_run: true

Dependencies (install beforehand):
    pip install torch torchvision transformers pillow pyyaml
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import yaml
from PIL import Image
from transformers import pipeline


DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-large"
DEFAULT_MAX_CHARS = 80
DEFAULT_MAX_NEW_TOKENS = 30
DEFAULT_CONFIG_FILENAME = "blip_caption_renamer_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".webp",
    ".jfif",
    ".avif",
)

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass
class CaptionRenameConfig:
    directory: Path
    model: str = DEFAULT_MODEL_ID
    device: str = "auto"
    max_chars: int = DEFAULT_MAX_CHARS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    prefix: str = ""
    suffix: str = ""
    extensions: Optional[Sequence[str]] = None
    recursive: bool = False
    dry_run: bool = False


def load_config(config_path: Path) -> CaptionRenameConfig:
    expanded_path = Path(config_path).expanduser()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"Config file not found: {expanded_path}")

    with expanded_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping of option names to values.")

    def require_directory() -> Path:
        directory_value = loaded.get("directory")
        if directory_value is None:
            raise ValueError("Config must define 'directory'.")
        return Path(str(directory_value)).expanduser()

    def coerce_str(key: str, default: str) -> str:
        value = loaded.get(key, default)
        return str(value)

    def coerce_int(key: str, default: int) -> int:
        value = loaded.get(key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config field '{key}' must be an integer.") from exc

    def coerce_bool(key: str, default: bool) -> bool:
        value = loaded.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        if value is None:
            return default
        raise ValueError(f"Config field '{key}' must be a boolean.")

    def coerce_extensions(key: str) -> Optional[Sequence[str]]:
        value = loaded.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [str(item) for item in value]
        raise ValueError(f"Config field '{key}' must be a string or list of strings.")

    directory = require_directory()
    model = coerce_str("model", DEFAULT_MODEL_ID)
    device = coerce_str("device", "auto")
    max_chars = coerce_int("max_chars", DEFAULT_MAX_CHARS)
    max_new_tokens = coerce_int("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    prefix = coerce_str("prefix", "")
    suffix = coerce_str("suffix", "")
    extensions = coerce_extensions("extensions")
    recursive = coerce_bool("recursive", False)
    dry_run = coerce_bool("dry_run", False)

    return CaptionRenameConfig(
        directory=directory,
        model=model,
        device=device,
        max_chars=max_chars,
        max_new_tokens=max_new_tokens,
        prefix=prefix,
        suffix=suffix,
        extensions=extensions,
        recursive=recursive,
        dry_run=dry_run,
    )


def resolve_device(requested: str) -> int | str:
    if requested == "auto":
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():  # pragma: no cover - Mac specific
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return 0
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available.")
        return "mps"
    return "cpu"


def sanitise_fragment(text: str) -> str:
    if not text:
        return ""
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().strip()
    ascii_text = re.sub(r"[^\w\s-]", "", ascii_text)
    compressed = re.sub(r"[\s-]+", "_", ascii_text)
    return compressed.strip("_")


def clamp_stem_length(stem: str, max_chars: int) -> str:
    if len(stem) <= max_chars:
        return stem
    trimmed = stem[:max_chars].rstrip("_")
    return trimmed or stem[:max_chars]


def is_reserved_name(name: str) -> bool:
    return name.upper() in WINDOWS_RESERVED_NAMES


def compose_stem(
    caption_stem: str,
    prefix_fragment: str,
    suffix_fragment: str,
    max_chars: int,
) -> str:
    fragments = [frag for frag in (prefix_fragment, caption_stem, suffix_fragment) if frag]
    combined = "_".join(fragments) if fragments else caption_stem
    combined = clamp_stem_length(combined, max_chars)
    combined = combined or "image"
    if is_reserved_name(combined):
        combined = f"{combined}_file"
    return combined


def find_images(
    directory: Path,
    extensions: Sequence[str],
    recursive: bool,
) -> list[Path]:
    search: Iterable[Path]
    pattern = "*"
    if recursive:
        search = directory.rglob(pattern)
    else:
        search = directory.glob(pattern)
    lower_exts = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
    lower_exts = {ext.lower() for ext in lower_exts}
    images: list[Path] = []
    for path in search:
        if path.is_file() and path.suffix.lower() in lower_exts:
            images.append(path)
    images.sort(key=lambda p: p.relative_to(directory).as_posix().lower())
    return images


def caption_image(
    captioner,
    image_path: Path,
    max_new_tokens: int,
) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        outputs = captioner(rgb, max_new_tokens=max_new_tokens)
    if not outputs:
        raise RuntimeError("Model did not return any captions.")
    caption = outputs[0].get("generated_text") or outputs[0].get("caption")
    if not caption:
        raise RuntimeError("Caption output did not contain text.")
    return caption.strip()


def ensure_unique_filename(
    parent: Path,
    stem: str,
    extension: str,
    used_names: set[str],
    current_name: str,
) -> str:
    extension = extension.lower()
    candidate = f"{stem}{extension}"
    counter = 1
    while True:
        if candidate == current_name:
            used_names.add(candidate)
            return candidate
        if candidate not in used_names and not (parent / candidate).exists():
            used_names.add(candidate)
            return candidate
        candidate = f"{stem}_{counter}{extension}"
        counter += 1


def build_used_name_cache(images: Sequence[Path]) -> dict[Path, set[str]]:
    cache: dict[Path, set[str]] = {}
    for image_path in images:
        parent = image_path.parent
        if parent not in cache:
            cache[parent] = {entry.name for entry in parent.iterdir() if entry.is_file()}
    return cache


def normalise_caption(caption: str, max_chars: int) -> str:
    stem = sanitise_fragment(caption)
    stem = clamp_stem_length(stem, max_chars)
    stem = stem or "image"
    if is_reserved_name(stem):
        stem = f"{stem}_file"
    return stem


def rename_images(config: CaptionRenameConfig) -> int:
    directory = config.directory.expanduser()
    if not directory.exists():
        print(f"Error: directory not found: {directory}", file=sys.stderr)
        return 1
    if not directory.is_dir():
        print(f"Error: path is not a directory: {directory}", file=sys.stderr)
        return 1
    if config.max_chars < 4:
        print("Error: max_chars must be at least 4.", file=sys.stderr)
        return 1
    if config.max_new_tokens < 1:
        print("Error: max_new_tokens must be positive.", file=sys.stderr)
        return 1

    extensions = (
        tuple(config.extensions)
        if config.extensions
        else IMAGE_EXTENSIONS
    )
    images = find_images(directory, extensions, config.recursive)
    if not images:
        print("No matching images were found.", file=sys.stderr)
        return 0

    device = resolve_device(config.device)
    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    captioner = pipeline(
        task="image-to-text",
        model=config.model,
        device=device,
        torch_dtype=torch_dtype,
    )

    used_cache = build_used_name_cache(images)
    renamed = 0
    prefix_fragment = sanitise_fragment(config.prefix)
    suffix_fragment = sanitise_fragment(config.suffix)

    for index, image_path in enumerate(images, 1):
        parent = image_path.parent
        used_names = used_cache[parent]
        original_name = image_path.name
        used_names.discard(original_name)

        try:
            caption = caption_image(captioner, image_path, config.max_new_tokens)
        except Exception as exc:  # pragma: no cover - runtime errors
            print(f"[error] {image_path}: {exc}", file=sys.stderr)
            used_names.add(original_name)
            continue

        caption_stem = normalise_caption(caption, config.max_chars)
        composed_stem = compose_stem(
            caption_stem,
            prefix_fragment,
            suffix_fragment,
            config.max_chars,
        )
        new_name = ensure_unique_filename(
            parent,
            composed_stem,
            image_path.suffix,
            used_names,
            original_name,
        )
        target_path = parent / new_name

        if new_name == original_name:
            print(f"[skip] {image_path} already matches the caption.")
            used_names.add(original_name)
            continue

        if config.dry_run:
            print(f"[dry-run] {image_path} -> {target_path}")
        else:
            source_str = str(image_path)
            image_path.rename(target_path)
            print(f"[renamed] {source_str} -> {target_path}")
            renamed += 1

    print(f"Processed {len(images)} image(s); renamed {renamed}.")
    return 0


class BlipCaptionRenamer:
    """Class-based entry point for the BLIP caption-driven renamer."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        self.config: CaptionRenameConfig | None = None

    def load(self) -> None:
        self.config = load_config(self.config_path)

    def run(self) -> int:
        if self.config is None:
            self.load()
        assert self.config is not None  # mypy/runtime guard
        return rename_images(self.config)


def main() -> None:
    runner = BlipCaptionRenamer()
    try:
        exit_code = runner.run()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        print("\nInterrupted by user.")
        exit_code = 130
    except Exception as exc:  # pragma: no cover - unexpected errors
        print(f"Error: {exc}", file=sys.stderr)
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
