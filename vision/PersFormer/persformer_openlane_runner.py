"""Launcher for OpenDriveLab/PersFormer_3DLane jobs using a YAML config.

This wrapper keeps all customization (paths, experiment names, checkpoints)
inside GeneralLabeller while leveraging the upstream repo for training and
evaluation. It can auto-clone the PersFormer source, patch dataset paths, and
stage checkpoints for evaluation runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional

import yaml


DEFAULT_REPO_URL = "https://github.com/OpenDriveLab/PersFormer_3DLane.git"
CONFIG_VARIANTS = {
    "openlane": "config.persformer_openlane",
    "once": "config.persformer_once",
    "apollo": "config.persformer_apollo",
}
DEFAULT_CONFIG_FILENAME = "persformer_openlane_runner_config.yaml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


@dataclass(slots=True)
class RepoSettings:
    path: Path
    git_url: str = DEFAULT_REPO_URL
    auto_clone: bool = True
    branch: str = "main"


@dataclass(slots=True)
class RunSettings:
    dataset_variant: str = "openlane"
    dataset_name: Optional[str] = None
    dataset_dir: Path = Path(".")
    annotations_dir: Path = Path(".")
    experiment_name: str = "PersFormer_openlane"
    output_root: Optional[Path] = None
    mode: str = "evaluate"  # "train" or "evaluate"
    checkpoint: Optional[Path] = None
    batch_size: int = 1
    num_epochs: Optional[int] = None
    workers: int = 4
    master_port: int = 29666
    local_rank: int = 0
    evaluate_case: bool = False
    no_tensorboard: bool = True
    use_memcache: bool = False
    sync_bn: bool = True
    overrides: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class JobConfig:
    repo: RepoSettings
    run: RunSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PersFormer runner wrapper")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config (defaults to persformer_openlane_runner_config.yaml).",
    )
    return parser.parse_args()


def load_job_config(config_path: Path) -> JobConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    base_dir = config_path.parent

    def resolve_path(
        value: Optional[str | Path],
        *,
        required: bool = False,
        default: Optional[Path] = None,
    ) -> Optional[Path]:
        if value is None or value == "":
            if required and default is None:
                raise ValueError("Required path missing in config.")
            return default
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return candidate

    repo_section = data.get("repo", {})
    if not isinstance(repo_section, dict):
        raise ValueError("Config key 'repo' must be a mapping.")
    repo_path = resolve_path(
        repo_section.get("path"),
        required=True,
    )
    assert repo_path is not None
    repo_settings = RepoSettings(
        path=repo_path,
        git_url=str(repo_section.get("git_url", DEFAULT_REPO_URL)),
        auto_clone=bool(repo_section.get("auto_clone", True)),
        branch=str(repo_section.get("branch", "main")),
    )

    run_section = data.get("run", {})
    if not isinstance(run_section, dict):
        raise ValueError("Config key 'run' must be a mapping.")

    dataset_dir = resolve_path(run_section.get("dataset_dir"), required=True)
    annotations_dir = resolve_path(run_section.get("annotations_dir"), required=True)
    output_root = resolve_path(
        run_section.get("output_root"),
        default=(config_path.parent / "runs").resolve(),
    )
    checkpoint = resolve_path(run_section.get("checkpoint"))

    run_settings = RunSettings(
        dataset_variant=str(run_section.get("dataset_variant", "openlane")).lower(),
        dataset_name=run_section.get("dataset_name"),
        dataset_dir=dataset_dir if dataset_dir else Path("."),
        annotations_dir=annotations_dir if annotations_dir else Path("."),
        experiment_name=str(run_section.get("experiment_name", "PersFormer_openlane")),
        output_root=output_root,
        mode=str(run_section.get("mode", "evaluate")).lower(),
        checkpoint=checkpoint,
        batch_size=int(run_section.get("batch_size", 1)),
        num_epochs=_optional_int(run_section.get("num_epochs")),
        workers=int(run_section.get("workers", 4)),
        master_port=int(run_section.get("master_port", 29666)),
        local_rank=int(run_section.get("local_rank", 0)),
        evaluate_case=bool(run_section.get("evaluate_case", False)),
        no_tensorboard=bool(run_section.get("no_tensorboard", True)),
        use_memcache=bool(run_section.get("use_memcache", False)),
        sync_bn=bool(run_section.get("sync_bn", True)),
        overrides=_extract_overrides(run_section.get("overrides")),
    )

    if run_settings.mode not in {"train", "evaluate"}:
        raise ValueError("run.mode must be 'train' or 'evaluate'.")
    if run_settings.dataset_variant not in CONFIG_VARIANTS:
        raise ValueError(
            f"Unsupported dataset_variant '{run_settings.dataset_variant}'. "
            f"Expected one of: {', '.join(CONFIG_VARIANTS)}."
        )
    if run_settings.mode == "evaluate" and run_settings.checkpoint is None:
        raise ValueError("run.checkpoint is required for evaluation mode.")

    return JobConfig(repo=repo_settings, run=run_settings)


def _optional_int(value: Optional[object]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Expected integer value, got {value!r}") from exc


def _extract_overrides(node: Optional[object]) -> Dict[str, object]:
    if node is None:
        return {}
    if not isinstance(node, dict):
        raise ValueError("run.overrides must be a mapping of arg -> value.")
    return dict(node)


def ensure_repo(repo: RepoSettings) -> None:
    if repo.path.is_dir():
        return
    if not repo.auto_clone:
        raise FileNotFoundError(
            f"PersFormer repo not found at {repo.path} and auto_clone is disabled."
        )
    print(f"[PersFormer] Cloning {repo.git_url} into {repo.path} ...")
    repo.path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if repo.branch:
        cmd += ["-b", repo.branch]
    cmd += [repo.git_url, str(repo.path)]
    subprocess.run(cmd, check=True)


def ensure_checkpoint_present(source: Path, target_dir: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / source.name

    if destination.exists():
        try:
            if destination.samefile(source):
                return destination
        except OSError:
            pass
        if destination.stat().st_mtime < source.stat().st_mtime:
            destination.unlink()

    try:
        os.link(source, destination)
        print(f"[PersFormer] Linked checkpoint to {destination}")
    except OSError:
        shutil.copy2(source, destination)
        print(f"[PersFormer] Copied checkpoint to {destination}")

    return destination


def format_data_dir(path: Path) -> str:
    text = path.as_posix()
    return text if text.endswith("/") else f"{text}/"


def apply_run_settings(args, job: JobConfig) -> None:
    run = job.run

    args.dataset_name = run.dataset_name or run.dataset_variant
    args.dataset_dir = run.dataset_dir.as_posix()
    args.data_dir = format_data_dir(run.annotations_dir)
    args.mod = run.experiment_name
    if run.output_root is None:
        run.output_root = Path.cwd() / "runs"
    args.save_prefix = run.output_root.as_posix()
    args.save_path = args.save_prefix

    args.batch_size = run.batch_size
    args.nworkers = run.workers
    args.evaluate = run.mode == "evaluate"
    args.evaluate_case = run.evaluate_case
    args.no_tb = run.no_tensorboard
    args.use_memcache = run.use_memcache
    args.sync_bn = run.sync_bn
    args.local_rank = run.local_rank
    args.nodes = 1
    args.no_cuda = False

    if run.num_epochs is not None:
        args.nepochs = run.num_epochs

    for key, value in run.overrides.items():
        if not hasattr(args, key):
            raise AttributeError(f"Cannot override unknown PersFormer arg '{key}'.")
        setattr(args, key, value)


def prepare_environment(run: RunSettings) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(run.master_port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = str(run.local_rank)
    os.environ["LOCAL_RANK"] = str(run.local_rank)


def run_job(job: JobConfig) -> None:
    ensure_repo(job.repo)
    repo_dir = job.repo.path.resolve()

    sys.path.insert(0, str(repo_dir))
    original_cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        parser = _import_define_args()
        args = parser.parse_args(["--local_rank", str(job.run.local_rank)])
        config_module = import_module(CONFIG_VARIANTS[job.run.dataset_variant])
        config_module.config(args)
        apply_run_settings(args, job)
        prepare_environment(job.run)

        ddp_init = _import_ddp_init()
        runner_cls = _import_runner()
        ddp_init(args)
        runner = runner_cls(args)

        if job.run.mode == "train":
            runner.train()
        else:
            assert job.run.checkpoint is not None
            ensure_checkpoint_present(job.run.checkpoint, Path(args.save_path))
            runner.eval()
    finally:
        os.chdir(original_cwd)
        if str(repo_dir) in sys.path:
            sys.path.remove(str(repo_dir))


def _import_define_args():
    from utils.utils import define_args

    return define_args()


def _import_ddp_init():
    from experiments.ddp import ddp_init

    return ddp_init


def _import_runner():
    from experiments.runner import Runner

    return Runner


def main() -> None:
    cli_args = parse_args()
    config = load_job_config(cli_args.config)
    run_job(config)


if __name__ == "__main__":  # pragma: no cover
    main()
