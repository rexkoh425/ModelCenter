# YOLOE Bounding Box Pipeline

`yoloe_bbox_pipeline.py` runs the official [YOLOE](https://github.com/THU-MIG/yoloe) checkpoints for zero-shot text-prompted detection on local images. It downloads weights from Hugging Face, performs inference with the `ultralytics` implementation, and saves both JSON outputs and optional visualisations.

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust to your CUDA/CPU build
python -m pip install "ultralytics>=8.3.60" huggingface-hub supervision opencv-python numpy
```

- If you do not need image previews, you can omit `supervision`, `opencv-python`, and `numpy`.
- Install the appropriate PyTorch wheel for your platform/GPU; the example above targets CUDA 12.1.

## 2. Run detection

```bash
python YOLOE/yoloe_bbox_pipeline.py path/to/image_or_dir \
  --prompt "person, bus, traffic light" \
  --model yoloe-v8l \
  --save-visuals
```

Key flags:

- `--prompt`: comma-separated list of open-set categories (default: `person, car`).
- `--model`: choose from `yoloe-v8s`, `yoloe-v8m`, `yoloe-v8l`, `yoloe-11s`, `yoloe-11m`, `yoloe-11l`.
- `--image-size`, `--confidence`, `--iou`: inference hyper-parameters mirroring the Ultralytics CLI.
- `--output-dir`: destination folder for JSON results (defaults to `outputs/yoloe`).
- `--save-visuals`: save annotated previews (requires the optional visualization deps).
- `--device`: force `cpu` / `cuda` or leave as `auto`.

Each processed image writes:

- `outputs/yoloe/<image_stem>/yoloe_result.json` – detection metadata (`bbox`, `confidence`, `class_name`).
- `outputs/yoloe/<image_stem>/annotated.jpg` – only when `--save-visuals` is enabled and helpers are installed.

## 3. Tips

- The first run will download the selected checkpoint (≈200–500 MB) to the local Hugging Face cache.
- For large batches, point the script at a directory; it automatically filters supported extensions.
- If you want prompt-free or visual-prompt modes, extend the script by following the reference implementation in the [YOLOE Space](https://huggingface.co/spaces/jameslahm/yoloe) (`app.py`).

This pipeline mirrors the ergonomics of the `DINO-X` helper so you can keep both toolchains side by side in this project.
