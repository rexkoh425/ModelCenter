# DINO-X Quickstart

This repository contains a small helper script (`dinox_client.py`) for running the [DINO-X](https://github.com/IDEA-Research/DINO-X-API) cloud model on your own images. It sends images to the DeepDataSpace API, saves the raw JSON response, and optionally produces annotated previews.

## Prerequisites

- Python 3.9 or newer (3.10–3.12 recommended for best wheel support)
- A DeepDataSpace account and DINO-X API token: https://cloud.deepdataspace.com
- `git` (optional, only needed if you plan to clone additional repos)

> 💡 If you are on Windows, ensure the *Desktop Development with C++* workload is installed via Visual Studio Installer before installing `pycocotools`. If that is not possible, use the fallback package: `pip install pycocotools-windows`.

## 1. Create a virtual environment (recommended)

```powershell
cd C:\NUS\MachineLearning\GeneralLabeller
python -m venv .venv
.\.venv\Scripts\activate
```

On macOS or Linux:

```bash
cd ~/NUS/MachineLearning/GeneralLabeller
python -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install dds-cloudapi-sdk supervision opencv-python numpy
python -m pip install pycocotools  # or: python -m pip install pycocotools-windows
```

If you only need JSON outputs and do not plan to render masks, you can skip the `pycocotools` step.

## 3. Store your API token

Create a `.env` file in the project root (same directory as `dinox_client.py`) with your token:

```text
DDS_API_TOKEN=your_deepdataspace_token_here
```

Alternatively, pass `--api-token` to the script or export `DDS_API_TOKEN` as an environment variable.

## 4. Run DINO-X on your images

### Text-prompted detection

```bash
python dinox_client.py path\to\image.jpg --prompt "dog . bicycle . backpack" --save-visuals
```

### Prompt-free detection and segmentation

```bash
python dinox_client.py path\to\folder\with\images --prompt-free --save-visuals
```

Key flags:

- `--save-visuals`: write annotated previews when OpenCV, numpy, and supervision are installed. Masks are drawn when `pycocotools` is present.
- `--output-dir`: change where JSON files and previews are stored (defaults to `outputs/dinox`).
- `--force`: overwrite existing outputs.

Each image gets its own sub-folder (e.g. `outputs/dinox/sample_image/`) containing `dinox_result.json` plus any generated previews.

## 5. Next steps

- Inspect the JSON payloads to integrate the detections into your labelling workflow.
- Adjust `--targets` if you only need bounding boxes (`--targets bbox`) or want captions/keypoints once the API supports them.
- Batch large datasets by pointing `dinox_client.py` at a directory; the script will iterate over every supported image file.
