# LaneATT Baseline

Lightweight wrapper to run a LaneATT lane detector alongside the existing Mask2Former tooling. Everything here lives in its own folder so you can bring LaneATT into experiments without touching the Mask2Former setup.

## Setup
- Create/activate your environment, then install the essentials:  
  `pip install -r LaneATT/requirements.txt`
- Grab weights (e.g., `laneatt_100.pt`) from the RealTime-LaneATT repo and drop them under `LaneATT/checkpoints/`. The default config expects `LaneATT/checkpoints/laneatt_100.pt`.
- Adjust `LaneATT/laneatt_model_config.yaml` if you want a different backbone or anchor grid. It already mirrors the pip package defaults.

## Run video inference
1) Point `input_video`, `weights`, and optional `output` in `LaneATT/laneatt_inference_config.yaml`.  
2) Run:
```
python LaneATT/laneatt_video_inference.py
```
The script resizes frames to the model’s training size (from `laneatt_model_config.yaml`), draws the predicted lanes, and writes `<input>_laneatt.mp4` if no output name is given.

Notable toggles in `laneatt_inference_config.yaml`:
- `apply_nms`: CPU NMS from the package (helps reduce duplicates).
- `frame_stride`: skip frames for faster previews.
- `max_frames`: limit render length when testing settings.

## Training entrypoint
If you want to fine-tune on your own TuSimple-style JSON splits, set the dataset roots in `laneatt_model_config.yaml`, then:
```
from laneatt import LaneATT

model = LaneATT("LaneATT/laneatt_model_config.yaml")
model.train_model()          # start training
model.eval()                 # switch to eval before inference
model.load("LaneATT/checkpoints/your_weights.pt")
```

## File map
- `laneatt_model_config.yaml` — model/backbone, anchor grid, and training directories.
- `laneatt_inference_config.yaml` — video paths plus NMS/stride knobs.
- `laneatt_video_inference.py` — video runner that wraps the pip `laneatt` package.
- `requirements.txt` — minimal dependencies (PyTorch, laneatt, cv2, yaml, numpy, scipy).
