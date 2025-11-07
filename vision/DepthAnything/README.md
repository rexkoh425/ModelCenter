# Depth Anything V2 Video Processor

`depth_anything_v2_video.py` runs the `LiheYoung/depth-anything-v2-large-hf` model (via Hugging Face `transformers`) over every frame of a video to produce dense depth maps. It can output pure depth-map videos or side-by-side composites with the RGB input, apply different OpenCV colormaps, clamp percentile ranges, and limit the number of processed frames for quick previews.

## How to use

1. Install dependencies:
   ```bash
   pip install torch torchvision transformers pillow opencv-python numpy pyyaml
   ```
2. Configure `depth_anything_v2_video_config.yaml` with:
   - `input_video`: path to the clip to analyze.
   - Optional `output`, `fps`, `colormap`, percentile clamps, and `side_by_side`.
   - Device preference (`auto`, `cpu`, `cuda`) and frame limit (`max_frames`) for debugging.
3. Run the script:
   ```bash
   python DepthAnything/depth_anything_v2_video.py
   ```

The processor saves `<input>_depth.mp4` by default and prints progress every 10 frames. Extend this folder with additional post-processing (e.g., normals, point-cloud export) by reusing the `DepthAnythingVideoApp` wrapper defined in the script.*** End Patch
