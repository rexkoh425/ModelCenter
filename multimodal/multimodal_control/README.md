# Multimodal Camera + LiDAR Control (Scaffold)

Lightweight end-to-end policy template that fuses a front RGB camera and a LiDAR BEV raster to predict low-level controls (steer, throttle, brake). This is a scaffold: wire in your data paths, training loop, and deployment hooks as needed.

## Files
- `config.yaml` — model/training hyperparameters and data paths.
- `dataset.py` — dataset stub that loads camera images, LiDAR point clouds (npy), and control labels.
- `bev.py` — simple LiDAR → BEV rasterizer.
- `model.py` — fusion model (ResNet18 encoder for RGB + small CNN for BEV).
- `train.py` — minimal training loop skeleton.
- `infer.py` — example inference script for a single sample (for debugging).

## Minimal usage
1) Prepare a dataset directory with samples like:
   ```
   sample_0001/
     image.png
     lidar.npy        # Nx4 float32 (x,y,z,intensity)
     controls.json    # {"steer": 0.0, "throttle": 0.3, "brake": 0.0}
   ```
   Update `config.yaml` with your train/val roots.

2) Train (placeholder loop):
   ```
   python train.py --config config.yaml
   ```

3) Debug inference on one sample:
   ```
   python infer.py --config config.yaml --sample path/to/sample_0001
   ```

## Notes
- The BEV rasterizer is intentionally simple. Replace with a pillar/voxel encoder if you need higher fidelity.
- The model outputs steer/throttle/brake in one head; add trajectory or risk heads as needed.
- Keep training in FP16 on GPU for speed once the pipeline is stable.
