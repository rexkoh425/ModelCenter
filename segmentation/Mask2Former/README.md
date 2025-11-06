# Mask2Former Lane Segmenter

Panoptic lane segmentation tailored for near-first-person driving videos. This tool applies the `facebook/mask2former-swin-large-cityscapes-panoptic` model to every frame and highlights drivable road/lane regions so you can visualize lane structure even in low-contrast scenes.

## Quick start

1. Install dependencies (once):
   ```
   pip install torch torchvision transformers pillow opencv-python numpy pyyaml
   ```
2. Edit `mask2former_lane_segmenter_config.yaml`:
   - Point `input_video` at your dashcam/ego-view clip.
   - Adjust `highlight_labels`, `overlay_alpha`, or `side_by_side` to taste.
3. Run the processor from the repo root:
   ```
   python Mask2Former/mask2former_lane_segmenter.py
   ```

The script writes `<input>_panoptic_lanes.mp4` by default, showing the original and highlighted frames side-by-side. Set `output_mask_only: true` to export a binary mask video that you can feed to other perception modules.

## Tips

- `max_long_edge` lets you trade accuracy for speed by downscaling before inference (helpful for 4K dashcam clips).
- Every label listed in `highlight_labels` must exist in the checkpoint's `id2label` mapping. For Cityscapes weights, useful values include `road`, `sidewalk`, `parking`, `ground`, or `lane-marking`.
- Use `max_frames` during configuration tweaks to render a short preview before committing to the whole video.

## Fine-tuning & dataset prep

If you want to fine-tune Mask2Former on custom road-lane datasets exported from Roboflow in COCO format, the repo now includes helper utilities and a training config template.

1. **Pool multiple Roboflow exports** (train/val/test folders, each containing `_annotations.coco.json` plus images):
   ```powershell
   python Mask2Former/dataset_tools/pool_coco_datasets.py `
       --inputs D:/lanes/set1/train D:/lanes/set2/train `
       --output-dir D:/lanes/pooled/train
   ```
   The script copies every image into `<output>/images` and writes `pooled_annotations.coco.json` with remapped IDs so Detectron2/Mask2Former can ingest the combined data.

2. **Create quick experiment subsets** with uniform sampling:
   ```powershell
   python Mask2Former/dataset_tools/sample_coco_subset.py `
       --dataset-root D:/lanes/pooled/train `
       --annotations pooled_annotations.coco.json `
       --output-dir D:/lanes/subset/train `
       --num-samples 800 --seed 42
   ```
   This keeps annotations consistent and copies the sampled images to `subset/train/images`.

3. **Re-split pooled data at arbitrary ratios** (e.g., a fresh 75/20/5 split). First merge any existing splits into a single directory:
   ```powershell
   python Mask2Former/dataset_tools/pool_coco_datasets.py `
       --inputs D:/lanes/pooled/train D:/lanes/pooled/valid D:/lanes/pooled/test `
       --annotation-name pooled_annotations.coco.json `
       --output-dir D:/lanes/pooled_all
   ```
   Then create the desired split in one shot:
   ```powershell
   python Mask2Former/dataset_tools/split_coco_dataset.py `
       --dataset-root D:/lanes/pooled_all `
       --annotations pooled_annotations.coco.json `
       --output-root D:/lanes/pooled_75205 `
       --splits train=75 val=20 test=5 --seed 42
   ```
   You’ll end up with `train/val/test` folders (plus JSON files) that share no images with each other.

4. **Describe your training recipe** in `mask2former_finetune_config.yaml`. The template already lists segmentation-relevant hyperparameters (epochs, batch size, optimizer, overlap_mask, mask_ratio) and augmentation knobs (HSV shifts, flips, mosaic, copy/paste, etc.) from the parameter list you provided. Update the dataset paths and tweak augmentation strengths before wiring the config into your Detectron2/Mask2Former training script.

With these pieces you can: export COCO JSON from Roboflow, pool/slice datasets as needed, and feed the curated splits plus augmentation settings straight into your fine-tuning pipeline.
