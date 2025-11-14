# PersFormer 3D Lane Launcher

This integration folder wraps the official [OpenDriveLab/PersFormer_3DLane](https://github.com/OpenDriveLab/PersFormer_3DLane) repository so you can trigger training or evaluation jobs via a single YAML config. It keeps experiment paths inside `GeneralLabeller`, and handles cloning the upstream repo plus staging checkpoints automatically.

> **Heads-up:** PersFormer targets Linux + CUDA GPUs and initialises NCCL even for single-GPU runs. Launch this script inside a CUDA-capable environment (Ubuntu + NVIDIA driver). Dataset preparation still follows the upstream instructions (OpenLane, ONCE, or Apollo formats).

## Quick start

1. Install dependencies (inside your CUDA environment):
   ```bash
   conda env create -f environment.yml  # or pip install torch torchvision tensorboardX opencv-python pyyaml
   ```
   Then install the repo requirements once:
   ```bash
   pip install -r external/PersFormer_3DLane/requirements.txt  # path taken from repo.path in the YAML
   ```
2. Download the target dataset (e.g. OpenLane) and place the image root + `lane3d_*/` annotations on disk.
3. Fetch the pretrained checkpoint you want to evaluate (see the Google Drive links in the PersFormer README for `model_best_epoch.pth.tar`).
4. Edit `persformer_openlane_runner_config.yaml`:
   - `repo.path`: where the upstream repo should live (auto-cloned on first run).
   - `run.dataset_dir` / `run.annotations_dir`: absolute paths to your data.
   - `run.checkpoint`: absolute path to the downloaded `model_best_epoch.pth.tar`.
   - Optionally tweak `batch_size`, `num_epochs`, or extra overrides (any attribute from `utils/utils.py`).
5. Run the launcher from the repo root:
   ```bash
   python PersFormer/persformer_openlane_runner.py \
       --config PersFormer/persformer_openlane_runner_config.yaml
   ```

The script will:

- Clone `OpenDriveLab/PersFormer_3DLane` (if missing) into `repo.path`.
- Import the selected dataset config (`openlane`, `once`, or `apollo`) and overwrite the paths with your YAML values.
- Stage the checkpoint into `run.output_root/<experiment_name>/model_best_epoch.pth.tar` (hard-link when possible).
- Launch PersFormer’s internal `Runner` for `train()` or `eval()`.

All logs, tensorboard files, and intermediate artifacts are saved under `run.output_root/<experiment_name>/`.

## Config highlights

| Key | Description |
| --- | --- |
| `run.mode` | `evaluate` loads `model_best*` and reports metrics; `train` runs full training. |
| `run.overrides` | Directly sets attributes on the PersFormer argparse namespace. Useful for experimenting with `learning_rate`, `prob_th`, etc. |
| `run.sync_bn` | Keep `true` when running multi-GPU, disable if you patch the upstream code to skip DDP. |
| `run.local_rank`, `run.master_port` | Handy when launching multiple jobs on the same machine; each needs a distinct NCCL port. |

## Notes

- The upstream scripts assume Unix-style slashes when they build `args.data_dir + "training/"`, so the launcher normalises paths to POSIX form automatically.
- Evaluation expects `model_best*.pth.tar` to live directly inside the experiment folder; the launcher creates a hard link when the filesystem supports it (falls back to copying).
- The official repo still controls the training loop, so any new features or bug fixes should be contributed upstream. This wrapper simply streamlines orchestration from this workspace.
