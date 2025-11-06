import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bev import pointcloud_to_bev


class CameraLidarControlsDataset(Dataset):
  """
  Expects a directory of samples, each containing:
    image.png        # RGB
    lidar.npy        # Nx4 float32 (x,y,z,intensity)
    controls.json    # {"steer": float, "throttle": float, "brake": float}
  """
  def __init__(self, root: Path, bev_extent, bev_size: Tuple[int, int], bev_channels=5, transform=None):
    self.root = Path(root)
    self.samples = sorted([p for p in self.root.iterdir() if p.is_dir()])
    self.extent = bev_extent
    self.bev_size = bev_size
    self.bev_channels = bev_channels
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx: int):
    sample_dir = self.samples[idx]
    img_path = sample_dir / "image.png"
    lidar_path = sample_dir / "lidar.npy"
    ctrl_path = sample_dir / "controls.json"

    img = Image.open(img_path).convert("RGB")
    if self.transform:
      img = self.transform(img)
    else:
      img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    points = np.load(lidar_path).astype(np.float32)
    bev = pointcloud_to_bev(points, self.extent, self.bev_size)
    bev_tensor = torch.from_numpy(bev[:self.bev_channels])

    with ctrl_path.open("r", encoding="utf-8") as f:
      ctrl = json.load(f)
    target = torch.tensor([ctrl.get("steer", 0.0), ctrl.get("throttle", 0.0), ctrl.get("brake", 0.0)], dtype=torch.float32)

    return img, bev_tensor, target
