"""
Train the full BEV controller using only front camera + steer labels.
Freezes LiDAR/IMU branches; feeds dummy BEV/IMU; optimizes steer loss only.

Expected JSONL schema (camera + steer only):
{"camera": "/path/to/image.png", "steer": -0.12}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.backends import cudnn

# Ensure project root on path for relative imports
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.multimodal.partial_bev.model import EndToEndPartialBEVNet, BEVConfig


class CameraOnlyDataset(Dataset):
    def __init__(self, jsonl_path: Path, bev_config: BEVConfig, img_size=(256, 512)):
        self.items = self._load(jsonl_path)
        self.bev_config = bev_config
        self.transform = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load(path: Path) -> List[dict]:
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        img = Image.open(rec["camera"]).convert("RGB")
        img_t = self.transform(img)
        steer = torch.tensor(float(rec["steer"]), dtype=torch.float32)
        # dummy BEV and IMU
        h, w = self.bev_config.shape
        bev = torch.zeros((1, h, w), dtype=torch.float32)
        imu = torch.zeros(6, dtype=torch.float32)
        accel = torch.tensor(0.0, dtype=torch.float32)
        return img_t, bev, imu, steer, accel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-jsonl", required=True, help="JSONL with camera + steer entries.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-height", type=int, default=256)
    p.add_argument("--img-width", type=int, default=512)
    p.add_argument("--out-dir", type=str, default="/app/Output/full_camera_only")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bev_cfg = BEVConfig()
    ds = CameraOnlyDataset(Path(args.data_jsonl), bev_cfg, img_size=(args.img_height, args.img_width))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = EndToEndPartialBEVNet(
        img_channels=3,
        imu_input_dim=6,
        freeze_bev=True,
        freeze_imu=True,
        img_feat_dim=256,
        bev_feat_dim=128,
        imu_feat_dim=32,
    ).to(device)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    mse = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        steps = len(loader)
        log_every = max(1, int(steps * 0.05))
        for step, (img, bev, imu, steer, accel) in enumerate(loader, start=1):
            img = img.to(device)
            bev = bev.to(device)
            imu = imu.to(device)
            steer = steer.to(device)
            # accel target unused; present for shape
            pred_steer, _ = model(img, bev, imu)
            loss = mse(pred_steer, steer)
            optim.zero_grad()
            loss.backward()
            optim.step()
            bs = img.size(0)
            total += loss.item() * bs
            n += bs
            if step % log_every == 0:
                pct = (step / steps) * 100
                print(f"Epoch {epoch} [{pct:5.1f}%] step {step}/{steps} loss={loss.item():.4f}")
        print(f"Epoch {epoch}: train_loss={total / max(1,n):.4f}")

    torch.save(model.state_dict(), out_dir / "full_camera_only.pt")
    print(f"Saved camera-only-steer (full model with frozen branches) to {out_dir / 'full_camera_only.pt'}")


if __name__ == "__main__":
    main()
