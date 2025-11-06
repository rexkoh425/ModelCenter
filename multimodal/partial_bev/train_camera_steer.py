"""
Camera-only steering trainer for CARLA data.

Expects a JSONL with entries like:
{"camera": "D:/Datasets/CarlaDataSet/CameraFront_Steer/frame_000123.png", "steer": -0.12}

Outputs a simple conv net that predicts steering in [-1, 1].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.backends import cudnn


class CameraSteerDataset(Dataset):
    def __init__(self, jsonl_path: Path, img_size=(128, 256)):
        self.items = self._load(jsonl_path)
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
        return img_t, steer


class CameraSteerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),  # steer in [-1, 1]
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-jsonl", required=True, help="Path to JSONL with camera/steer entries.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-height", type=int, default=128)
    p.add_argument("--img-width", type=int, default=256)
    p.add_argument("--out-dir", type=str, default="Output/camera_steer_model")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = CameraSteerDataset(Path(args.data_jsonl), img_size=(args.img_height, args.img_width))
    prefetch_factor = 2 if args.num_workers > 0 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    model = CameraSteerNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        steps = len(loader)
        log_every = max(1, int(steps * 0.05))  # every 5% of an epoch
        for step, (img, steer) in enumerate(loader, start=1):
            img = img.to(device)
            steer = steer.to(device)
            pred = model(img)
            loss = loss_fn(pred, steer)
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

    torch.save(model.state_dict(), out_dir / "camera_steer.pt")
    print(f"Saved camera-only steer model to {out_dir / 'camera_steer.pt'}")


if __name__ == "__main__":
    main()
