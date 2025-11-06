"""
End-to-end partial-BEV driving stack:
- LiDAR -> BEV occupancy grid (2D CNN)
- Front RGB camera (perspective CNN)
- IMU + speed MLP
- Fused MLP head -> steering [-1, 1], accel [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    import timm  # type: ignore
except Exception:  # pragma: no cover
    timm = None


# --------------------------- LiDAR -> BEV ------------------------------------


@dataclass
class BEVConfig:
    x_range: Tuple[float, float] = (-20.0, 20.0)  # left/right (meters)
    y_range: Tuple[float, float] = (0.0, 40.0)    # front (meters)
    resolution: float = 0.2                       # meters per cell

    @property
    def shape(self) -> Tuple[int, int]:
        h = int((self.y_range[1] - self.y_range[0]) / self.resolution)
        w = int((self.x_range[1] - self.x_range[0]) / self.resolution)
        return h, w


def lidar_to_bev(
    lidar_ranges: Union[np.ndarray, torch.Tensor],
    lidar_angles: Union[np.ndarray, torch.Tensor],
    cfg: BEVConfig = BEVConfig(),
) -> torch.Tensor:
    """
    Convert polar LiDAR scan to 1-channel occupancy BEV grid.
    Returns: torch.Tensor, shape (1, H, W), float32 in [0, 1]
    """
    if isinstance(lidar_ranges, np.ndarray):
        r = torch.from_numpy(lidar_ranges)
    else:
        r = lidar_ranges
    if isinstance(lidar_angles, np.ndarray):
        a = torch.from_numpy(lidar_angles)
    else:
        a = lidar_angles

    r = r.float()
    a = a.float()

    mask = torch.isfinite(r) & torch.isfinite(a) & (r > 0)
    r = r[mask]
    a = a[mask]

    x = r * torch.cos(a)
    y = r * torch.sin(a)

    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    keep = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x = x[keep]
    y = y[keep]

    h, w = cfg.shape
    j = torch.floor((x - x_min) / cfg.resolution).long()  # width index
    i = torch.floor((y - y_min) / cfg.resolution).long()  # height index

    bev = torch.zeros((h, w), dtype=torch.float32, device=x.device)
    if i.numel() > 0:
        bev[i.clamp(0, h - 1), j.clamp(0, w - 1)] = 1.0

    return bev.unsqueeze(0)  # (1, H, W)


# --------------------------- Model -------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EndToEndPartialBEVNet(nn.Module):
    def __init__(
        self,
        img_feat_dim: int = 256,
        bev_feat_dim: int = 128,
        imu_feat_dim: int = 32,
        imu_input_dim: int = 6,
        img_channels: int = 3,
        freeze_camera: bool = False,
        freeze_bev: bool = False,
        freeze_imu: bool = False,
        freeze_head: bool = False,
    ):
        super().__init__()
        # Camera branch (ConvNeXt-Tiny if timm is available; fallback to small CNN)
        self.cam_input_adapter = None
        if img_channels != 3:
            # Map arbitrary channels to 3 for pretrained backbone
            self.cam_input_adapter = nn.Conv2d(img_channels, 3, kernel_size=1, stride=1, padding=0, bias=False)
            cam_in_ch = 3
        else:
            cam_in_ch = img_channels

        if timm is not None:
            backbone = timm.create_model("convnext_tiny", features_only=True, pretrained=True, in_chans=cam_in_ch)
            cam_out_ch = backbone.feature_info.channels()[-1]
            self.cam_backbone = backbone
            self.cam_pool = nn.AdaptiveAvgPool2d((4, 8))
            self.cam_proj = nn.Linear(cam_out_ch * 4 * 8, img_feat_dim)
        else:
            self.cam_backbone = nn.Sequential(
                ConvBlock(cam_in_ch, 32, stride=2),
                ConvBlock(32, 64, stride=2),
                ConvBlock(64, 128, stride=2),
                nn.AdaptiveAvgPool2d((4, 8)),
            )
            self.cam_proj = nn.Linear(128 * 4 * 8, img_feat_dim)

        # BEV branch
        self.bev_backbone = nn.Sequential(
            ConvBlock(1, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.bev_proj = nn.Linear(128 * 4 * 4, bev_feat_dim)

        # IMU + speed branch
        self.imu_mlp = nn.Sequential(
            nn.Linear(imu_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, imu_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + bev_feat_dim + imu_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64, 2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Optional freezing of branches/head based on config
        if freeze_camera:
            self._freeze_module(self.cam_backbone)
            self._freeze_module(self.cam_proj)
        if freeze_bev:
            self._freeze_module(self.bev_backbone)
            self._freeze_module(self.bev_proj)
        if freeze_imu:
            self._freeze_module(self.imu_mlp)
        if freeze_head:
            self._freeze_module(self.fusion)
            self._freeze_module(self.head)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def forward(
        self,
        img: torch.Tensor,
        bev: torch.Tensor,
        imu_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cam_input_adapter is not None:
            img = self.cam_input_adapter(img)
        cam_feat = self.cam_backbone(img)
        if isinstance(cam_feat, (list, tuple)):
            cam_feat = cam_feat[-1]
        cam_feat = self.cam_pool(cam_feat) if hasattr(self, "cam_pool") else cam_feat
        cam_feat = cam_feat.flatten(1)
        cam_feat = self.cam_proj(cam_feat)

        bev_feat = self.bev_backbone(bev)
        bev_feat = bev_feat.flatten(1)
        bev_feat = self.bev_proj(bev_feat)

        imu_feat = self.imu_mlp(imu_state)

        fused = torch.cat([cam_feat, bev_feat, imu_feat], dim=1)
        fused = self.fusion(fused)
        out = self.head(fused)
        steer = self.tanh(out[:, 0:1])         # [-1, 1]
        accel = self.sigmoid(out[:, 1:2])      # [0, 1]
        return steer.squeeze(1), accel.squeeze(1)


ARCH_DIAGRAM = r"""
Camera RGB (3xHxW) ---> CNN + GAP ---> img_feat (256)
                                  \
                                   +--> concat --> MLP --> [steer, accel]
LiDAR BEV (1xHb xWb) -> CNN + GAP ---> bev_feat (128)
                                  \
IMU+speed(+extras) (6-8+) -------> MLP ----------> imu_feat (32)
"""


# --------------------------- Dataset -----------------------------------------


class DrivingDataset(Dataset):
    """
    Expects index entries with keys:
    - camera: np.ndarray or torch.Tensor (H,W,3) or (3,H,W)
    - lidar_ranges: np.ndarray / torch.Tensor (N,)
    - lidar_angles: np.ndarray / torch.Tensor (N,)
    - imu: Sequence[float] length 5
    - speed: float
    - steer: float, accel: float
    Optional:
    - extra_image_keys: list of keys whose values are HxW masks to be stacked as extra channels (e.g., mask2former/yolop outputs)
    - extra_state_keys: list of keys appended to the state vector (e.g., traffic light state)
    """

    def __init__(
        self,
        index: Sequence[dict],
        img_transform: Optional[Callable] = None,
        bev_config: BEVConfig = BEVConfig(),
        imu_scale: Sequence[float] = (5.0, 5.0, 2.0, 1.0, 1.0, 30.0),
        extra_state_keys: Optional[Sequence[str]] = None,
        extra_image_keys: Optional[Sequence[str]] = None,
        extra_image_transform: Optional[Callable] = None,
    ):
        self.index = index
        self.img_transform = img_transform
        self.bev_config = bev_config
        self.imu_scale = torch.tensor(imu_scale, dtype=torch.float32)
        self.extra_state_keys = list(extra_state_keys or [])
        self.extra_image_keys = list(extra_image_keys or [])
        self.extra_image_transform = extra_image_transform
        # base channels (RGB) plus any extra image channels
        self.num_img_channels = 3 + len(self.extra_image_keys)

    def __len__(self) -> int:
        return len(self.index)

    def _prep_img(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                arr = torch.from_numpy(arr)
        img = arr.float() / 255.0 if arr.max() > 1.0 else arr.float()
        if self.img_transform:
            img = self.img_transform(img)
        return img

    def _prep_extra_channel(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        # Expect HxW; expand to 1xHxW
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[0] != 1 and arr.shape[2] == 1:
            arr = arr.permute(2, 0, 1)
        arr = arr.float()
        if arr.max() > 1.0:
            arr = arr / 255.0
        if self.extra_image_transform:
            arr = self.extra_image_transform(arr)
        return arr

    def __getitem__(self, idx: int):
        rec = self.index[idx]
        img = self._prep_img(rec["camera"])
        # Stack any extra image channels (e.g., mask2former/yolop outputs)
        if self.extra_image_keys:
            extras = []
            for k in self.extra_image_keys:
                if k in rec:
                    extras.append(self._prep_extra_channel(rec[k]))
            if extras:
                img = torch.cat([img] + extras, dim=0)
        bev = lidar_to_bev(rec["lidar_ranges"], rec["lidar_angles"], self.bev_config)

        extras = [rec[k] for k in self.extra_state_keys if k in rec]
        imu = torch.tensor(list(rec["imu"]) + [rec["speed"]] + extras, dtype=torch.float32)
        scale = torch.cat([self.imu_scale, torch.ones(len(extras))])
        imu_norm = imu / scale

        steer = torch.tensor(rec["steer"], dtype=torch.float32)
        accel = torch.tensor(rec["accel"], dtype=torch.float32)
        return img, bev, imu_norm, steer, accel


# --------------------------- Training / Eval ---------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steer_w: float = 1.0,
    accel_w: float = 1.0,
) -> Tuple[float, float, float]:
    model.train()
    mse = nn.MSELoss()
    total_loss = total_steer = total_accel = 0.0
    n = 0
    for img, bev, imu, steer, accel in loader:
        img, bev, imu = img.to(device), bev.to(device), imu.to(device)
        steer, accel = steer.to(device), accel.to(device)
        optimizer.zero_grad()
        pred_steer, pred_accel = model(img, bev, imu)
        loss_steer = mse(pred_steer, steer)
        loss_accel = mse(pred_accel, accel)
        loss = steer_w * loss_steer + accel_w * loss_accel
        loss.backward()
        optimizer.step()
        bs = img.size(0)
        n += bs
        total_loss += loss.item() * bs
        total_steer += loss_steer.item() * bs
        total_accel += loss_accel.item() * bs
    return total_loss / n, total_steer / n, total_accel / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    mae = nn.L1Loss()
    total_steer = total_accel = 0.0
    n = 0
    for img, bev, imu, steer, accel in loader:
        img, bev, imu = img.to(device), bev.to(device), imu.to(device)
        steer, accel = steer.to(device), accel.to(device)
        pred_steer, pred_accel = model(img, bev, imu)
        total_steer += mae(pred_steer, steer).item() * img.size(0)
        total_accel += mae(pred_accel, accel).item() * img.size(0)
        n += img.size(0)
    return total_steer / n, total_accel / n


# --------------------------- Inference helper --------------------------------


def run_inference(
    model: nn.Module,
    camera_frame: Union[np.ndarray, torch.Tensor],
    lidar_ranges: Union[np.ndarray, torch.Tensor],
    lidar_angles: Union[np.ndarray, torch.Tensor],
    imu_vec: Sequence[float],
    speed: float,
    extra_state: Optional[Sequence[float]] = None,
    bev_config: BEVConfig = BEVConfig(),
    img_transform: Optional[Callable] = None,
    imu_scale: Sequence[float] = (5.0, 5.0, 2.0, 1.0, 1.0, 30.0),
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    model.eval()
    device = device or next(model.parameters()).device
    ds = DrivingDataset(
        index=[],
        img_transform=img_transform,
        bev_config=bev_config,
        imu_scale=imu_scale,
    )
    img = ds._prep_img(camera_frame).unsqueeze(0).to(device)
    bev = lidar_to_bev(lidar_ranges, lidar_angles, bev_config).unsqueeze(0).to(device)
    extras = list(extra_state or [])
    imu = torch.tensor(list(imu_vec) + [speed] + extras, dtype=torch.float32, device=device)
    scale = torch.cat([torch.tensor(imu_scale, dtype=torch.float32, device=device), torch.ones(len(extras), device=device)])
    imu = imu / scale
    with torch.no_grad():
        steer, accel = model(img, bev, imu.unsqueeze(0))
    steer = float(torch.clamp(steer, -1.0, 1.0).item())
    accel = float(torch.clamp(accel, 0.0, 1.0).item())
    return steer, accel


# --------------------------- Example usage -----------------------------------


def example():
    dummy = [
        {
            "camera": np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8),
            "lidar_ranges": np.random.uniform(0.1, 40.0, size=(1024,)).astype(np.float32),
            "lidar_angles": np.linspace(-np.pi, np.pi, 1024, dtype=np.float32),
            "imu": [0.0, 0.0, 0.0, 0.0, 0.0],
            "speed": 5.0,
            "steer": 0.0,
            "accel": 0.5,
        }
        for _ in range(8)
    ]
    train_ds = DrivingDataset(dummy)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndPartialBEVNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss, ls, la = train_one_epoch(model, train_loader, optim, device)
    mae_s, mae_a = evaluate(model, train_loader, device)
    print("Example train loss:", loss, "MAE steer:", mae_s, "MAE accel:", mae_a)
    print("Architecture:\n", ARCH_DIAGRAM)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")


if __name__ == "__main__":
    example()
