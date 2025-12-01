"""
Comprehensive hourly swing trading model with TCN + Transformer + sparse MoE head.
Includes dataset definitions, model architecture, and a full training loop example.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------
class HourlySwingDataset(Dataset):
    """Simple dataset storing (sequence, target) pairs for hourly swing trading."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            inputs: Tensor of shape (num_samples, seq_len, feature_dim)
            targets: Tensor of shape (num_samples, target_dim) for regression
                     or (num_samples,) for classification.
        """
        super().__init__()
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have matching number of samples")
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# -----------------------------------------------------------------------------
# Positional encoding
# -----------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds maximum positional encoding length.")
        return x + self.pe[:, : x.size(1)].to(x.dtype)


# -----------------------------------------------------------------------------
# Temporal Convolutional Network blocks
# -----------------------------------------------------------------------------
class TCNBlock(nn.Module):
    """Single residual TCN block with dilated Conv1d."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out + residual


class TCNStack(nn.Module):
    """Stack of TCN blocks."""

    def __init__(self, d_model: int, kernel_sizes: List[int], dilations: List[int]) -> None:
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must be the same length")
        blocks = []
        for k, d in zip(kernel_sizes, dilations):
            blocks.append(TCNBlock(d_model, k, d))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# -----------------------------------------------------------------------------
# Transformer encoder layers (standard and MoE variant)
# -----------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer with pre-norm architecture."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ffn = self.linear2(self.dropout2(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ffn))
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x, aux_loss


class MoEBlock(nn.Module):
    """Sparse mixture-of-experts block with top-2 gating and load balancing."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        hidden_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x.shape
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(gate_probs, k=2, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        expert_mask = torch.zeros_like(gate_probs)
        expert_mask.scatter_(-1, topk_idx, topk_vals)

        x_flat = x.reshape(-1, d_model)
        expert_outputs: List[torch.Tensor] = []
        for expert in self.experts:
            out = expert(x_flat).reshape(bsz, seq_len, d_model)
            expert_outputs.append(out)

        moe_out = torch.zeros_like(x)
        for idx, expert_out in enumerate(expert_outputs):
            weight = expert_mask[..., idx].unsqueeze(-1)
            moe_out = moe_out + weight * expert_out
        moe_out = self.dropout(moe_out)

        target_usage = torch.full(
            (self.num_experts,), 1.0 / self.num_experts, device=x.device, dtype=x.dtype
        )
        load_loss = F.mse_loss(gate_probs.mean(dim=(0, 1)), target_usage)
        return moe_out, load_loss


class TransformerEncoderLayerMoE(nn.Module):
    """Transformer encoder layer whose feed-forward block is replaced with a sparse MoE."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        moe_hidden_dim: int = 384,
        dropout: float = 0.1,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.moe_block = MoEBlock(
            d_model=d_model,
            num_experts=num_experts,
            hidden_dim=moe_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        moe_out, load_loss = self.moe_block(x)
        x = self.norm2(x + self.dropout(moe_out))
        return x, load_loss


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class HourlySwingModel(nn.Module):
    """Main trading model combining TCN, Transformer, and sparse MoE head."""

    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        task_type: str = "regression",
        d_model: int = 192,
        n_heads: int = 4,
        transformer_layers: int = 4,
    ) -> None:
        super().__init__()
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")
        self.seq_len = seq_len
        self.task_type = task_type
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.tcn = TCNStack(
            d_model=d_model,
            kernel_sizes=[3, 5, 7],
            dilations=[1, 2, 4],
        )
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        layers: List[nn.Module] = []
        for idx in range(transformer_layers):
            if idx < transformer_layers - 1:
                layers.append(
                    TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=512,
                        dropout=0.1,
                    )
                )
            else:
                layers.append(
                    TransformerEncoderLayerMoE(
                        d_model=d_model,
                        nhead=n_heads,
                        moe_hidden_dim=384,
                        dropout=0.1,
                        num_experts=4,
                    )
                )
        self.transformer_layers = nn.ModuleList(layers)

        if task_type == "regression":
            self.head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 3),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, _ = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {seq_len}")

        x = self.input_proj(x)  # (B, S, d_model)
        x = x.transpose(1, 2)  # (B, d_model, S)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (B, S, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, S+1, d_model)
        x = self.positional_encoding(x)

        total_moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for layer in self.transformer_layers:
            x, aux_loss = layer(x)
            total_moe_loss = total_moe_loss + aux_loss

        cls_repr = x[:, 0, :]  # (B, d_model)
        logits = self.head(cls_repr)
        return logits, total_moe_loss


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    seq_len: int
    feature_dim: int
    task_type: str
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 3
    lambda_moe: float = 0.01
    warmup_steps: int = 200
    min_lr: float = 1e-5
    total_steps: int = 0
    num_workers: int = 4
    grad_clip: float = 1.0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path: Optional[str] = None


class WarmupCosineScheduler:
    """Cosine decay scheduler with linear warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if self._step <= self.warmup_steps:
                lr = self.min_lr + (base_lr - self.min_lr) * self._step / self.warmup_steps
            else:
                progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            group["lr"] = lr


def train_one_epoch(
    model: HourlySwingModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainingConfig,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            preds, moe_loss = model(inputs)
            if config.task_type == "regression":
                targets = targets.view(targets.size(0), -1).float()
                preds = preds.view_as(targets)
                main_loss = criterion(preds, targets)
            else:
                targets = targets.long()
                main_loss = criterion(preds, targets)
            total_loss = main_loss + config.lambda_moe * moe_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += total_loss.item()
        if (step + 1) % max(1, len(dataloader) // 5) == 0:
            avg_loss = running_loss / (step + 1)
            print(f"  Step {step + 1}/{len(dataloader)} - Loss: {avg_loss:.4f}")

    return running_loss / max(1, len(dataloader))


def evaluate(
    model: HourlySwingModel,
    dataloader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    metric_total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds, moe_loss = model(inputs)
            if config.task_type == "regression":
                targets = targets.view(targets.size(0), -1).float()
                preds = preds.view_as(targets)
                main_loss = criterion(preds, targets)
                metric_total += F.l1_loss(preds, targets, reduction="sum").item()
                count += targets.size(0)
            else:
                targets = targets.long()
                main_loss = criterion(preds, targets)
                preds_labels = preds.argmax(dim=-1)
                metric_total += (preds_labels == targets).sum().item()
                count += targets.numel()
            total_loss += (main_loss + config.lambda_moe * moe_loss).item()

    avg_loss = total_loss / max(1, len(dataloader))
    if config.task_type == "regression":
        metric = metric_total / max(1, count)
        metric_name = "val_mae"
    else:
        metric = metric_total / max(1, count)
        metric_name = "val_accuracy"
    return {"val_loss": avg_loss, metric_name: metric}


def train_model(
    model: HourlySwingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> None:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.total_steps or config.max_epochs * max(1, len(train_loader))
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.MSELoss() if config.task_type == "regression" else nn.CrossEntropyLoss()

    best_metric = float("inf") if config.task_type == "regression" else -float("inf")
    best_state = None

    for epoch in range(1, config.max_epochs + 1):
        print(f"Epoch {epoch}/{config.max_epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, config, criterion
        )
        val_metrics = evaluate(model, val_loader, device, config, criterion)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        if config.task_type == "regression":
            print(f"  Val MAE: {val_metrics['val_mae']:.4f}")
            current_metric = -val_metrics["val_mae"]
        else:
            print(f"  Val Accuracy: {val_metrics['val_accuracy'] * 100:.2f}%")
            current_metric = val_metrics["val_accuracy"]

        if current_metric > best_metric:
            best_metric = current_metric
            best_state = model.state_dict()
            if config.checkpoint_path:
                torch.save(best_state, config.checkpoint_path)
                print(f"  Saved checkpoint to {config.checkpoint_path}")

    if best_state is not None and not config.checkpoint_path:
        print("Training complete; best model retained in memory.")


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
def example_usage() -> None:
    torch.manual_seed(42)
    seq_len = 256
    feature_dim = 32
    num_samples = 512

    inputs = torch.randn(num_samples, seq_len, feature_dim)
    regression_targets = torch.randn(num_samples, 1)
    classification_targets = torch.randint(0, 3, (num_samples,))

    split = int(0.8 * num_samples)
    train_inputs, val_inputs = inputs[:split], inputs[split:]
    train_targets_reg, val_targets_reg = regression_targets[:split], regression_targets[split:]
    train_targets_cls, val_targets_cls = classification_targets[:split], classification_targets[split:]

    # Regression example
    print("\n=== Regression Task ===")
    train_dataset_reg = HourlySwingDataset(train_inputs, train_targets_reg)
    val_dataset_reg = HourlySwingDataset(val_inputs, val_targets_reg)
    train_loader_reg = DataLoader(
        train_dataset_reg,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    val_loader_reg = DataLoader(
        val_dataset_reg,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    reg_config = TrainingConfig(
        seq_len=seq_len,
        feature_dim=feature_dim,
        task_type="regression",
        batch_size=16,
        max_epochs=2,
        warmup_steps=20,
        lambda_moe=0.01,
        num_workers=0,
    )
    reg_config.total_steps = reg_config.max_epochs * max(1, len(train_loader_reg))
    reg_model = HourlySwingModel(seq_len=seq_len, feature_dim=feature_dim, task_type="regression")
    train_model(reg_model, train_loader_reg, val_loader_reg, reg_config)

    # Classification example
    print("\n=== Classification Task ===")
    train_dataset_cls = HourlySwingDataset(train_inputs, train_targets_cls)
    val_dataset_cls = HourlySwingDataset(val_inputs, val_targets_cls)
    train_loader_cls = DataLoader(
        train_dataset_cls,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    val_loader_cls = DataLoader(
        val_dataset_cls,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    cls_config = TrainingConfig(
        seq_len=seq_len,
        feature_dim=feature_dim,
        task_type="classification",
        batch_size=16,
        max_epochs=2,
        warmup_steps=20,
        lambda_moe=0.01,
        num_workers=0,
    )
    cls_config.total_steps = cls_config.max_epochs * max(1, len(train_loader_cls))
    cls_model = HourlySwingModel(seq_len=seq_len, feature_dim=feature_dim, task_type="classification")
    train_model(cls_model, train_loader_cls, val_loader_cls, cls_config)


if __name__ == "__main__":
    example_usage()
