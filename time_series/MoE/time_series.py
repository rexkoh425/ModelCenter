"""
Time-series MoE module.

This module implements a sequential mixture-of-experts (MoE) transformer stack
that is tailored for time-series forecasting/fine-tuning on modest hardware
budgets (~16 GB). Each expert specializes in a distinct signal characteristic
and the router calls experts sequentially, deciding how much to trust each
output before passing control to the next expert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(slots=True)
class ExpertConfig:
    """Configuration for a single transformer expert."""

    name: str
    d_model: int = 256
    num_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1
    specialization_hint: str = ""
    num_questions: int = 8
    question_layers: int = 2
    question_heads: int = 4
    question_templates: Optional[List[str]] = None
    indicator_name: str = "generic_indicator"
    doubt_bias: float = 0.15


@dataclass(slots=True)
class RouterConfig:
    """Settings for the sequential router/aggregator."""

    metadata_dim: int = 16
    hidden_dim: int = 128
    precision: str = "fp16"


@dataclass(slots=True)
class ExpertForwardOutput:
    """Payload returned by each expert."""

    forecast: Tensor
    representation: Tensor
    questions: Tensor
    question_text: List[str]
    doubt_logit: Tensor
    indicator_name: str


class QuestionFormerBlock(nn.Module):
    """Single cross-attention block used by the Q-former."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        norm_queries = self.query_norm(queries)
        norm_context = self.context_norm(context)
        attn_output, _ = self.attn(norm_queries, norm_context, norm_context)
        queries = queries + attn_output
        ff_output = self.ffn(self.ffn_norm(queries))
        return queries + ff_output


class QuestionFormer(nn.Module):
    """
    Lightweight Q-former that lets each expert "ask" a small set of learned questions.

    The learned question tokens can attend to the encoded window and surface the
    pieces of evidence that are most relevant for that expert before the router
    aggregates their responses.
    """

    def __init__(
        self,
        d_model: int,
        num_questions: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_questions, d_model))
        self.blocks = nn.ModuleList(
            QuestionFormerBlock(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        )

    def forward(self, context: Tensor) -> Tensor:
        batch = context.size(0)
        queries = self.query_tokens.expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return queries


class TransformerExpert(nn.Module):
    """
    Transformer encoder that acts as an MoE expert.

    Each expert projects the raw window to a latent space, prepends a learned
    specialization token (to bias the expert toward its niche), and produces a
    horizon-length forecast plus a latent representation consumed by the router.
    """

    def __init__(self, config: ExpertConfig, window_dim: int, horizon: int) -> None:
        super().__init__()
        self.name = config.name
        self.representation_dim = config.d_model
        self.horizon = horizon
        self.num_questions = config.num_questions
        self.indicator_name = config.indicator_name
        self.doubt_bias = config.doubt_bias

        self.input_proj = nn.Linear(window_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 2,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.specialization_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.specialization_bias = nn.Parameter(torch.randn(config.d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, horizon),
        )
        self.question_former = QuestionFormer(
            d_model=config.d_model,
            num_questions=config.num_questions,
            nhead=config.question_heads,
            num_layers=config.question_layers,
            dropout=config.dropout,
        )
        self.question_text = self._build_question_text(config)
        self.doubt_head = nn.Linear(config.d_model, 1)

        if config.specialization_hint:
            # Encode the hint as a deterministic vector to keep the model size stable.
            hint_bytes = config.specialization_hint.encode("utf-8")
            with torch.no_grad():
                for i, byte in enumerate(hint_bytes[: config.d_model]):
                    self.specialization_bias[i] = (byte / 255.0) * 2 - 1

    def _build_question_text(self, config: ExpertConfig) -> List[str]:
        prompts = list(config.question_templates) if config.question_templates else []
        if len(prompts) < config.num_questions:
            base = config.specialization_hint or "time-series behavior"
            for idx in range(len(prompts), config.num_questions):
                prompts.append(f"{config.name}: how does the {base} evolve for slot {idx + 1}?")
        else:
            prompts = prompts[: config.num_questions]
        return prompts

    def forward(self, window: Tensor) -> "ExpertForwardOutput":
        """
        Args:
            window: Tensor of shape (batch, seq_len, window_dim)
        Returns:
            ExpertForwardOutput with forecast, latent representation, and question embeddings.
        """
        x = self.input_proj(window)
        spec = self.specialization_token.expand(x.size(0), -1, -1)
        x = torch.cat([spec, x], dim=1)
        encoded = self.encoder(x)
        question_embeddings = self.question_former(encoded)
        question_summary = question_embeddings.mean(dim=1)
        rep = encoded[:, 0] + question_summary + self.specialization_bias
        forecast = self.head(rep)
        doubt_logit = self.doubt_head(rep) + self.doubt_bias
        return ExpertForwardOutput(
            forecast=forecast,
            representation=rep,
            questions=question_embeddings,
            question_text=self.question_text,
            doubt_logit=doubt_logit,
            indicator_name=self.indicator_name,
        )


class SequentialRouter(nn.Module):
    """
    Router that activates experts sequentially.

    Unlike typical MoE implementations that route tokens in parallel, this
    router updates an internal hidden state after each expert call. This keeps
    VRAM usage predictable and creates a "debate" dynamic because later experts
    can respond to the partial output from earlier ones.
    """

    def __init__(
        self,
        experts: Iterable[TransformerExpert],
        router_config: RouterConfig,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(list(experts))
        if not self.experts:
            raise ValueError("At least one expert is required.")

        self.metadata_encoder = nn.Linear(router_config.metadata_dim, router_config.hidden_dim)
        self.router_gru = nn.GRUCell(
            router_config.hidden_dim,
            router_config.hidden_dim,
        )
        self.representation_projections = nn.ModuleList(
            nn.Linear(expert.representation_dim, router_config.hidden_dim)
            for expert in self.experts
        )
        self.router_gate = nn.Linear(router_config.hidden_dim, 1)
        self.router_config = router_config

    def forward(
        self,
        window: Tensor,
        metadata: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        """
        Args:
            window: (batch, seq_len, window_dim)
            metadata: (batch, metadata_dim)

        Returns:
            aggregated forecast and a log with per-expert weights/output snapshots.
        """
        device = window.device
        batch_size = window.size(0)
        metadata_dim = self.router_config.metadata_dim

        if metadata is None:
            metadata = torch.zeros(batch_size, metadata_dim, device=device)
        elif metadata.size(1) != metadata_dim:
            raise ValueError(f"metadata dim mismatch: expected {metadata_dim}, got {metadata.size(1)}")

        state = torch.tanh(self.metadata_encoder(metadata))
        aggregated = torch.zeros(batch_size, self.experts[0].horizon, device=device)
        log: List[Dict[str, Any]] = []

        for idx, expert in enumerate(self.experts):
            output = expert(window)
            projected_rep = self.representation_projections[idx](output.representation)
            state = self.router_gru(projected_rep, state)
            weight = torch.sigmoid(self.router_gate(state))
            aggregated = aggregated + weight * output.forecast
            log.append(
                {
                    "expert_name": expert.name,
                    "weight": weight.detach(),
                    "forecast": output.forecast.detach(),
                    "questions": output.questions.detach(),
                    "question_text": output.question_text,
                    "doubt_prob": torch.sigmoid(output.doubt_logit).detach(),
                    "indicator_name": output.indicator_name,
                }
            )

        return aggregated, log


class TimeSeriesMoE(nn.Module):
    """High-level wrapper that combines the experts + router."""

    def __init__(
        self,
        expert_configs: Iterable[ExpertConfig],
        window_dim: int,
        horizon: int,
        router_config: RouterConfig = RouterConfig(),
    ) -> None:
        super().__init__()
        experts = [
            TransformerExpert(cfg, window_dim=window_dim, horizon=horizon)
            for cfg in expert_configs
        ]
        self.router = SequentialRouter(experts, router_config=router_config)

    def forward(
        self,
        window: Tensor,
        metadata: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        return self.router(window, metadata)


class TimeSeriesWindowDataset(Dataset):
    """
    Simple dataset that slices continuous series into (input_window, target) pairs.
    """

    def __init__(
        self,
        series: Tensor,
        window_size: int,
        horizon: int,
        metadata: Optional[Tensor] = None,
    ) -> None:
        if series.ndim != 2:
            raise ValueError("series must have shape (total_length, feature_dim)")
        total_length, feature_dim = series.shape
        limit = total_length - window_size - horizon + 1
        if limit <= 0:
            raise ValueError("series is too short for the requested window/horizon")

        self.series = series
        self.window_size = window_size
        self.horizon = horizon
        self.feature_dim = feature_dim
        self.limit = limit
        self.metadata = metadata

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        start = idx
        end = start + self.window_size
        window = self.series[start:end]
        target = self.series[end : end + self.horizon, 0]
        meta = None
        if self.metadata is not None:
            meta = self.metadata[idx]
        return window, target, meta


@dataclass(slots=True)
class DebateCritique:
    """Represents a single expert's challenge during the debate loop."""

    expert_name: str
    indicator_name: str
    doubt: float
    stance: str
    suggested_forecast: Tensor


class StockPredictor(nn.Module):
    """
    Main stock predictor that proposes a price and rationale before experts debate it.
    """

    def __init__(
        self,
        input_dim: int,
        metadata_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.meta_proj = nn.Linear(metadata_dim, hidden_dim)
        self.price_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.explainer_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, window: Tensor, metadata: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        encoded, _ = self.encoder(window)
        latent = encoded[:, -1]
        if metadata is not None:
            latent = latent + torch.tanh(self.meta_proj(metadata))
        price = self.price_head(latent)
        rationale = self.explainer_head(latent)
        return price, rationale


class StockConsensusSystem:
    """
    Coordinates the main stock predictor and MoE experts until they reach consensus.
    """

    def __init__(
        self,
        main_model: StockPredictor,
        moe_model: TimeSeriesMoE,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.main_model = main_model.to(device)
        self.moe_model = moe_model.to(device)

    def run_debate(
        self,
        window: Tensor,
        metadata: Optional[Tensor] = None,
        max_rounds: int = 4,
        doubt_threshold: float = 0.35,
        consensus_tolerance: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Executes the debate until experts agree that the main prediction is within tolerance.

        Args:
            window: Stock window tensor.
            metadata: Indicator tensor (optional).
            max_rounds: Maximum debate rounds before forcing consensus.
            doubt_threshold: Probability threshold for experts to raise doubts.
            consensus_tolerance: Allowed price difference (absolute) between experts and base model.
        """
        self.main_model.eval()
        self.moe_model.eval()

        if window.ndim == 2:
            window = window.unsqueeze(0)
        window = window.to(self.device)
        metadata = metadata.to(self.device) if metadata is not None else None

        debate_log: List[Dict[str, Any]] = []
        prediction, rationale = self.main_model(window, metadata)
        consensus = False
        dynamic_threshold = doubt_threshold
        dynamic_tolerance = consensus_tolerance
        round_count = 0
        last_expert_logs: List[Dict[str, Any]] = []

        while round_count < max_rounds:
            moe_forecast, expert_logs = self.moe_model(window, metadata)
            last_expert_logs = expert_logs
            critiques = self._collect_critiques(
                expert_logs=expert_logs,
                doubt_threshold=dynamic_threshold,
                prediction=prediction,
                consensus_tolerance=dynamic_tolerance,
            )
            explanation = self._compose_explanation(round_count, prediction, rationale, critiques)
            consensus = len(critiques) == 0
            debate_log.append(
                {
                    "round": round_count + 1,
                    "prediction": prediction.detach().cpu(),
                    "explanation": explanation,
                    "critiques": [
                        {
                            "expert": c.expert_name,
                            "indicator": c.indicator_name,
                            "doubt": c.doubt,
                            "stance": c.stance,
                        }
                        for c in critiques
                    ],
                    "consensus": consensus,
                }
            )
            round_count += 1
            if consensus:
                break
            prediction = self._revise_prediction(prediction, critiques)
            rationale = rationale + torch.randn_like(rationale) * 0.05  # encourage new rationales
            dynamic_threshold = min(0.95, dynamic_threshold + 0.1)
            dynamic_tolerance *= 1.25

        if not consensus:
            prediction = self._force_consensus(prediction, last_expert_logs)
            explanation = (
                "Forced consensus: averaged expert proposals with the main call after sustained disagreement."
            )
            debate_log.append(
                {
                    "round": round_count + 1,
                    "prediction": prediction.detach().cpu(),
                    "explanation": explanation,
                    "critiques": [],
                    "consensus": True,
                }
            )
            consensus = True

        return {
            "final_prediction": prediction.detach().cpu(),
            "consensus": consensus,
            "rounds": debate_log,
        }

    def _collect_critiques(
        self,
        expert_logs: List[Dict[str, Any]],
        doubt_threshold: float,
        prediction: Tensor,
        consensus_tolerance: float,
    ) -> List[DebateCritique]:
        critiques: List[DebateCritique] = []
        for log in expert_logs:
            doubt = float(log["doubt_prob"].mean().item())
            if doubt < doubt_threshold:
                continue
            target = float(log["forecast"][:, 0].mean().item())
            base = float(prediction.squeeze(-1).mean().item())
            if abs(target - base) <= consensus_tolerance:
                continue
            stance = (
                f"{log['expert_name']} ({log['indicator_name']}) prefers {target:.2f} vs base {base:.2f} "
                f"({doubt*100:.1f}% doubt)"
            )
            critiques.append(
                DebateCritique(
                    expert_name=log["expert_name"],
                    indicator_name=log["indicator_name"],
                    doubt=doubt,
                    stance=stance,
                    suggested_forecast=log["forecast"][:, :1],
                )
            )
        # Encourage frequent doubts by ensuring at least one expert speaks up.
        if not critiques and expert_logs:
            top_log = max(expert_logs, key=lambda item: float(item["doubt_prob"].mean().item()))
            target = float(top_log["forecast"][:, 0].mean().item())
            base = float(prediction.squeeze(-1).mean().item())
            if abs(target - base) <= consensus_tolerance:
                return critiques
            stance = (
                f"{top_log['expert_name']} ({top_log['indicator_name']}) cautiously challenges "
                f"the call, preferring {target:.2f} vs base {base:.2f}."
            )
            critiques.append(
                DebateCritique(
                    expert_name=top_log["expert_name"],
                    indicator_name=top_log["indicator_name"],
                    doubt=float(top_log["doubt_prob"].mean().item()),
                    stance=stance,
                    suggested_forecast=top_log["forecast"][:, :1],
                )
            )
        return critiques

    def _revise_prediction(self, prediction: Tensor, critiques: List[DebateCritique]) -> Tensor:
        aggregate = prediction.clone()
        if not critiques:
            return aggregate
        weights = torch.tensor([c.doubt for c in critiques], device=prediction.device).unsqueeze(-1)
        suggested = torch.stack([c.suggested_forecast.to(prediction.device) for c in critiques], dim=0)
        weighted = (weights * suggested).sum(dim=0) / (weights.sum(dim=0) + 1e-6)
        aggregate = 0.5 * prediction + 0.5 * weighted
        return aggregate

    def _force_consensus(self, prediction: Tensor, expert_logs: List[Dict[str, Any]]) -> Tensor:
        if not expert_logs:
            return prediction
        forecasts = torch.stack([log["forecast"][:, :1] for log in expert_logs], dim=0).to(prediction.device)
        averaged = forecasts.mean(dim=0)
        return 0.5 * prediction + 0.5 * averaged

    def _compose_explanation(
        self,
        round_idx: int,
        prediction: Tensor,
        rationale: Tensor,
        critiques: List[DebateCritique],
    ) -> str:
        base = float(prediction.mean().item())
        rationale_stats = ", ".join(f"{val:.2f}" for val in rationale.flatten().tolist())
        if critiques:
            critique_summary = "; ".join(c.stance for c in critiques)
        else:
            critique_summary = "All experts align with the base outlook."
        return (
            f"Round {round_idx + 1}: base model targets price {base:.2f}. "
            f"Signal factors [{rationale_stats}] informed the call. {critique_summary}"
        )


def fine_tune(
    model: TimeSeriesMoE,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    device: str = "cuda",
) -> None:
    """
    Lightweight fine-tuning loop. Works on GPU or CPU and fits within 16 GB
    when window sizes/batch sizes are moderate (e.g., batch <= 64, d_model<=384).
    """

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=model.router.router_config.precision == "fp16")
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for window, target, meta in dataloader:
            window = window.to(device)
            target = target.to(device)
            meta = meta.to(device) if meta is not None else None

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=model.router.router_config.precision == "fp16"):
                forecast, _ = model(window, meta)
                loss = loss_fn(forecast, target)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * window.size(0)

        avg = running_loss / len(dataloader.dataset)
        print(f"epoch={epoch+1} loss={avg:.4f}")


def train_stock_predictor(
    model: StockPredictor,
    dataloader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """Train the main stock predictor that the MoE experts will challenge."""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for window, target, meta in dataloader:
            window = window.to(device)
            target_price = target[:, 0].unsqueeze(-1).to(device)
            meta = meta.to(device) if meta is not None else None
            optimizer.zero_grad()
            pred, _ = model(window, meta)
            loss = loss_fn(pred, target_price)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * window.size(0)
        avg = running_loss / len(dataloader.dataset)
        print(f"[StockPredictor] epoch={epoch+1} loss={avg:.4f}")


def build_default_time_series_model(
    window_dim: int,
    horizon: int,
) -> TimeSeriesMoE:
    """
    Factory that builds a 3-expert stack specialized for trend/seasonality/anomalies.
    """

    expert_configs = [
        ExpertConfig(
            name="trend_expert",
            specialization_hint="low_frequency_trend",
            d_model=256,
            num_layers=2,
            indicator_name="trend_sma",
            doubt_bias=0.25,
        ),
        ExpertConfig(
            name="seasonal_expert",
            specialization_hint="periodic_patterns",
            d_model=256,
            num_layers=2,
            indicator_name="momentum_cycle",
            doubt_bias=0.2,
        ),
        ExpertConfig(
            name="anomaly_expert",
            specialization_hint="residual_outliers",
            d_model=192,
            num_layers=1,
            indicator_name="volatility_shocks",
            doubt_bias=0.15,
        ),
    ]
    router_config = RouterConfig(metadata_dim=16, hidden_dim=192, precision="fp16")
    return TimeSeriesMoE(expert_configs, window_dim=window_dim, horizon=horizon, router_config=router_config)


def _generate_synthetic_series(length: int = 2000) -> Tensor:
    """Utility that fabricates a toy stock-like series with price + volume."""

    time = torch.arange(length, dtype=torch.float32)
    drift = 0.0004 * time
    daily_cycle = 0.8 * torch.sin(2 * torch.pi * time / 24)
    weekly_cycle = 0.4 * torch.sin(2 * torch.pi * time / (24 * 5))
    noise = 0.05 * torch.randn(length)
    price = 50 + drift + daily_cycle + weekly_cycle + noise
    volume = 100 + 10 * torch.cos(2 * torch.pi * time / 12) + 5 * torch.randn(length)
    series = torch.stack([price, volume], dim=-1)
    return series


def _build_stock_metadata(series: Tensor, dim: int = 16) -> Tensor:
    """Compute lightweight indicators (SMA ratios, momentum, volatility) used by experts."""

    if series.ndim != 2:
        raise ValueError("series must have shape (length, feature_dim)")
    prices = series[:, 0]
    length = prices.numel()
    meta = torch.zeros(length, dim)

    def _moving_average(signal: Tensor, window: int) -> Tensor:
        kernel = torch.ones(1, 1, window, device=signal.device) / window
        padded = F.pad(signal.view(1, 1, -1), (window - 1, 0), mode="replicate")
        return F.conv1d(padded, kernel).view(-1)

    sma_short = _moving_average(prices, 5)
    sma_long = _moving_average(prices, 20)
    momentum = torch.zeros_like(prices)
    momentum[1:] = prices[1:] - prices[:-1]
    volatility = torch.zeros_like(prices)
    window = 10
    padded = torch.cat([prices[:1].repeat(window - 1), prices])
    for idx in range(length):
        slice_ = padded[idx : idx + window]
        volatility[idx] = slice_.std()

    meta[:, 0] = (prices - prices.mean()) / prices.std().clamp_min(1e-6)
    meta[:, 1] = (momentum - momentum.mean()) / momentum.std().clamp_min(1e-6)
    meta[:, 2] = (sma_short - prices) / prices.clamp_min(1e-6)
    meta[:, 3] = (sma_long - prices) / prices.clamp_min(1e-6)
    meta[:, 4] = (volatility - volatility.mean()) / volatility.std().clamp_min(1e-6)
    meta[:, 5] = torch.linspace(0, 1, steps=length)
    if dim > 6:
        meta[:, 6] = torch.sin(torch.arange(length) / 12.0)
        meta[:, 7] = torch.cos(torch.arange(length) / 6.0)
    return meta


def main() -> None:
    """Example end-to-end usage."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window_size = 64
    horizon = 1
    series = _generate_synthetic_series(length=4096)
    metadata = _build_stock_metadata(series, dim=16)

    dataset = TimeSeriesWindowDataset(series, window_size=window_size, horizon=horizon, metadata=metadata)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    moe_model = build_default_time_series_model(window_dim=series.shape[1], horizon=horizon)
    fine_tune(
        model=moe_model,
        dataloader=dataloader,
        epochs=3,
        lr=3e-4,
        grad_clip=1.0,
        device=device,
    )

    main_model = StockPredictor(input_dim=series.shape[1], metadata_dim=metadata.shape[1])
    train_stock_predictor(
        model=main_model,
        dataloader=dataloader,
        epochs=3,
        lr=1e-3,
        device=device,
    )

    consensus_system = StockConsensusSystem(main_model=main_model, moe_model=moe_model, device=device)
    sample_window, _, sample_meta = dataset[200]
    sample_window = sample_window.unsqueeze(0)
    sample_meta = sample_meta.unsqueeze(0) if sample_meta is not None else None
    debate = consensus_system.run_debate(
        window=sample_window,
        metadata=sample_meta,
        max_rounds=5,
        doubt_threshold=0.3,
        consensus_tolerance=0.75,
    )
    print(f"Consensus achieved: {debate['consensus']}")
    for round_info in debate["rounds"]:
        print(round_info["explanation"])
        if round_info["critiques"]:
            for critique in round_info["critiques"]:
                print(f"   - {critique['stance']}")
    flat_prediction = debate["final_prediction"].view(-1).tolist()
    if len(flat_prediction) == 1:
        print(f"Final price prediction: {flat_prediction[0]:.2f}")
    else:
        print("Final price prediction vector:", ", ".join(f"{val:.2f}" for val in flat_prediction))


if __name__ == "__main__":
    main()
