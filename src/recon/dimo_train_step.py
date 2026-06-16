# src/recon/dimo_train_step.py
"""Training-step utilities for DiMo DDPM reconstruction.

This version is deliberately compatibility-first.  It merges the useful parts of
older project versions:

- legacy unconditional training on ``batch['img_target']``;
- current conditional training on ``batch['img_target_2ch']`` plus ``zf_2ch`` and
  ``mask``;
- one scalar normalization per image, so complex real/imag phase is not broken;
- support for model APIs using either ``model.T``, ``model.timesteps``,
  ``model.schedule.T``, ``model.q_sample`` or ``model.schedule.q_sample``;
- old call style: ``loss_float = train_step(model, batch, optimizer, device)``;
- new call style: ``loss, stats = train_step(model=model, batch=batch, ...)``.

Recommended current usage in ``train_dimo.py``::

    loss, stats = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        cond_mode=args.cond_mode,
    )

For conditional models, use ``cond_mode='zf'`` or ``cond_mode='zf_mask'`` and
ensure the dataset returns ``zf_2ch`` and ``mask``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # package layout
    from src.recon.dimo_dataset import DimoKspaceDataset
except Exception:  # pragma: no cover - local layout fallback
    try:
        from dimo_dataset import DimoKspaceDataset  # type: ignore
    except Exception:  # pragma: no cover
        DimoKspaceDataset = None  # type: ignore

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Optional dataloader factory retained for older scripts
# -----------------------------------------------------------------------------


def make_dataloader(
    root_dir: Union[str, Path, None] = None,
    *,
    acc_root: Union[str, Path, None] = None,
    case_dirs: Optional[list[Union[str, Path]]] = None,
    subset: str = "TrainingSet",
    acc_factor: str = "04",
    coil_type: str = "SingleCoil",
    task: str = "Mapping",
    multi_coil: bool = False,
    undersampled: bool = True,
    target_mode: str = "complex",
    cond_mode: str = "none",
    simulate_mask: bool = False,
    sim_accel: Union[int, str] = 4,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader with several historical dataset constructor styles.

    The current project usually calls the dataset with ``acc_root=...`` or
    ``case_dirs=...``.  Some old scripts used ``root_dir/subset/coil_type``.
    This helper accepts all of those forms and forwards only the arguments that
    your current ``DimoKspaceDataset`` can reasonably understand.
    """
    if DimoKspaceDataset is None:
        raise ImportError("Could not import DimoKspaceDataset")

    # If only a challenge root was provided, build the usual CMRxRecon path.
    if acc_root is None and case_dirs is None and root_dir is not None:
        root = Path(root_dir)
        coil_folder = "MultiCoil" if multi_coil else coil_type
        acc_name = str(acc_factor)
        if not acc_name.startswith("AccFactor") and acc_name != "FullSample":
            acc_name = f"AccFactor{int(acc_name):02d}" if str(acc_name).isdigit() else acc_name
        if undersampled and acc_name != "FullSample":
            acc_root = root / coil_folder / task / subset / "AccFactorXX" / acc_name
        else:
            acc_root = root / coil_folder / task / subset / "FullSample"

    kwargs: Dict[str, Any] = dict(
        acc_factor=acc_factor,
        target_mode=target_mode,
        cond_mode=cond_mode,
        simulate_mask=simulate_mask,
        sim_accel=sim_accel,
        multi_coil=multi_coil,
    )
    kwargs.update(dataset_kwargs)
    if acc_root is not None:
        kwargs["acc_root"] = acc_root
    if case_dirs is not None:
        kwargs["case_dirs"] = case_dirs

    # Drop keys that older dataset classes may not accept.
    import inspect

    sig = inspect.signature(DimoKspaceDataset.__init__)
    accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_var_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    dataset = DimoKspaceDataset(**kwargs)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


# -----------------------------------------------------------------------------
# Tensor/key helpers
# -----------------------------------------------------------------------------


def _first_present(batch: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in batch and batch[k] is not None:
            return batch[k]
    raise KeyError(f"None of the expected batch keys were found: {keys}")


def _ensure_bchw_2ch(x: Tensor, *, name: str = "tensor") -> Tensor:
    """Return a float tensor shaped [B,2,H,W]."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 3:
        # Either [2,H,W] or [B,H,W].  For 2-channel data, [2,H,W] is expected.
        if x.shape[0] == 2:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"{name} must be [2,H,W] or [B,2,H,W], got {tuple(x.shape)}")
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"{name} must be [B,2,H,W], got {tuple(x.shape)}")
    return x.float()


def _ensure_mask_b1hw(mask: Tensor, *, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
    """Return mask as float [B,1,H,W]."""
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=device).float()
    if mask.ndim == 2:  # [H,W]
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:  # [B,H,W] or [1,H,W]
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"mask channel dim must be 1, got {tuple(mask.shape)}")
    else:
        raise ValueError(f"mask must be [H,W], [B,H,W], or [B,1,H,W], got {tuple(mask.shape)}")

    if mask.shape[-2:] != (height, width):
        raise ValueError(f"mask spatial shape {tuple(mask.shape[-2:])} does not match {(height, width)}")
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"mask batch size {mask.shape[0]} does not match image batch size {batch_size}")
    return mask


def _get_T(model: Any) -> int:
    for attr in ("T", "timesteps"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    if hasattr(model, "schedule"):
        schedule = model.schedule
        for attr in ("T", "timesteps"):
            if hasattr(schedule, attr):
                return int(getattr(schedule, attr))
        if hasattr(schedule, "alpha_bars"):
            return int(schedule.alpha_bars.numel())
    raise AttributeError("Could not determine diffusion length. Expected model.T/model.timesteps/model.schedule.T.")


def _q_sample(model: Any, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """Forward diffuse using whatever API the model exposes."""
    if hasattr(model, "q_sample"):
        return model.q_sample(x0, t, noise)
    if hasattr(model, "schedule") and hasattr(model.schedule, "q_sample"):
        return model.schedule.q_sample(x0, t, noise)
    if hasattr(model, "alpha_bars"):
        alpha_bars = model.alpha_bars.to(device=x0.device, dtype=x0.dtype)
    elif hasattr(model, "schedule") and hasattr(model.schedule, "alpha_bars"):
        alpha_bars = model.schedule.alpha_bars.to(device=x0.device, dtype=x0.dtype)
    else:
        raise AttributeError("Model exposes neither q_sample nor alpha_bars.")
    a_bar = alpha_bars[t.long()].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * noise


def _autocast(device_type: str, enabled: bool):
    """Return an autocast context manager across torch versions.

    Prefers the modern ``torch.amp.autocast(device_type, ...)`` API (torch>=2.x)
    and falls back to the deprecated ``torch.cuda.amp.autocast`` if needed.
    """
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old torch
        return torch.cuda.amp.autocast(enabled=enabled)


# -----------------------------------------------------------------------------
# Conditioning and training step
# -----------------------------------------------------------------------------


def build_cond(
    batch: Mapping[str, Tensor],
    cond_mode: str,
    device: torch.device,
    *,
    scale: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Optional[Tensor]:
    """Build model conditioning from a batch.

    ``cond_mode='zf'`` returns normalized ``zf_2ch``.
    ``cond_mode='zf_mask'`` returns ``cat([normalized zf_2ch, mask], dim=1)``.

    The mask is intentionally **not** divided by ``scale``.  It is geometry, not
    intensity.  Only image-valued channels are intensity-normalized.
    """
    cond_mode = str(cond_mode or "none").lower()
    if cond_mode in {"none", "", "null", "false"}:
        return None

    # If a dataset already provides cond and the mode is not explicit, accept it.
    if cond_mode == "auto" and "cond" in batch and batch["cond"] is not None:
        cond = batch["cond"].to(device).float()
        if scale is not None and cond.shape[1] >= 2:
            cond = cond.clone()
            cond[:, :2] = cond[:, :2] / scale
        return cond

    zf = _ensure_bchw_2ch(_first_present(batch, ("zf_2ch", "x_in_2ch", "zero_filled_2ch")), name="zf_2ch").to(device)
    if scale is not None:
        zf = zf / scale

    if cond_mode in {"auto", "zf"}:
        return zf

    if cond_mode == "zf_mask":
        mask = _first_present(batch, ("mask", "sampling_mask"))
        B, _, H, W = zf.shape
        if batch_size is not None:
            B = int(batch_size)
        if height is not None and width is not None:
            H, W = int(height), int(width)
        mask_b1hw = _ensure_mask_b1hw(mask, batch_size=B, height=H, width=W, device=device)
        return torch.cat([zf, mask_b1hw], dim=1)

    raise ValueError("cond_mode must be one of: none | auto | zf | zf_mask")


def _train_step_impl(
    *,
    model: Any,
    batch: Mapping[str, Tensor],
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_mode: str = "none",
    max_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    scaler: Optional[Any] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    model.train()

    x0 = _ensure_bchw_2ch(
        _first_present(batch, ("img_target_2ch", "x_target_2ch", "x0_2ch", "img_target")),
        name="img_target_2ch",
    ).to(device)

    B, _, H, W = x0.shape

    # Single scalar per image.  This preserves real/imag relative scaling.
    scale = x0.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    x0_n = x0 / scale

    cond = build_cond(batch, cond_mode=cond_mode, device=device, scale=scale, batch_size=B, height=H, width=W)

    T = _get_T(model)
    t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
    eps = torch.randn_like(x0_n)
    x_t = _q_sample(model, x0_n, t, eps)

    # Mixed precision is only meaningful on CUDA. Keep the forward and the loss
    # inside the autocast region so the graph carries grad; the previous design
    # (forward with optimizer=None elsewhere, then backward on a detached loss)
    # is what raised "element 0 ... does not require grad".
    autocast_enabled = bool(use_amp and device.type == "cuda")
    with _autocast(device.type, autocast_enabled):
        eps_pred = model(x_t, t, cond=cond)
        loss = F.mse_loss(eps_pred, eps)

    scaler_enabled = scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)())
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        if scaler_enabled:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

    stats = {
        "loss": float(loss.detach().cpu()),
        "scale_mean": float(scale.mean().detach().cpu()),
        "scale_min": float(scale.min().detach().cpu()),
        "scale_max": float(scale.max().detach().cpu()),
        "t_mean": float(t.float().mean().detach().cpu()),
        "cond_ch": float(0 if cond is None else cond.shape[1]),
    }
    return loss.detach(), stats


def train_step(*args: Any, **kwargs: Any) -> Union[float, Tuple[Tensor, Dict[str, float]]]:
    """Run one DDPM epsilon-prediction step.

    Compatibility behavior:
    - old positional style ``train_step(model, batch, optimizer, device)`` returns
      a Python float loss;
    - keyword style returns ``(loss_tensor, stats_dict)`` unless
      ``return_stats=False`` is supplied.
    """
    legacy_call = len(args) > 0

    if legacy_call:
        names = ["model", "batch", "optimizer", "device"]
        for name, value in zip(names, args):
            kwargs.setdefault(name, value)

    return_stats = kwargs.pop("return_stats", None)
    if return_stats is None:
        return_stats = not legacy_call

    model = kwargs.pop("model")
    batch = kwargs.pop("batch")
    optimizer = kwargs.pop("optimizer", None)
    device = kwargs.pop("device", None)
    cond_mode = kwargs.pop("cond_mode", "none")
    max_grad_norm = kwargs.pop("max_grad_norm", None)
    use_amp = kwargs.pop("use_amp", False)
    scaler = kwargs.pop("scaler", None)

    if kwargs:
        # Do not silently swallow typos in training code.
        raise TypeError(f"Unexpected train_step keyword arguments: {sorted(kwargs.keys())}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    loss, stats = _train_step_impl(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        cond_mode=cond_mode,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
        scaler=scaler,
    )

    if return_stats:
        return loss, stats
    return float(stats["loss"])


__all__ = ["make_dataloader", "sample_timesteps", "build_cond", "train_step"]


def sample_timesteps(batch_size: int, T: int, device: torch.device) -> Tensor:
    return torch.randint(low=0, high=int(T), size=(int(batch_size),), device=device, dtype=torch.long)
