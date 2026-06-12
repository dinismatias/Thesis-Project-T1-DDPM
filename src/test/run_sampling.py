# src/test/run_sampling.py
"""
Canonical Stage-2 sampling script for the T1_DDPM / DiMo thesis pipeline.

This merges the useful capabilities from the historical run_sampling variants:
- legacy CLI: --root + --acc;
- current CLI: --acc_root;
- conditional checkpoints: cond_mode none/zf/zf_mask/auto;
- checkpoint schemas: model_state/state_dict/model/config/args;
- init modes: noise, zf, pt, stage1;
- Stage-1 bundle dumping;
- DC modes: replace/hard, grad/soft, cg, none;
- residual logging;
- scale-aware inference to match training normalization.

Important output keys in stage2_recon_output.pt:
- x_rec_2ch: final Stage-2 reconstruction, original units;
- x_init_2ch: Stage-2 initialization, original units if used;
- x_in_2ch / zf_2ch: zero-filled input, original units;
- x_target_2ch: target/reference if available;
- y_2ch: undersampled k-space, original units;
- mask: sampling mask;
- scale_2ch: scalar scale used internally for normalized diffusion/DC.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

try:
    from src.recon.dimo_model import DiMoDDPM
    try:
        from src.recon.dimo_model import load_model as _repo_load_model  # type: ignore
    except Exception:
        _repo_load_model = None
    from src.recon.encoding import SenseOp
    from src.utils.complex_ops import twoch_to_complex as ch2_to_complex
    from src.utils.complex_ops import complex_to_twoch as complex_to_2ch
    from src.recon.dimo_sample import ddim_with_dc_from_model
    from src.recon.dimo_dataset import DimoKspaceDataset
except Exception:  # pragma: no cover - fallback for direct execution layouts
    from dimo_model import DiMoDDPM  # type: ignore
    try:
        from dimo_model import load_model as _repo_load_model  # type: ignore
    except Exception:
        _repo_load_model = None
    from encoding import SenseOp  # type: ignore
    from complex_ops import twoch_to_complex as ch2_to_complex  # type: ignore
    from complex_ops import complex_to_twoch as complex_to_2ch  # type: ignore
    from dimo_sample import ddim_with_dc_from_model  # type: ignore
    from dimo_dataset import DimoKspaceDataset  # type: ignore


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _namespace_to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        return dict(vars(x))
    return {}


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(str(k).startswith("module.") for k in state.keys()):
        return {str(k).replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_state_and_config(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError("Checkpoint model state is not a state_dict-like dictionary")

    cfg = _namespace_to_dict(ckpt.get("config"))
    legacy_args = _namespace_to_dict(ckpt.get("args"))
    return _strip_module_prefix(state), cfg, legacy_args, ckpt


def _infer_timesteps(state: Dict[str, torch.Tensor], cfg: Dict[str, Any], legacy_args: Dict[str, Any], fallback: int) -> int:
    for k in ("schedule.alpha_bars", "schedule.betas", "alpha_bars", "betas"):
        v = state.get(k)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    for src in (cfg, legacy_args):
        for k in ("timesteps", "T"):
            if k in src and src[k] is not None:
                return int(src[k])
    return int(fallback)


def _infer_cond_mode(user_cond_mode: str, cfg: Dict[str, Any], legacy_args: Dict[str, Any]) -> str:
    user_cond_mode = str(user_cond_mode).lower()
    if user_cond_mode != "auto":
        return user_cond_mode
    for src in (cfg, legacy_args):
        v = src.get("cond_mode")
        if isinstance(v, str) and v.lower() in {"none", "zf", "zf_mask"}:
            return v.lower()
    cond_ch = cfg.get("cond_ch", legacy_args.get("cond_ch", None))
    if str(cond_ch) == "2":
        return "zf"
    if str(cond_ch) == "3":
        return "zf_mask"
    return "none"


def _infer_cond_ch(cond_mode: str, user_cond_ch: Optional[int], cfg: Dict[str, Any], legacy_args: Dict[str, Any], state: Dict[str, torch.Tensor]) -> int:
    if user_cond_ch is not None:
        return int(user_cond_ch)
    for src in (cfg, legacy_args):
        v = src.get("cond_ch")
        if v is not None:
            return int(v)
    # Best-effort weight-based inference: first conv often has input channels data_ch+cond_ch.
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4 and ("conv" in k.lower() or "net" in k.lower() or "down" in k.lower()):
            in_ch = int(v.shape[1])
            if in_ch in {2, 4, 5}:
                return max(0, in_ch - 2)
    if cond_mode == "none":
        return 0
    if cond_mode == "zf":
        return 2
    if cond_mode == "zf_mask":
        return 3
    raise ValueError(f"Unknown cond_mode: {cond_mode}")


def _infer_acc_factor(acc_root: Optional[str], acc: Optional[str], cfg: Dict[str, Any], legacy_args: Dict[str, Any], user_acc_factor: Optional[str]) -> str:
    if user_acc_factor:
        return str(user_acc_factor).zfill(2)[-2:]
    for src in (cfg, legacy_args):
        if src.get("acc_factor") is not None:
            return str(src["acc_factor"]).zfill(2)[-2:]
    source = acc or (Path(acc_root).name if acc_root else "")
    digits = "".join(c for c in str(source) if c.isdigit())
    if digits:
        return digits[-2:].zfill(2)
    return "04"


def _ensure_bchw_2ch(x: torch.Tensor, *, name: str = "tensor") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 2:
        # real image [H,W]
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
    elif x.ndim == 3:
        # [2,H,W] or [B,H,W] real batch
        if x.shape[0] == 2:
            x = x.unsqueeze(0)
        else:
            x = torch.cat([x.unsqueeze(1), torch.zeros_like(x.unsqueeze(1))], dim=1)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"{name}: unsupported shape {tuple(x.shape)}")
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"{name}: expected [B,2,H,W], got {tuple(x.shape)}")
    return x.float()


def _get_first(item: Dict[str, Any], keys: Iterable[str], *, required: bool = True) -> Optional[Any]:
    for k in keys:
        if k in item and item[k] is not None:
            return item[k]
    if required:
        raise KeyError(f"None of the expected keys were found: {list(keys)}. Available keys: {list(item.keys())}")
    return None


def _mask_to_bhw(mask: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = mask.to(device=device).float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch_size != 1:
        mask = mask.expand(batch_size, -1, -1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"mask batch {mask.shape[0]} does not match batch size {batch_size}")
    return (mask > 0.5).float()


def _scale_from_2ch(x_2ch: torch.Tensor, *, scale_min: float = 1e-8) -> torch.Tensor:
    x_2ch = _ensure_bchw_2ch(x_2ch)
    mag = torch.sqrt(x_2ch[:, 0] ** 2 + x_2ch[:, 1] ** 2)
    s = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(float(scale_min))
    return s.unsqueeze(1)  # [B,1,1,1]


def _get_alpha_bars(model: Any, device: torch.device) -> torch.Tensor:
    if hasattr(model, "alpha_bars"):
        return model.alpha_bars.to(device)
    if hasattr(model, "schedule") and hasattr(model.schedule, "alpha_bars"):
        return model.schedule.alpha_bars.to(device)
    raise AttributeError("Model exposes neither alpha_bars nor schedule.alpha_bars")


def _call_model(model: Any, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
    if cond is None:
        try:
            return model(x, t)
        except TypeError:
            return model(x, t, cond=None)
    return model(x, t, cond=cond)


@torch.no_grad()
def _ddim_img2img_no_dc(
    model: Any,
    x_init_2ch: torch.Tensor,
    *,
    cond: Optional[torch.Tensor],
    strength: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Conditional-aware Stage-1 DDIM/img2img denoise without data consistency."""
    model.eval().to(device)
    x_init_2ch = _ensure_bchw_2ch(x_init_2ch, name="x_init_2ch").to(device)
    if cond is not None:
        cond = cond.to(device)

    alpha_bars = _get_alpha_bars(model, device=device)
    T = int(getattr(model, "T", getattr(model, "timesteps", alpha_bars.shape[0])))
    t_start = int(round(float(strength) * (T - 1)))
    t_start = max(0, min(T - 1, t_start))

    t_seq = torch.linspace(t_start, 0, steps=max(2, int(num_steps)), device=device).round().long()
    t_seq = torch.unique_consecutive(t_seq)
    if int(t_seq[-1]) != 0:
        t_seq = torch.cat([t_seq, torch.zeros(1, dtype=torch.long, device=device)])

    eps0 = torch.randn_like(x_init_2ch)
    a0 = alpha_bars[t_start].view(1, 1, 1, 1)
    x_t = torch.sqrt(a0) * x_init_2ch + torch.sqrt(1.0 - a0) * eps0

    B = x_init_2ch.shape[0]
    for i in range(len(t_seq) - 1):
        t = int(t_seq[i].item())
        t_prev = int(t_seq[i + 1].item())
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = _call_model(model, x_t, t_batch, cond)
        a_t = alpha_bars[t].view(1, 1, 1, 1)
        a_prev = alpha_bars[t_prev].view(1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t).clamp_min(1e-8)
        x_t = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps_pred
    return x_t


def _load_init_tensor(init_pt: str, init_key: Optional[str], device: torch.device) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    obj = torch.load(init_pt, map_location="cpu")
    if isinstance(obj, dict):
        candidate_keys: List[str] = []
        if init_key:
            candidate_keys.append(init_key)
        candidate_keys.extend(["x_out_2ch", "x_rec_2ch", "x_init_2ch", "x_in_2ch", "zf_2ch", "img_target_2ch", "x_target_2ch"])
        seen = set()
        ordered = []
        for k in candidate_keys:
            if k not in seen:
                ordered.append(k)
                seen.add(k)
        for k in ordered:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return _ensure_bchw_2ch(obj[k], name=f"init_pt[{k}]").to(device), k, obj
        raise KeyError(f"No usable init tensor found in {init_pt}. Tried {ordered}. Available keys: {list(obj.keys())}")
    if isinstance(obj, torch.Tensor):
        return _ensure_bchw_2ch(obj, name="init_pt").to(device), "<tensor>", {"tensor": obj}
    raise ValueError(f"Unsupported init_pt payload type: {type(obj)}")


def _adjoint(sense_op: Any, y_complex: torch.Tensor, mask_bhw: torch.Tensor) -> torch.Tensor:
    try:
        return sense_op.adjoint(y_complex)
    except TypeError:
        return sense_op.adjoint(y_complex, mask_bhw)


def _make_dataset(args: argparse.Namespace, cfg: Dict[str, Any], legacy_args: Dict[str, Any], acc_factor: str, target_mode: str, cond_mode: str) -> Any:
    acc_root = args.acc_root
    if acc_root is None:
        if args.root is None:
            raise ValueError("Provide either --acc_root or legacy --root + --acc")
        acc_root = str(Path(args.root) / args.acc)

    sig = inspect.signature(DimoKspaceDataset.__init__)
    accepted = set(sig.parameters.keys())
    kwargs: Dict[str, Any] = {}

    def maybe(name: str, value: Any) -> None:
        if name in accepted:
            kwargs[name] = value

    maybe("acc_root", acc_root)
    maybe("root", args.root if args.root is not None else acc_root)
    maybe("acc_factor", acc_factor)
    maybe("target_mode", target_mode)
    maybe("cond_mode", cond_mode)
    maybe("multi_coil", False)
    maybe("use_full_as_target", True)

    maybe("simulate_mask", bool(args.simulate_mask))
    maybe("sim_accel", args.sim_accel)
    maybe("sim_mask_type", args.sim_mask_type)
    maybe("sim_seed", int(args.sim_seed))
    maybe("sim_center", bool(args.sim_center))
    maybe("sim_fixed_mask", not bool(args.sim_vary_per_slice))

    if "case_dirs" in accepted and "acc_root" not in accepted:
        case_dirs = sorted([p for p in Path(acc_root).glob("P*") if p.is_dir()])
        if not case_dirs:
            raise FileNotFoundError(f"No P* case directories found under {acc_root}")
        kwargs["case_dirs"] = case_dirs

    try:
        return DimoKspaceDataset(**kwargs)
    except TypeError as e:
        # Last-resort compatibility for old constructors.
        if args.root is not None:
            return DimoKspaceDataset(root=args.root, acc_factor=acc_factor, coil_type="single", split="test")  # type: ignore[call-arg]
        raise e


def _build_model(state: Dict[str, torch.Tensor], args: argparse.Namespace, timesteps: int, cond_ch: int, device: torch.device) -> Any:
    try:
        model = DiMoDDPM(T=timesteps, data_ch=args.data_ch, cond_ch=cond_ch).to(device)
    except TypeError:
        model = DiMoDDPM(timesteps=timesteps, data_ch=args.data_ch, cond_ch=cond_ch).to(device)
    load_res = model.load_state_dict(state, strict=False)
    if getattr(load_res, "missing_keys", None):
        print(f"[WARN] missing_keys: {load_res.missing_keys}")
    if getattr(load_res, "unexpected_keys", None):
        print(f"[WARN] unexpected_keys: {load_res.unexpected_keys}")
    return model


def _build_cond(cond_mode: str, *, zf_2ch_norm: torch.Tensor, mask_bhw: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
    cond_mode = str(cond_mode).lower()
    if cond_mode in {"none", "", "null"}:
        return None
    zf_2ch_norm = _ensure_bchw_2ch(zf_2ch_norm).to(device)
    if cond_mode == "zf":
        return zf_2ch_norm
    if cond_mode == "zf_mask":
        m = _mask_to_bhw(mask_bhw, batch_size=zf_2ch_norm.shape[0], device=device).unsqueeze(1)
        return torch.cat([zf_2ch_norm, m], dim=1)
    raise ValueError("cond_mode must be one of: auto | none | zf | zf_mask")


def _normalize_residual_log(residual_log: Any) -> Optional[List[Dict[str, float]]]:
    if residual_log is None:
        return None
    if isinstance(residual_log, list):
        return [{k: (float(v) if k in {"before", "after"} else int(v) if k in {"iter", "t", "t_prev"} else v) for k, v in row.items()} for row in residual_log]
    if hasattr(residual_log, "to_dict"):
        residual_log = residual_log.to_dict()
    if isinstance(residual_log, dict):
        t_list = residual_log.get("t", residual_log.get("timesteps", []))
        before = residual_log.get("before", residual_log.get("before_dc", []))
        after = residual_log.get("after", residual_log.get("after_dc", []))
        rows = []
        for i, (t, rb, ra) in enumerate(zip(t_list, before, after)):
            rows.append({"iter": int(i), "t": int(t), "before": float(rb), "after": float(ra)})
        return rows
    return None


def _run_sampler(
    *,
    model: Any,
    sense_op: Any,
    y_2ch_norm: torch.Tensor,
    mask_bhw: torch.Tensor,
    x_init_norm: Optional[torch.Tensor],
    cond: Optional[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[List[Dict[str, float]]]]:
    dc_mode = str(args.dc_mode).lower()
    if dc_mode == "hard":
        dc_mode = "replace"
    if dc_mode == "soft":
        dc_mode = "grad"

    try:
        out = ddim_with_dc_from_model(
            model=model,
            sense_op=sense_op,
            y_k_2ch=y_2ch_norm,
            mask=mask_bhw,
            x_init_2ch=x_init_norm,
            cond=cond,
            strength=float(args.stage2_strength),
            num_steps=int(args.num_steps),
            dc_mode=dc_mode,
            dc_lam=float(args.dc_lam),
            dc_cg_iter=int(args.dc_cg_iter),
            dc_cg_tol=float(args.dc_cg_tol),
            log_residuals=bool(args.log_residuals),
            device=device,
        )
    except TypeError as e:
        if cond is not None:
            raise TypeError(
                "Your src.recon.dimo_sample.ddim_with_dc_from_model does not appear to accept conditional sampling. "
                "Install the newer dimo_sample.py before using a conditional checkpoint."
            ) from e
        out = ddim_with_dc_from_model(
            model=model,
            sense_op=sense_op,
            y_k_2ch=y_2ch_norm,
            mask=mask_bhw,
            x_init_2ch=x_init_norm,
            init_strength=float(args.stage2_strength),
            num_steps=int(args.num_steps),
            dc_mode=dc_mode,
            dc_lambda=float(args.dc_lam),
            return_residuals=bool(args.log_residuals),
        )

    if args.log_residuals:
        x_rec_norm, residual_log = out  # type: ignore[misc]
        return x_rec_norm, _normalize_residual_log(residual_log)
    return out, None  # type: ignore[return-value]


def _save_png_panel(payload: Dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Could not import matplotlib for PNG preview: {e}")
        return

    def mag(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or not isinstance(x, torch.Tensor):
            return None
        x = _ensure_bchw_2ch(x).detach().cpu()
        return torch.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2)

    panels = [
        ("ZF", mag(payload.get("zf_2ch"))),
        ("Init", mag(payload.get("x_init_2ch"))),
        ("Stage-2", mag(payload.get("x_rec_2ch"))),
        ("Target", mag(payload.get("x_target_2ch"))),
    ]
    panels = [(t, im) for t, im in panels if im is not None]
    if not panels:
        return
    vmax = max(float(im.quantile(0.995)) for _, im in panels if im is not None)
    vmax = max(vmax, 1e-8)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, (title, im) in zip(axes[0], panels):
        ax.imshow(im.numpy(), cmap="gray", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Canonical Stage-2 sampling with conditional DiMo + data consistency")

    # Data / checkpoint: current and legacy styles.
    p.add_argument("--acc_root", default=None, help="Path to AccFactorXX/AccFactor04-like directory")
    p.add_argument("--root", default=None, help="Legacy dataset root; combined with --acc")
    p.add_argument("--acc", default="AccFactor04", help="Legacy acceleration folder name, e.g. AccFactor04")
    p.add_argument("--acc_factor", default=None, help="Acceleration id, e.g. 04/08/10. Inferred if omitted.")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--ckpt", required=True)

    # Dataset options.
    p.add_argument("--target_mode", default=None, choices=[None, "complex", "rss"], help="Dataset target mode. Defaults to checkpoint config or complex.")
    p.add_argument("--simulate_mask", action="store_true")
    p.add_argument("--sim_accel", type=int, default=None)
    p.add_argument("--sim_mask_type", default="random_1D")
    p.add_argument("--sim_seed", type=int, default=0)
    p.add_argument("--no_sim_center", action="store_false", dest="sim_center")
    p.set_defaults(sim_center=True)
    p.add_argument("--sim_vary_per_slice", action="store_true")

    # Model / conditioning.
    p.add_argument("--cond_mode", default="auto", choices=["auto", "none", "zf", "zf_mask"])
    p.add_argument("--cond_ch", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--data_ch", type=int, default=2)

    # Init handling.
    p.add_argument("--init_mode", default="zf", choices=["noise", "zf", "pt", "stage1"])
    p.add_argument("--init_pt", default=None, help=".pt file to use as Stage-2 image-domain initialization")
    p.add_argument("--init_key", default="x_out_2ch", help="Key inside --init_pt, e.g. x_out_2ch")
    p.add_argument("--init_is_normalized", action="store_true", help="Use this only if --init_pt is already in normalized units")
    p.add_argument("--stage1_strength", type=float, default=0.30)
    p.add_argument("--stage1_steps", type=int, default=None)
    p.add_argument("--stage2_strength", type=float, default=0.10)
    p.add_argument("--num_steps", type=int, default=50)

    # Data consistency.
    p.add_argument("--dc_mode", default="replace", choices=["replace", "hard", "grad", "soft", "cg", "none", "off"])
    p.add_argument("--dc_lam", type=float, default=0.1)
    p.add_argument("--dc_lambda", type=float, default=None, help="Legacy alias for --dc_lam")
    p.add_argument("--dc_cg_iter", type=int, default=10)
    p.add_argument("--dc_cg_tol", type=float, default=1e-6)
    p.add_argument("--log_residuals", action="store_true")

    # Scaling. This is critical for matching train_step normalization.
    p.add_argument("--scale_mode", default="auto", choices=["auto", "zf", "target", "none"], help="Internal normalization scale. auto=zf.")
    p.add_argument("--scale_min", type=float, default=1e-8)

    # Outputs.
    p.add_argument("--dump_stage1_bundle", default=None)
    p.add_argument("--bundle_scale", default="original", choices=["original", "normalized"], help="Whether dumped Stage-1 bundle stores original or normalized tensors")
    p.add_argument("--out_dir", default="outputs/stage2_run")
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_argparser().parse_args()
    if args.dc_lambda is not None:
        args.dc_lam = float(args.dc_lambda)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    state, cfg, legacy_args, raw_ckpt = _load_state_and_config(args.ckpt)
    args.timesteps = _infer_timesteps(state, cfg, legacy_args, fallback=args.timesteps)
    cond_mode = _infer_cond_mode(args.cond_mode, cfg, legacy_args)
    cond_ch = _infer_cond_ch(cond_mode, args.cond_ch, cfg, legacy_args, state)
    target_mode = args.target_mode or cfg.get("target_mode") or legacy_args.get("target_mode") or "complex"
    acc_factor = _infer_acc_factor(args.acc_root, args.acc, cfg, legacy_args, args.acc_factor)

    model = _build_model(state, args, args.timesteps, cond_ch, device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    print(f"[INFO] timesteps={args.timesteps} cond_mode={cond_mode} cond_ch={cond_ch} target_mode={target_mode} acc_factor={acc_factor}")

    ds = _make_dataset(args, cfg, legacy_args, acc_factor, target_mode, cond_mode)
    sample: Dict[str, Any] = ds[args.index]

    y_raw = _get_first(sample, ["kspace_und_2ch", "y_2ch", "kspace_und", "kspace", "y"])
    mask_raw = _get_first(sample, ["mask", "P", "sampling_mask"])
    target_raw = _get_first(sample, ["x_target_2ch", "img_target_2ch", "x0_2ch", "img_target", "x0"], required=False)

    y_2ch = _ensure_bchw_2ch(y_raw, name="y_2ch").to(device)
    B, _, H, W = y_2ch.shape
    mask_bhw = _mask_to_bhw(mask_raw, batch_size=B, device=device)
    x_target = _ensure_bchw_2ch(target_raw, name="x_target_2ch").to(device) if target_raw is not None else None

    sense_op = SenseOp(mask=mask_bhw)

    # Prefer computing ZF from y/mask; fallback to dataset zf if SenseOp API is incompatible.
    try:
        zf_c = _adjoint(sense_op, ch2_to_complex(y_2ch), mask_bhw)
        zf_2ch = _ensure_bchw_2ch(complex_to_2ch(zf_c), name="zf_2ch").to(device)
    except Exception as e:
        zf_raw = _get_first(sample, ["zf_2ch", "x_in_2ch"], required=False)
        if zf_raw is None:
            raise RuntimeError(f"Could not compute ZF via SenseOp and no zf_2ch exists in sample. Original error: {e}")
        print(f"[WARN] SenseOp adjoint failed ({e}); using dataset zf_2ch.")
        zf_2ch = _ensure_bchw_2ch(zf_raw, name="zf_2ch").to(device)

    # Scale-aware inference. Training normalizes x0 by one scalar per image;
    # sampling should use the same linear scale for zf, y, init, target.
    scale_mode = "zf" if args.scale_mode == "auto" else args.scale_mode
    if scale_mode == "none":
        scale_2ch = torch.ones((B, 1, 1, 1), device=device, dtype=zf_2ch.dtype)
    elif scale_mode == "target":
        if x_target is None:
            print("[WARN] --scale_mode target requested but no target exists; falling back to zf scale.")
            scale_2ch = _scale_from_2ch(zf_2ch, scale_min=args.scale_min).to(device)
        else:
            scale_2ch = _scale_from_2ch(x_target, scale_min=args.scale_min).to(device)
    else:
        scale_2ch = _scale_from_2ch(zf_2ch, scale_min=args.scale_min).to(device)

    y_norm = y_2ch / scale_2ch
    zf_norm = zf_2ch / scale_2ch
    target_norm = x_target / scale_2ch if x_target is not None else None
    cond = _build_cond(cond_mode, zf_2ch_norm=zf_norm, mask_bhw=mask_bhw, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_stage1_bundle is not None:
        bundle_path = Path(args.dump_stage1_bundle)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        if args.bundle_scale == "normalized":
            b_zf, b_y, b_tgt = zf_norm, y_norm, target_norm
        else:
            b_zf, b_y, b_tgt = zf_2ch, y_2ch, x_target
        torch.save(
            {
                "x_in_2ch": b_zf.detach().cpu(),
                "zf_2ch": b_zf.detach().cpu(),
                "x_target_2ch": (b_tgt.detach().cpu() if b_tgt is not None else None),
                "y_2ch": b_y.detach().cpu(),
                "mask": mask_bhw.detach().cpu(),
                "scale_2ch": scale_2ch.detach().cpu(),
                "meta": {
                    "acc_root": args.acc_root or str(Path(args.root) / args.acc),
                    "index": args.index,
                    "ckpt": args.ckpt,
                    "cond_mode": cond_mode,
                    "cond_ch": cond_ch,
                    "timesteps": args.timesteps,
                    "target_mode": target_mode,
                    "scale_mode": scale_mode,
                    "bundle_scale": args.bundle_scale,
                },
            },
            bundle_path,
        )
        print(f"[INFO] Saved Stage-1 bundle to: {bundle_path}")

    # Decide Stage-2 init in normalized units.
    x_init_norm: Optional[torch.Tensor] = None
    x_init_original: Optional[torch.Tensor] = None
    init_source = args.init_mode
    init_key_used: Optional[str] = None

    effective_init_mode = args.init_mode
    if args.init_pt is not None and args.init_mode == "noise":
        print("[WARN] --init_pt was provided with --init_mode noise; treating this as --init_mode pt.")
        effective_init_mode = "pt"

    if effective_init_mode == "noise":
        x_init_norm = None
        x_init_original = None
        init_source = "noise"
    elif effective_init_mode == "zf":
        x_init_norm = zf_norm
        x_init_original = zf_2ch
        init_source = "zf"
    elif effective_init_mode == "pt":
        if args.init_pt is None:
            raise ValueError("--init_pt is required with --init_mode pt")
        init_tensor_original, init_key_used, init_obj = _load_init_tensor(args.init_pt, args.init_key, device)
        if init_tensor_original.shape[-2:] != (H, W):
            raise ValueError(f"init tensor spatial shape {tuple(init_tensor_original.shape[-2:])} does not match sample {(H, W)}")
        x_init_original = init_tensor_original
        x_init_norm = init_tensor_original if args.init_is_normalized else init_tensor_original / scale_2ch
        init_source = f"pt:{args.init_pt}::{init_key_used}"
    elif effective_init_mode == "stage1":
        stage1_steps = int(args.stage1_steps) if args.stage1_steps is not None else int(args.num_steps)
        x_init_norm = _ddim_img2img_no_dc(
            model,
            zf_norm,
            cond=cond,
            strength=float(args.stage1_strength),
            num_steps=stage1_steps,
            device=device,
        )
        x_init_original = x_init_norm * scale_2ch
        init_source = "stage1(zf)"
        torch.save(
            {
                "x_in_2ch": zf_2ch.detach().cpu(),
                "zf_2ch": zf_2ch.detach().cpu(),
                "x_out_2ch": x_init_original.detach().cpu(),
                "x_target_2ch": (x_target.detach().cpu() if x_target is not None else None),
                "mask": mask_bhw.detach().cpu(),
                "scale_2ch": scale_2ch.detach().cpu(),
                "meta": {
                    "ckpt": args.ckpt,
                    "cond_mode": cond_mode,
                    "cond_ch": cond_ch,
                    "timesteps": args.timesteps,
                    "stage1_strength": float(args.stage1_strength),
                    "stage1_steps": stage1_steps,
                    "scale_mode": scale_mode,
                },
            },
            out_dir / "stage1_denoise_output.pt",
        )
    else:
        raise ValueError(f"Unknown init_mode: {effective_init_mode}")

    x_rec_norm, residual_rows = _run_sampler(
        model=model,
        sense_op=sense_op,
        y_2ch_norm=y_norm,
        mask_bhw=mask_bhw,
        x_init_norm=x_init_norm,
        cond=cond,
        args=args,
        device=device,
    )
    x_rec = _ensure_bchw_2ch(x_rec_norm, name="x_rec_norm").to(device) * scale_2ch

    payload: Dict[str, Any] = {
        "x_rec_2ch": x_rec.detach().cpu(),
        "x_rec_norm_2ch": x_rec_norm.detach().cpu(),
        "x_init_2ch": (x_init_original.detach().cpu() if x_init_original is not None else None),
        "x_init_norm_2ch": (x_init_norm.detach().cpu() if x_init_norm is not None else None),
        "x_in_2ch": zf_2ch.detach().cpu(),
        "zf_2ch": zf_2ch.detach().cpu(),
        "zf_norm_2ch": zf_norm.detach().cpu(),
        "x_target_2ch": (x_target.detach().cpu() if x_target is not None else None),
        "x_target_norm_2ch": (target_norm.detach().cpu() if target_norm is not None else None),
        "y_2ch": y_2ch.detach().cpu(),
        "y_norm_2ch": y_norm.detach().cpu(),
        "mask": mask_bhw.detach().cpu(),
        "scale_2ch": scale_2ch.detach().cpu(),
        "residual_log": residual_rows,
        "meta": {
            "acc_root": args.acc_root or (str(Path(args.root) / args.acc) if args.root is not None else None),
            "root": args.root,
            "acc": args.acc,
            "acc_factor": acc_factor,
            "index": args.index,
            "ckpt": args.ckpt,
            "cond_mode": cond_mode,
            "cond_ch": cond_ch,
            "init_mode": effective_init_mode,
            "init_source": init_source,
            "init_key": init_key_used,
            "stage1_strength": float(args.stage1_strength),
            "stage1_steps": args.stage1_steps,
            "stage2_strength": float(args.stage2_strength),
            "num_steps": int(args.num_steps),
            "dc_mode": str(args.dc_mode).lower(),
            "dc_lam": float(args.dc_lam),
            "dc_cg_iter": int(args.dc_cg_iter),
            "dc_cg_tol": float(args.dc_cg_tol),
            "timesteps": int(args.timesteps),
            "target_mode": target_mode,
            "scale_mode": scale_mode,
            "scale_mean": float(scale_2ch.mean().detach().cpu()),
            "device": str(device),
        },
    }

    out_pt = out_dir / "stage2_recon_output.pt"
    torch.save(payload, out_pt)

    if residual_rows is not None:
        with open(out_dir / "dc_residuals.json", "w", encoding="utf-8") as f:
            json.dump(residual_rows, f, indent=2)
        print("\n[DC residuals] (measured k-space lines only; internal normalized units)\n")
        print(f"{'iter':>4} {'t':>6} {'before':>14} {'after':>14}")
        for r in residual_rows:
            print(f"{int(r.get('iter', 0)):4d} {int(r.get('t', r.get('t_prev', 0))):6d} {float(r['before']):14.6e} {float(r['after']):14.6e}")

    if args.save_png:
        _save_png_panel(payload, out_dir / "stage2_preview.png")

    print(f"Saved outputs to: {out_dir}")
    print(f"[INFO] scale_mode={scale_mode} scale_mean={float(scale_2ch.mean().detach().cpu()):.6g}")


if __name__ == "__main__":
    main()
