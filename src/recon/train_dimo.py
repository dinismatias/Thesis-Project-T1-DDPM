# src/recon/train_dimo.py
"""
Ultimate DiMo/DDPM trainer for CMRxRecon T1 mapping.

Goals of this version
---------------------
1) Keep the working conditional training path: none / zf / zf_mask.
2) Support both dataset constructor styles used in this project:
   - DimoKspaceDataset(acc_root=...)
   - DimoKspaceDataset(case_dirs=...)
3) Support single-AF and mixed-AF training, e.g. AF=04/08/10 in one run.
4) Use one canonical checkpoint schema while remaining able to resume old checkpoints.
5) Be GPU-ready: AMP, pinned memory, persistent workers, optional torch.compile.

Typical single-AF run
---------------------
python -m src.recon.train_dimo ^
  --acc_root "C:\\Users\\Admin\\Tese\\T1_DDPM_Project\\ChallengeData\\SingleCoil\\Mapping\\TrainingSet\\AccFactorXX\\AccFactor04" ^
  --acc_factor 04 ^
  --target_mode complex ^
  --cond_mode zf_mask ^
  --epochs 20 ^
  --batch_size 4 ^
  --timesteps 100 ^
  --device cuda ^
  --amp ^
  --ckpt_dir "checkpoints/dimo_cond_r4_gpu"

Typical mixed-AF run
--------------------
python -m src.recon.train_dimo ^
  --acc_roots "...\\AccFactor04" "...\\AccFactor08" "...\\AccFactor10" ^
  --acc_factors 04 08 10 ^
  --target_mode complex ^
  --cond_mode zf_mask ^
  --epochs 40 ^
  --batch_size 4 ^
  --timesteps 100 ^
  --device cuda ^
  --amp ^
  --ckpt_dir "checkpoints/dimo_cond_mixed_4_8_10"
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from src.recon.dimo_dataset import DimoKspaceDataset
    from src.recon.dimo_model import DiMoDDPM
    from src.recon.dimo_train_step import train_step
except Exception:  # pragma: no cover
    from dimo_dataset import DimoKspaceDataset  # type: ignore
    from dimo_model import DiMoDDPM  # type: ignore
    from dimo_train_step import train_step  # type: ignore


# -----------------------------------------------------------------------------
# Small compatibility helpers
# -----------------------------------------------------------------------------


def _namespace_to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        return dict(vars(x))
    return {}


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(state)
    if any(k.startswith("module.") for k in out.keys()):
        out = {k.replace("module.", "", 1): v for k, v in out.items()}
    return out


def _extract_state_and_meta(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Accept old and new checkpoint schemas.

    New canonical schema saved by this script:
        {"model_state": ..., "optimizer_state": ..., "config": ...}

    Legacy schemas seen in this project:
        {"state_dict": ...}
        {"model": ..., "opt": ..., "args": ...}
        raw state_dict
    """
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
        raise ValueError("Checkpoint model state is not dict-like.")

    meta = {}
    meta.update(_namespace_to_dict(ckpt.get("config")))
    # Do not overwrite explicit config with old args unless missing.
    for k, v in _namespace_to_dict(ckpt.get("args")).items():
        meta.setdefault(k, v)
    for k in ("epoch", "global_step"):
        if k in ckpt:
            meta[k] = ckpt[k]

    return _strip_module_prefix(state), meta


def _extract_optimizer_state(ckpt: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ckpt, dict):
        return None
    opt = ckpt.get("optimizer_state", ckpt.get("opt", None))
    return opt if isinstance(opt, dict) else None


def _load_state_dict_checked(model: torch.nn.Module, state: Dict[str, torch.Tensor], *, strict: bool) -> None:
    res = model.load_state_dict(state, strict=strict)
    missing = getattr(res, "missing_keys", [])
    unexpected = getattr(res, "unexpected_keys", [])
    if missing:
        print(f"[WARN] Missing model keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected model keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
    if strict and (missing or unexpected):
        raise RuntimeError("Strict checkpoint load failed because model keys did not match.")


def _call_model_constructor(*, data_ch: int, cond_ch: int, timesteps: int) -> torch.nn.Module:
    """Handle both DiMoDDPM(T=...) and DiMoDDPM(timesteps=...)."""
    try:
        return DiMoDDPM(data_ch=data_ch, cond_ch=cond_ch, T=timesteps)
    except TypeError:
        return DiMoDDPM(data_ch=data_ch, cond_ch=cond_ch, timesteps=timesteps)


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _infer_cond_ch(cond_mode: str, cond_ch_arg: Optional[int]) -> int:
    if cond_ch_arg is not None:
        return int(cond_ch_arg)
    cond_mode = str(cond_mode).lower()
    if cond_mode == "none":
        return 0
    if cond_mode == "zf":
        return 2
    if cond_mode == "zf_mask":
        return 3
    raise ValueError(f"Unknown cond_mode: {cond_mode}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_case_dirs(acc_root: Path, limit_cases: int = 0) -> List[Path]:
    candidates = sorted([p for p in acc_root.glob("P*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(
            f"No case directories found under {acc_root}. Expected subject folders like P001, P002, ..."
        )
    return candidates[:limit_cases] if limit_cases and limit_cases > 0 else candidates


def _split_cases(case_dirs: List[Path], val_frac: float, seed: int) -> Tuple[List[Path], List[Path]]:
    if val_frac <= 0.0:
        return case_dirs, []
    rng = np.random.RandomState(seed)
    idx = np.arange(len(case_dirs))
    rng.shuffle(idx)
    n_val = int(round(len(case_dirs) * val_frac))
    n_val = min(max(n_val, 1), max(len(case_dirs) - 1, 1)) if len(case_dirs) > 1 else 0
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]
    return [case_dirs[i] for i in trn_idx], [case_dirs[i] for i in val_idx]


def _make_dataset_for_root(
    *,
    acc_root: Path,
    acc_factor: str,
    case_dirs: Optional[List[Path]],
    args: argparse.Namespace,
) -> Dataset:
    """Create a DimoKspaceDataset while respecting whatever constructor exists locally."""
    base_kwargs: Dict[str, Any] = {
        "acc_root": str(acc_root),
        "case_dirs": case_dirs,
        "acc_factor": acc_factor,
        "multi_coil": bool(args.multi_coil),
        "use_full_as_target": bool(args.use_full_as_target),
        "target_mode": args.target_mode,
        "cond_mode": args.cond_mode,
        "simulate_mask": bool(args.simulate_mask),
        "sim_accel": args.sim_accel,
        "sim_mask_type": args.sim_mask_type,
        "sim_seed": int(args.sim_seed),
        "sim_center": bool(args.sim_center),
        "sim_vary_per_slice": bool(args.sim_vary_per_slice),
        # Some old versions used the inverse name.
        "sim_fixed_mask": (not bool(args.sim_vary_per_slice)),
    }
    # Avoid sending case_dirs=None if constructor does not expect it or if acc_root-only style is used.
    if case_dirs is None:
        base_kwargs.pop("case_dirs", None)
    kwargs = _filter_kwargs_for_callable(DimoKspaceDataset.__init__, base_kwargs)
    return DimoKspaceDataset(**kwargs)


def _make_dataset_group(
    *,
    roots: Sequence[Path],
    acc_factors: Sequence[str],
    split: str,
    args: argparse.Namespace,
) -> Dataset:
    datasets: List[Dataset] = []
    for root, af in zip(roots, acc_factors):
        all_cases = _build_case_dirs(root, args.limit_cases)
        train_cases, val_cases = _split_cases(all_cases, args.val_frac, args.seed)
        selected = train_cases if split == "train" else val_cases
        print(f"[INFO] {split:5s} | AF={af} | root={root} | cases={len(selected)}")
        if not selected:
            continue
        datasets.append(_make_dataset_for_root(acc_root=root, acc_factor=af, case_dirs=selected, args=args))

    if not datasets:
        raise RuntimeError(f"No datasets were created for split={split!r}.")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _train_step_compat(
    *,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_mode: str,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Call project train_step across the different signatures used in the repo.

    Preferred modern signature:
        train_step(model=model, batch=batch, optimizer=optimizer, device=device, cond_mode=cond_mode)

    Older signature:
        train_step(model, batch, optimizer, device, cond_mode=cond_mode)

    AMP path is implemented here only when the imported train_step exposes a pure
    forward loss with optimizer=None. If your local train_step owns backward/step,
    this still falls back cleanly.
    """
    if use_amp and optimizer is not None:
        # Try to obtain a loss without stepping, then do the AMP backward here.
        try:
            with torch.cuda.amp.autocast(enabled=True):
                out = train_step(model=model, batch=batch, optimizer=None, device=device, cond_mode=cond_mode)
            loss, stats = _parse_train_step_output(out)
            optimizer.zero_grad(set_to_none=True)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss.detach(), stats
        except TypeError:
            # Fall back to native train_step below.
            pass

    try:
        out = train_step(model=model, batch=batch, optimizer=optimizer, device=device, cond_mode=cond_mode)
    except TypeError:
        try:
            out = train_step(model, batch, optimizer, device, cond_mode=cond_mode)
        except TypeError:
            # Very old train_step did not know cond_mode.
            out = train_step(model, batch, optimizer, device)
    return _parse_train_step_output(out)


def _parse_train_step_output(out: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    if isinstance(out, tuple):
        loss = out[0]
        stats = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
    else:
        loss = out
        stats = {}
    if not torch.is_tensor(loss):
        loss = torch.tensor(float(loss))
    stats = dict(stats)
    stats.setdefault("loss", float(loss.detach().cpu()))
    return loss, stats


@torch.no_grad()
def _eval_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, cond_mode: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss, _ = _train_step_compat(
            model=model,
            batch=batch,
            optimizer=None,
            device=device,
            cond_mode=cond_mode,
        )
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def _save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    ckpt_path: Path,
    config: Dict[str, Any],
    best_val_loss: Optional[float] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": "dimo-ultimate-v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": dict(config),
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}")


def _load_resume(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: Path,
    resume: str,
    strict_load: bool,
) -> Tuple[int, int, Optional[float]]:
    if resume.lower() in {"", "none", "false", "no"}:
        print("[INFO] Resume disabled; starting from scratch.")
        return 0, 0, None

    if resume.lower() in {"auto", "true", "yes"}:
        ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
        if not ckpts:
            print("[INFO] No checkpoint found, starting from scratch.")
            return 0, 0, None
        path = ckpts[-1]
    else:
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {path}")

    print(f"[INFO] Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    state, meta = _extract_state_and_meta(ckpt)
    _load_state_dict_checked(model, state, strict=strict_load)
    opt_state = _extract_optimizer_state(ckpt)
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"[WARN] Could not load optimizer state: {e}")
    start_epoch = int(meta.get("epoch", -1)) + 1
    global_step = int(meta.get("global_step", 0))
    best_val_loss = None
    if isinstance(ckpt, dict) and ckpt.get("best_val_loss") is not None:
        best_val_loss = float(ckpt["best_val_loss"])
    return max(start_epoch, 0), global_step, best_val_loss


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train conditional/unconditional DiMo DDPM for CMRxRecon T1 mapping.")

    # Data. Keep old single-root interface and add mixed-root interface.
    p.add_argument("--acc_root", type=str, default=None, help="Single AF root, e.g. .../AccFactor04")
    p.add_argument("--acc_factor", type=str, default="04", help="AF string for --acc_root: 04/08/10")
    p.add_argument("--acc_roots", nargs="*", default=None, help="Multiple AF roots for mixed training")
    p.add_argument("--acc_factors", nargs="*", default=None, help="AF strings matching --acc_roots, e.g. 04 08 10")
    p.add_argument("--multi_coil", action="store_true")
    p.add_argument("--use_full_as_target", action="store_true")
    p.add_argument("--target_mode", default="complex", choices=["complex", "rss"])
    p.add_argument("--cond_mode", default="none", choices=["none", "zf", "zf_mask"])

    # Optional simulated mask path.
    p.add_argument("--simulate_mask", action="store_true")
    p.add_argument("--sim_accel", default=None)
    p.add_argument("--sim_mask_type", default="out_r_1D")
    p.add_argument("--sim_seed", type=int, default=0)
    p.add_argument("--no_sim_center", action="store_false", dest="sim_center")
    p.set_defaults(sim_center=True)
    p.add_argument("--sim_vary_per_slice", action="store_true")

    # Training.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--limit_cases", type=int, default=0)
    p.add_argument("--drop_last", action="store_true", default=True)
    p.add_argument("--no_drop_last", action="store_false", dest="drop_last")

    # Model.
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--data_ch", type=int, default=2)
    p.add_argument("--cond_ch", type=int, default=None)

    # Runtime/checkpointing.
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision when possible")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    p.add_argument("--ckpt_dir", default="checkpoints/dimo")
    p.add_argument("--resume", default="auto", help="auto, none, or explicit checkpoint path")
    p.add_argument("--strict_load", action="store_true", help="Require exact model key match when resuming")
    p.add_argument("--save_best", action="store_true", help="When val set exists, also save best.pt")

    return p


def _resolve_roots_and_afs(args: argparse.Namespace) -> Tuple[List[Path], List[str]]:
    if args.acc_roots:
        roots = [Path(x) for x in args.acc_roots]
        if args.acc_factors:
            afs = [str(x).zfill(2) for x in args.acc_factors]
            if len(afs) != len(roots):
                raise ValueError("--acc_factors must have the same length as --acc_roots")
        else:
            afs = []
            for root in roots:
                name = root.name
                digits = "".join(ch for ch in name if ch.isdigit())
                afs.append(digits[-2:].zfill(2) if digits else str(args.acc_factor).zfill(2))
        return roots, afs

    if args.acc_root is None:
        raise ValueError("Provide either --acc_root or --acc_roots")
    return [Path(args.acc_root)], [str(args.acc_factor).zfill(2)]


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = build_argparser().parse_args()

    _set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    roots, afs = _resolve_roots_and_afs(args)
    for r in roots:
        if not r.exists():
            raise FileNotFoundError(f"Data root does not exist: {r}")

    train_ds = _make_dataset_group(roots=roots, acc_factors=afs, split="train", args=args)
    val_ds: Optional[Dataset] = None
    if args.val_frac > 0:
        try:
            val_ds = _make_dataset_group(roots=roots, acc_factors=afs, split="val", args=args)
        except RuntimeError:
            val_ds = None

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=bool(args.drop_last), **loader_kwargs)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    # Quick first-batch shape report: catches most interface problems immediately.
    first = next(iter(train_loader))
    print("[INFO] First batch keys/shapes:")
    for k, v in first.items():
        if torch.is_tensor(v):
            print(f"       {k:18s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"       {k:18s} {type(v).__name__}")

    cond_ch = _infer_cond_ch(args.cond_mode, args.cond_ch)
    model = _call_model_constructor(data_ch=int(args.data_ch), cond_ch=cond_ch, timesteps=int(args.timesteps)).to(device)
    if args.compile and hasattr(torch, "compile"):
        print("[INFO] Using torch.compile(model)")
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    config: Dict[str, Any] = {
        "acc_roots": [str(x) for x in roots],
        "acc_factors": afs,
        "target_mode": args.target_mode,
        "cond_mode": args.cond_mode,
        "timesteps": int(args.timesteps),
        "data_ch": int(args.data_ch),
        "cond_ch": int(cond_ch),
        "simulate_mask": bool(args.simulate_mask),
        "sim_accel": args.sim_accel,
        "sim_mask_type": args.sim_mask_type,
        "sim_seed": int(args.sim_seed),
        "sim_center": bool(args.sim_center),
        "sim_vary_per_slice": bool(args.sim_vary_per_slice),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "amp": bool(args.amp and device.type == "cuda"),
    }

    ckpt_dir = Path(args.ckpt_dir)
    start_epoch, global_step, best_val_loss = _load_resume(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        resume=args.resume,
        strict_load=bool(args.strict_load),
    )

    # Save config as JSON beside checkpoints for human inspection.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[INFO] Training samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"[INFO] Validation samples: {len(val_ds)}")
    print(f"[INFO] cond_mode={args.cond_mode} cond_ch={cond_ch} T={args.timesteps}")

    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        running_loss = 0.0
        running_scale = 0.0
        n_batches = 0

        for batch in train_loader:
            loss, stats = _train_step_compat(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                cond_mode=args.cond_mode,
                use_amp=bool(args.amp and device.type == "cuda"),
                scaler=scaler,
            )
            loss_val = float(stats.get("loss", float(loss.detach().cpu())))
            scale_val = float(stats.get("scale_mean", 0.0))
            running_loss += loss_val
            running_scale += scale_val
            n_batches += 1
            global_step += 1

            if global_step % int(args.log_interval) == 0:
                avg_loss = running_loss / max(n_batches, 1)
                avg_scale = running_scale / max(n_batches, 1)
                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] Step {global_step} | "
                    f"Avg loss: {avg_loss:.6f} | scale_mean: {avg_scale:.4g}"
                )

        epoch_loss = running_loss / max(n_batches, 1)
        print(f"[INFO] Epoch {epoch + 1} finished. Mean loss = {epoch_loss:.6f}")

        val_loss = None
        if val_loader is not None:
            val_loss = _eval_loss(model=model, loader=val_loader, device=device, cond_mode=args.cond_mode)
            print(f"[INFO] Val loss = {val_loss:.6f}")

        if (epoch + 1) % int(args.save_interval) == 0:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                ckpt_path=ckpt_dir / f"epoch_{epoch + 1:04d}.pt",
                config=config,
                best_val_loss=best_val_loss,
            )
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                ckpt_path=ckpt_dir / "last.pt",
                config=config,
                best_val_loss=best_val_loss,
            )

        if args.save_best and val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = float(val_loss)
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                ckpt_path=ckpt_dir / "best.pt",
                config=config,
                best_val_loss=best_val_loss,
            )


if __name__ == "__main__":
    main()
