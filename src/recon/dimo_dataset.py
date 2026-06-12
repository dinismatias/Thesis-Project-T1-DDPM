# src/recon/dimo_dataset.py
"""Robust CMRxRecon SingleCoil Mapping dataset for DiMo/DDPM reconstruction.

This version is designed to replace the many intermediate dimo_dataset_v*.py files.
It keeps the working behavior of the latest loader while re-adding the useful
compatibility from older versions:

- acc_root=... OR case_dirs=... construction.
- legacy root=... alias when possible.
- MATLAB classic .mat and MATLAB v7.3/HDF5 .mat loading.
- undersampled k-space loaded from T1map_sub.mat first, then T1map.mat fallback.
- full target loaded from the sibling FullSample/P###/T1map.mat folder.
- robust frame extraction for Mapping arrays such as [H,W,Z,TI] or [TI,Z,W,H]:
  the two largest axes are treated as spatial axes and all other axes are
  flattened into frame_idx.
- 1D, 2D, and per-frame masks.
- optional retrospective mask simulation.
- target_mode='complex' or 'rss'.
- cond_mode='none', 'zf', or 'zf_mask'.
- output aliases used by the current training/sampling scripts.

Expected main output keys per sample:
    kspace_und_2ch / y_2ch: [2,H,W]
    mask:                  [H,W]
    img_target_2ch / x0_2ch:[2,H,W]
    zf_2ch:                [2,H,W]
    cond:                  [2,H,W] or [3,H,W], only when cond_mode != 'none'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import scipy.io as sio
except Exception:  # pragma: no cover
    sio = None  # type: ignore

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

try:
    from src.recon.encoding import ifft2c
    from src.utils.complex_ops import complex_to_2ch
except Exception:  # pragma: no cover
    from encoding import ifft2c  # type: ignore
    from complex_ops import complex_to_2ch  # type: ignore


# -----------------------------------------------------------------------------
# MATLAB / HDF5 loading helpers
# -----------------------------------------------------------------------------


def _as_numpy(x: Any) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def _unwrap_mat_value(v: Any) -> Any:
    """Unwrap MATLAB object arrays that contain a single real array."""
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.flat[0]
    return v


def _h5_to_python(obj: Any) -> Any:
    """Recursively convert h5py objects to numpy arrays/dicts."""
    if h5py is None:
        raise ImportError("h5py is required to read MATLAB v7.3 files")

    if isinstance(obj, h5py.Dataset):
        return np.array(obj[()])

    if isinstance(obj, h5py.Group):
        # Some MATLAB v7.3 complex arrays appear as a group with real/imag datasets.
        lower_keys = {k.lower(): k for k in obj.keys()}
        if "real" in lower_keys and "imag" in lower_keys:
            real = _h5_to_python(obj[lower_keys["real"]])
            imag = _h5_to_python(obj[lower_keys["imag"]])
            return {"real": real, "imag": imag}
        return {k: _h5_to_python(obj[k]) for k in obj.keys()}

    return np.array(obj)


def _load_mat_dict(path: Path) -> Dict[str, Any]:
    """Load classic or v7.3 MATLAB .mat file into a python dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if sio is not None:
        try:
            return sio.loadmat(str(path))
        except NotImplementedError:
            # MATLAB v7.3: fall through to h5py.
            pass
        except ValueError as e:
            # Some scipy versions throw ValueError for v7.3/HDF5.
            if "unknown mat file type" not in str(e).lower() and "hdf" not in str(e).lower():
                raise
        except Exception as e:
            # Do not hide real classic-MAT errors unless h5py can actually open it.
            if h5py is None:
                raise
            try:
                with h5py.File(str(path), "r"):
                    pass
            except Exception:
                raise e

    if h5py is None:
        raise ImportError(f"{path} may be MATLAB v7.3, but h5py is not installed")

    out: Dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for k in f.keys():
            out[k] = _h5_to_python(f[k])
    return out


def _visible_keys(d: Dict[str, Any]) -> List[str]:
    return [k for k in d.keys() if not k.startswith("__")]


def _load_mat_first_key(path: Path, key_candidates: Sequence[str]) -> Any:
    d = _load_mat_dict(Path(path))
    for k in key_candidates:
        if k in d:
            return _unwrap_mat_value(d[k])
    raise KeyError(
        f"None of the candidate keys {list(key_candidates)} found in {path}. "
        f"Keys present: {_visible_keys(d)}"
    )


def _to_complex_ndarray(arr: Any) -> np.ndarray:
    """Convert common MATLAB complex encodings to a np.complex64 ndarray."""
    arr = _unwrap_mat_value(arr)

    if isinstance(arr, dict):
        lower = {str(k).lower(): k for k in arr.keys()}
        if "real" in lower and "imag" in lower:
            real = np.asarray(arr[lower["real"]])
            imag = np.asarray(arr[lower["imag"]])
            return (real + 1j * imag).astype(np.complex64)
        raise TypeError(f"Cannot convert dict to complex array; keys={list(arr.keys())}")

    arr = np.asarray(arr)

    if arr.dtype.fields is not None:
        fields = {k.lower(): k for k in arr.dtype.fields.keys()}
        if "real" in fields and "imag" in fields:
            return (arr[fields["real"]] + 1j * arr[fields["imag"]]).astype(np.complex64)
        raise TypeError(f"Unsupported structured dtype for complex array: {arr.dtype}")

    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)

    # Common real/imag channel conventions.
    if arr.ndim >= 3 and arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.number):
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)
    if arr.ndim >= 3 and arr.shape[0] == 2 and np.issubdtype(arr.dtype, np.number):
        return (arr[0, ...] + 1j * arr[1, ...]).astype(np.complex64)

    # A purely real k-space array is unusual but still valid for debugging.
    return arr.astype(np.complex64)


# -----------------------------------------------------------------------------
# Shape / indexing helpers
# -----------------------------------------------------------------------------


def _strip_trivial_dims(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return np.squeeze(arr)


def _infer_hw_axes(arr: np.ndarray) -> Tuple[int, int]:
    """Infer spatial H,W axes as the two largest dimensions.

    This is deliberately robust to Mapping layouts like [H,W,Z,TI] and HDF5
    reversed layouts like [TI,Z,W,H]. The output order is largest axis first,
    second-largest axis second, giving H≈512 and W≈144 for your data.
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"Need at least a 2D array, got shape {arr.shape}")
    sizes = sorted(
        [(i, int(arr.shape[i])) for i in range(arr.ndim)],
        key=lambda t: (-t[1], t[0]),
    )
    return int(sizes[0][0]), int(sizes[1][0])


def _canonical_permute(arr: np.ndarray, hw_axes: Tuple[int, int]) -> List[int]:
    """Return permutation putting non-spatial frame axes first and H,W last."""
    other = [i for i in range(arr.ndim) if i not in hw_axes]
    # Largest non-spatial dimension first keeps [Z,TI] or [TI,Z] consistent and deterministic.
    other = sorted(other, key=lambda i: (-int(arr.shape[i]), i))
    return other + [int(hw_axes[0]), int(hw_axes[1])]


def _frame_count(arr: np.ndarray, hw_axes: Optional[Tuple[int, int]] = None) -> int:
    arr = _strip_trivial_dims(arr)
    if arr.ndim <= 2:
        return 1
    if hw_axes is None:
        hw_axes = _infer_hw_axes(arr)
    arrp = np.transpose(arr, _canonical_permute(arr, hw_axes))
    return int(np.prod(arrp.shape[:-2])) if arrp.ndim > 2 else 1


def _select_frame_hw(arr: np.ndarray, frame_idx: int, hw_axes: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Select one true [H,W] frame from an arbitrary-dimensional array."""
    arr = _strip_trivial_dims(arr)
    if arr.ndim == 2:
        return arr
    if hw_axes is None:
        hw_axes = _infer_hw_axes(arr)
    arrp = np.transpose(arr, _canonical_permute(arr, hw_axes))
    H, W = int(arrp.shape[-2]), int(arrp.shape[-1])
    frames = int(np.prod(arrp.shape[:-2]))
    if not (0 <= int(frame_idx) < frames):
        raise IndexError(f"frame_idx={frame_idx} out of range for {frames} frames from shape {arr.shape}")
    return arrp.reshape(frames, H, W)[int(frame_idx)]


def _match_hw_2d(arr: np.ndarray, H: int, W: int, *, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (H, W):
        return arr
    if arr.ndim == 2 and arr.T.shape == (H, W):
        return arr.T
    raise ValueError(f"{name} shape {arr.shape} does not match expected {(H, W)}")


def _prepare_mask(mask: Optional[np.ndarray], H: int, W: int, frame_idx: int = 0) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if np.iscomplexobj(m):
        m = np.abs(m)
    m = np.squeeze(m)

    if m.ndim == 0:
        return None
    if m.ndim == 1:
        if m.shape[0] == W:
            return np.tile(m[None, :], (H, 1)).astype(np.float32)
        if m.shape[0] == H:
            return np.tile(m[:, None], (1, W)).astype(np.float32)
        raise ValueError(f"1D mask length {m.shape[0]} matches neither H={H} nor W={W}")
    if m.ndim == 2:
        return _match_hw_2d(m, H, W, name="mask").astype(np.float32)

    # Per-frame mask: select using same generic frame logic.
    m2 = _select_frame_hw(m, frame_idx)
    return _match_hw_2d(m2, H, W, name="mask frame").astype(np.float32)


def _parse_acc_list(x: Union[str, int, Sequence[int], None], default: int = 4) -> List[int]:
    if x is None:
        return [int(default)]
    if isinstance(x, int):
        return [int(x)]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    s = str(x).strip()
    if "," in s:
        return [int(t.strip()) for t in s.split(",") if t.strip()]
    return [int(s)]


def _acc_to_2digits(acc_factor: Union[str, int]) -> str:
    s = str(acc_factor).strip().lower().replace("accfactor", "")
    if s in {"full", "fullsample", "fs", "1", "01"}:
        return "01"
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return s
    return digits.zfill(2)


def _resolve_acc_root(root: Path, acc_factor: Union[str, int]) -> Path:
    """Resolve legacy root=TrainingSet or current acc_root=AccFactorXX/AccFactor04."""
    root = Path(root)
    if any(p.is_dir() and p.name.startswith("P") for p in root.iterdir() if root.exists()):
        return root
    af = _acc_to_2digits(acc_factor)
    candidates = [
        root / "AccFactorXX" / f"AccFactor{af}",
        root / f"AccFactor{af}",
        root / "TrainingSet" / "AccFactorXX" / f"AccFactor{af}",
        root / "TrainingSet" / f"AccFactor{af}",
    ]
    for c in candidates:
        if c.exists():
            return c
    return root


def _find_case_dirs(acc_root: Path) -> List[Path]:
    if not acc_root.exists():
        raise FileNotFoundError(f"acc_root does not exist: {acc_root}")
    p_cases = sorted([p for p in acc_root.iterdir() if p.is_dir() and p.name.startswith("P")])
    if p_cases:
        return p_cases
    case_dirs = sorted([p for p in acc_root.iterdir() if p.is_dir() and p.name != "FullSample"])
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under {acc_root}")
    return case_dirs


@dataclass(frozen=True)
class SampleMeta:
    case_id: str
    shape_full: Tuple[int, ...]
    hw_axes_full: Tuple[int, int]
    num_frames: int


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class DimoKspaceDataset(Dataset):
    def __init__(
        self,
        acc_root: Optional[Union[str, Path]] = None,
        *,
        # Compatibility aliases/options used by older scripts.
        root: Optional[Union[str, Path]] = None,
        case_dirs: Optional[Sequence[Union[str, Path]]] = None,
        acc_factor: Union[str, int] = "04",
        coil_type: Optional[str] = None,
        split: Optional[str] = None,
        multi_coil: bool = False,
        use_full_as_target: bool = True,
        target_mode: str = "complex",
        cond_mode: str = "zf_mask",
        simulate_mask: bool = False,
        sim_accel: Union[int, str, Sequence[int], None] = 4,
        sim_mask_type: str = "out_r_1D",
        sim_seed: int = 0,
        sim_center: bool = True,
        sim_vary_per_slice: bool = False,
        sim_fixed_mask: Optional[bool] = None,
        limit_cases: Optional[int] = None,
    ) -> None:
        super().__init__()

        del coil_type, split  # accepted only for legacy constructor compatibility

        self.acc_factor = _acc_to_2digits(acc_factor)
        self.multi_coil = bool(multi_coil)
        self.use_full_as_target = bool(use_full_as_target)
        self.target_mode = str(target_mode).lower()
        self.cond_mode = str(cond_mode).lower()
        self.simulate_mask = bool(simulate_mask)
        self.sim_accel_list = _parse_acc_list(sim_accel, default=int(self.acc_factor or 4))
        self.sim_mask_type = str(sim_mask_type)
        self.sim_seed = int(sim_seed)
        if sim_fixed_mask is None:
            self.sim_vary_per_slice = bool(sim_vary_per_slice)
        else:
            self.sim_vary_per_slice = not bool(sim_fixed_mask)
        self.sim_center = bool(sim_center)

        if self.target_mode not in {"complex", "rss"}:
            raise ValueError("target_mode must be one of {'complex', 'rss'}")
        if self.cond_mode not in {"none", "zf", "zf_mask"}:
            raise ValueError("cond_mode must be one of {'none', 'zf', 'zf_mask'}")
        if self.multi_coil:
            raise NotImplementedError(
                "This unified loader currently targets the SingleCoil Mapping pipeline. "
                "Multi-coil reconstruction needs sensitivity maps/coils and should use a dedicated loader."
            )

        if case_dirs is not None:
            self.case_dirs = [Path(p) for p in case_dirs]
            if not self.case_dirs:
                raise FileNotFoundError("case_dirs is empty")
            self.acc_root = self.case_dirs[0].parent
        else:
            if acc_root is None:
                acc_root = root
            if acc_root is None:
                raise ValueError("Provide acc_root=..., case_dirs=..., or legacy root=...")
            self.acc_root = _resolve_acc_root(Path(acc_root), self.acc_factor)
            self.case_dirs = _find_case_dirs(self.acc_root)

        if limit_cases is not None:
            self.case_dirs = self.case_dirs[: int(limit_cases)]

        self.trainset_root = self._find_training_set_root(self.acc_root)
        self.full_root = self.trainset_root / "FullSample"

        self._index: List[Tuple[int, int]] = []
        self._meta_by_case: Dict[int, SampleMeta] = {}

        for ci, cdir in enumerate(self.case_dirs):
            k_full_all = _strip_trivial_dims(_to_complex_ndarray(self._load_full_kspace_all(cdir.name)))
            hw_axes = _infer_hw_axes(k_full_all)
            n_frames = _frame_count(k_full_all, hw_axes)
            self._meta_by_case[ci] = SampleMeta(
                case_id=cdir.name,
                shape_full=tuple(int(x) for x in k_full_all.shape),
                hw_axes_full=hw_axes,
                num_frames=int(n_frames),
            )
            for fi in range(n_frames):
                self._index.append((ci, fi))

        self._mask_gen = None
        if self.simulate_mask:
            try:
                from src.utils.mask_utils_v1 import generate_mask_1d
            except Exception:  # pragma: no cover
                from mask_utils import generate_mask_1d  # type: ignore
            self._mask_gen = generate_mask_1d

    @staticmethod
    def _find_training_set_root(p: Path) -> Path:
        cur = Path(p).resolve()
        for _ in range(12):
            if (cur / "FullSample").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        # Common fallback: AccFactor04 parent may be AccFactorXX, whose parent is TrainingSet.
        if Path(p).name.lower().startswith("accfactor") and Path(p).parent.name.lower() == "accfactorxx":
            return Path(p).parent.parent
        return Path(p).resolve().parent

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # MAT key loading
    # ------------------------------------------------------------------

    def _load_full_kspace_all(self, case_id: str) -> np.ndarray:
        paths = [
            self.full_root / case_id / "T1map.mat",
            self.full_root / case_id / "T1map_full.mat",
        ]
        p = next((x for x in paths if x.exists()), None)
        if p is None:
            raise FileNotFoundError(f"Full-sample MAT not found for case {case_id}; tried {paths}")
        return _load_mat_first_key(
            p,
            [
                "kspace_single_full",
                "kspace_full",
                "FullSample_kspace",
                "kspace_single",
                "kspace",
            ],
        )

    def _load_und_kspace_all(self, acc_case_dir: Path) -> np.ndarray:
        # Important: challenge undersampled k-space is usually in T1map_sub.mat.
        paths = [acc_case_dir / "T1map_sub.mat", acc_case_dir / "T1map.mat"]
        p = next((x for x in paths if x.exists()), None)
        if p is None:
            raise FileNotFoundError(f"Undersampled MAT not found in {acc_case_dir}; tried {paths}")

        af = self.acc_factor
        af_int = str(int(af)) if af.isdigit() else af
        return _load_mat_first_key(
            p,
            [
                f"kspace_single_sub{af}",
                f"kspace_sub{af}",
                f"kspace_single_sub{af_int}",
                f"kspace_sub{af_int}",
                "kspace_single_sub",
                "kspace_sub",
                "Undersampled_kspace",
                "kspace_single",
                "kspace",
            ],
        )

    def _load_mask_all(self, acc_case_dir: Path) -> Optional[np.ndarray]:
        p = acc_case_dir / "T1map_mask.mat"
        if not p.exists():
            return None
        af = self.acc_factor
        af_int = str(int(af)) if af.isdigit() else af
        try:
            raw = _load_mat_first_key(
                p,
                [
                    f"mask{af}",
                    f"mask{af_int}",
                    "T1map_mask",
                    "mask_sampling",
                    "kspace_mask",
                    "mask",
                ],
            )
            return np.asarray(raw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _generate_sim_mask(self, H: int, W: int, frame_idx: int) -> np.ndarray:
        if self._mask_gen is None:
            raise RuntimeError("simulate_mask=True but mask generator is unavailable")
        accel = self.sim_accel_list[(int(frame_idx) + self.sim_seed) % len(self.sim_accel_list)]
        seed = self.sim_seed + (int(frame_idx) if self.sim_vary_per_slice else 0)
        fn = self._mask_gen
        try:
            m = fn(H=H, W=W, accel=int(accel), mask_type=self.sim_mask_type, seed=seed, center=self.sim_center)
        except TypeError:
            m = fn(H, W, accel=int(accel), mask_type=self.sim_mask_type, seed=seed, center=self.sim_center)
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = np.asarray(m, dtype=np.float32)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        return _prepare_mask(m, H, W, frame_idx=0).astype(np.float32)  # type: ignore[union-attr]

    def _load_case_frame(self, case_idx: int, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        acc_case_dir = self.case_dirs[int(case_idx)]
        case_id = acc_case_dir.name

        k_full_all = _strip_trivial_dims(_to_complex_ndarray(self._load_full_kspace_all(case_id)))
        k_full_slice = _select_frame_hw(k_full_all, int(frame_idx), hw_axes=_infer_hw_axes(k_full_all))
        if k_full_slice.ndim != 2:
            raise ValueError(f"Full k-space frame is not 2D after selection: {k_full_slice.shape}")
        H, W = map(int, k_full_slice.shape)

        if self.simulate_mask:
            mask_hw = self._generate_sim_mask(H, W, frame_idx)
            k_und_slice = k_full_slice * mask_hw
        else:
            k_und_all = _strip_trivial_dims(_to_complex_ndarray(self._load_und_kspace_all(acc_case_dir)))
            k_und_slice = _select_frame_hw(k_und_all, int(frame_idx), hw_axes=_infer_hw_axes(k_und_all))
            k_und_slice = _match_hw_2d(k_und_slice, H, W, name=f"undersampled k-space for {case_id}")

            mask_all = self._load_mask_all(acc_case_dir)
            mask_hw = _prepare_mask(mask_all, H, W, frame_idx=int(frame_idx)) if mask_all is not None else None
            if mask_hw is None:
                # Robust fallback: infer the sampling mask directly from the k-space zeros.
                mask_hw = (np.abs(k_und_slice) > 0).astype(np.float32)
            mask_hw = _match_hw_2d(mask_hw, H, W, name=f"mask for {case_id}").astype(np.float32)

            # Ensure no accidental measured values outside the declared/acquired mask.
            k_und_slice = k_und_slice * mask_hw

        return (
            k_und_slice.astype(np.complex64, copy=False),
            mask_hw.astype(np.float32, copy=False),
            k_full_slice.astype(np.complex64, copy=False),
        )

    # ------------------------------------------------------------------
    # Torch sample output
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_idx, frame_idx = self._index[int(idx)]
        k_und, mask_hw, k_full = self._load_case_frame(case_idx, frame_idx)

        # Use full target for supervised reconstruction by default; keep the old flag for compatibility.
        k_target = k_full if self.use_full_as_target else k_und

        k_und_t = torch.from_numpy(k_und).to(torch.complex64)
        k_target_t = torch.from_numpy(k_target).to(torch.complex64)
        mask_t = torch.from_numpy(mask_hw).float()  # [H,W]

        y_2ch = complex_to_2ch(k_und_t).float()  # [2,H,W]
        zf_2ch = complex_to_2ch(ifft2c(k_und_t)).float()
        img_target_c = ifft2c(k_target_t)

        if self.target_mode == "complex":
            x0_2ch = complex_to_2ch(img_target_c).float()
        else:
            mag = torch.abs(img_target_c).float()
            x0_2ch = torch.stack([mag, torch.zeros_like(mag)], dim=0)

        out: Dict[str, torch.Tensor] = {
            "kspace_und_2ch": y_2ch,
            "y_2ch": y_2ch,
            "kspace_und": y_2ch,  # legacy alias, still 2-channel
            "mask": mask_t,
            "img_target_2ch": x0_2ch,
            "x0_2ch": x0_2ch,
            "x_target_2ch": x0_2ch,
            "img_target": x0_2ch,
            "zf_2ch": zf_2ch,
            "case_idx": torch.tensor(case_idx, dtype=torch.int64),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int64),
            "slice_idx": torch.tensor(frame_idx, dtype=torch.int64),
            "weight_idx": torch.tensor(0, dtype=torch.int64),
        }

        if self.cond_mode == "zf":
            out["cond"] = zf_2ch.float()
            out["cond_2ch"] = out["cond"]
        elif self.cond_mode == "zf_mask":
            out["cond"] = torch.cat([zf_2ch.float(), mask_t.unsqueeze(0)], dim=0)
            out["cond_2ch"] = out["cond"]

        return out
