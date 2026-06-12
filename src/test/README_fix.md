# DiMo compatibility patch

This patch focuses on the two scripts that are blocking your current workflow:

- `run_sampling_fixed.py`
- `run_denoise_fixed.py`

## What it fixes

### `run_sampling_fixed.py`
- accepts checkpoint schemas with `model_state`, `state_dict`, or `model`
- reads metadata from `config` or `args`
- instantiates `DiMoDDPM(T=...)` instead of `timesteps=...`
- always writes `x_in_2ch` inside `stage2_recon_output.pt`
- still supports `--dump_stage1_bundle`, and now writes both `x_in_2ch` and `zf_2ch` there
- prints `missing_keys` and `unexpected_keys` after checkpoint load

### `run_denoise_fixed.py`
- accepts checkpoint schemas with `model_state`, `state_dict`, or `model`
- reads metadata from `config` or `args`
- infers `cond_mode` / `cond_ch` from checkpoint metadata when possible
- supports conditional denoising with `zf` or `zf_mask`
- reads `zf_2ch` and `mask` from the input bundle/output dict
- uses `x_in_2ch` as fallback `zf_2ch` when the bundle is old and only has `x_in_2ch`
- keeps the old direct `.pt` workflow intact

## Drop-in usage

Replace your existing scripts with these files, or test them first as separate scripts.

### Stage-2 sampling
```bash
python -m src.test.run_sampling ^
  --acc_root "...\AccFactor04" ^
  --index 0 ^
  --ckpt "checkpoints/dimo_cond_r4/epoch_0020.pt" ^
  --cond_mode zf_mask ^
  --init_mode zf ^
  --stage2_strength 0.05 ^
  --num_steps 50 ^
  --dc_mode replace ^
  --log_residuals ^
  --dump_stage1_bundle "outputs/stage1_input_bundle.pt" ^
  --out_dir "outputs/stage2_run"
```

### Stage-1 denoise from bundle
```bash
python -m src.test.run_denoise ^
  --ckpt "checkpoints/dimo_simmask_r4/epoch_0020.pt" ^
  --input_pt "outputs/stage1_input_bundle.pt" ^
  --input_key x_in_2ch ^
  --strength 0.3 ^
  --num_steps 25 ^
  --out_dir "outputs/stage1_run"
```

### Stage-1 denoise directly from Stage-2 output
```bash
python -m src.test.run_denoise ^
  --ckpt "checkpoints/dimo_simmask_r4/epoch_0020.pt" ^
  --input_pt "outputs/stage2_run/stage2_recon_output.pt" ^
  --input_key x_in_2ch ^
  --strength 0.3 ^
  --num_steps 25 ^
  --out_dir "outputs/stage1_run"
```

Because `run_sampling_fixed.py` always saves `x_in_2ch` into `stage2_recon_output.pt`, the last command works even without a separate bundle.

## Important note

If `checkpoints/dimo_simmask_r4/epoch_0020.pt` is conditional and does **not** contain `config`/`args`, pass the conditioning explicitly:

```bash
--cond_mode zf_mask --cond_ch 3
```

or

```bash
--cond_mode zf --cond_ch 2
```

That is only needed for older checkpoints with incomplete metadata.
