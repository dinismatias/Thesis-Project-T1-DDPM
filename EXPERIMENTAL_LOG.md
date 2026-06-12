# Current thesis baseline

Model:
- checkpoints/dimo_cond_r4/epoch_0020.pt

Training:
- AF4 conditional model
- cond_mode = zf_mask
- target_mode = complex
- timesteps = 100
- final mean loss around 0.0119

Known working commands:
- Stage-1 denoise
- Stage-2 sampling from ZF
- Stage-2 from Stage-1
- replace DC working
- CG DC not yet strong

Important note:
- For Stage-2 output, use x_rec_2ch.
- x_in_2ch is the input to Stage-2, not the final Stage-2 result.
