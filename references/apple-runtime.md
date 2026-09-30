# Apple Silicon MPS runtime contract

This workflow has two verified, purpose-specific PyTorch MPS paths on an Apple M-series Mac: Real-ESRGAN for enlargement of an approved master, and [VOSR2](vosr2-apple.md) for photographic creative reconstruction after a passing 4x crop trial. Core ML Tools is not required. There is no CPU or CUDA fallback.

## Mandatory preflight

Run `scripts/apple_preflight.py --check-mps` before any operation that can expand storage or unified-memory pressure. Check every distinct volume used by:

- macOS swap (`/private/var/vm`), even if every job file is on an external drive;
- the source image;
- the tile workspace;
- scratch or temporary files;
- the selected neural runtime and weights;
- `CODEX_HOME/generated_images` when a semantic editor writes there;
- the final output.

Requirements:

- macOS;
- `arm64` architecture;
- CPU brand containing `Apple M`;
- report free space on the swap volume and every involved native volume without a fixed cutoff;
- PyTorch built with MPS and `torch.backends.mps.is_available()` returning true.

Do not offer a lower-resolution fallback, CPU inference, CUDA, Core ML conversion, ordinary interpolation, or another output size.

Use one command to check every path and retain the report:

```bash
python scripts/apple_preflight.py --check-mps --include-codex-imagegen \
  --runtime /safe/volume/super-resolution-runtime \
  --report /safe/volume/job/preflight.json \
  /absolute/source.png /safe/volume/job /absolute/output.png
```

For the Real-ESRGAN route, if the bundled weight is on a low-space volume, stage its verified bytes before inference:

```bash
python scripts/stage_runtime.py /safe/volume/super-resolution-runtime
```

Run staging only when the selected runtime needs it. A valid existing staged weight is hash-checked and reused without another copy. Pass that directory to `apple_mps_upscale.py --runtime`. The stage report and model hash are delivery evidence.

## Verified Real-ESRGAN MPS executor

Use `scripts/apple_mps_upscale.py`. Its minimal Python runtime contains only the RRDBNet and tiled Real-ESRGAN inference path required by `RealESRGAN_x4plus.pth`; training code and non-MPS accelerator kernels are intentionally excluded. The command requires a hash-bound semantic-master approval created by `scripts/approve_semantic_master.py`; it must reject missing approvals, approvals for another file, and masters changed after review.

- Set `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing PyTorch.
- Require `torch.device("mps")`; abort if MPS is unavailable.
- Use the x4 RRDBNet model with tiled inference. The validated baseline is tile size 256, tile padding 24, no half precision, and no pre-padding.
- The x4 model performs neural reconstruction first. When the requested delivery scale is between native model scales, Real-ESRGAN may downsample the x4 neural result to the exact target, matching the validated workflow used for the prior 12K delivery.
- Preserve the source aspect ratio exactly. Reject target dimensions that do not match it.
- Record model hash, runtime, PyTorch version, MPS device, disabled fallback state, tile settings, source hash, output hash, dimensions, and elapsed time.

For VOSR2, use only the pinned runtime patch and the exact 4x workflow in [vosr2-apple.md](vosr2-apple.md). The verified M2/16 GiB run released the temporary checkpoint after loading and released DiT/DINOv2 from MPS before Qwen VAE decode; omitting those lifecycle steps caused severe swapping and unusable runtime. A passing crop never substitutes for a full-resolution native-pixel review.

## Storage and stability

Unified memory, MPS allocations, decoded source pixels, neural x4 tiles, final PNG encoding, macOS swap, and generated semantic tiles can coexist temporarily. Re-run preflight immediately before loading the model because available space may have changed during reconstruction.

Low system-disk space can still cause memory pressure or a restart even when job files are external. Keep the report visible, stop on actual disk-full or MPS out-of-memory errors, and never delete user files automatically. There is no fixed free-space cutoff.

## Dependencies

The bundled runtime still requires a compatible Python environment with PyTorch, Pillow, NumPy, and OpenCV. Missing dependencies are blocking errors. Do not install Core ML Tools for this workflow.
