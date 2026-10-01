# Apple Silicon MPS runtime contract

This workflow has Real-ESRGAN on PyTorch MPS and two purpose-specific VOSR2 routes after a passing 4x photographic crop trial: [MPS-only](vosr2-apple.md) and [MPS–ANE hybrid](vosr2-ane.md). Core ML Tools is required only to prepare and run the ANE hybrid. Choose the route before inference; there is no CPU-only, CUDA, or lower-resolution fallback.

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

Do not offer a lower-resolution fallback, CPU-only image inference, CUDA, ordinary interpolation, or another output size. The ANE route's Core ML conversion traces on CPU, while image inference uses MPS plus Core ML `CPU_AND_NE` with a small number of CPU-preferred operations.

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
- `apple_mps_upscale.py --preset 6k|12k` uses a 6144- or 12288-pixel long edge and rounds only the short edge to an integer pixel. It rejects a requested scale above the model's native 4×. The tiled executor copies each completed tile to an 8-bit output canvas instead of holding a full float32 output on MPS; a non-4× result uses a work-volume temporary native canvas that is removed after resampling. Small 4× and 2× parity fixtures were byte-identical to the earlier full-canvas implementation.
- For explicit dimensions, require the source aspect ratio exactly. Presets preserve it to the nearest integer output pixel; record both final dimensions.
- Record model hash, runtime, PyTorch version, MPS device, disabled fallback state, tile settings, source hash, output hash, dimensions, and elapsed time.

For VOSR2, use the pinned runtime patch and choose one exact 4x route: [MPS-only](vosr2-apple.md) or [MPS–ANE hybrid](vosr2-ane.md). The MPS-only run released the temporary checkpoint after loading and released DiT/DINOv2 from MPS before Qwen VAE decode; omitting those lifecycle steps caused severe swapping and unusable runtime. The hybrid uses separate Core ML packages for DiT and decode. A passing crop never substitutes for a full-resolution native-pixel review.

## Storage and stability

Unified memory, MPS allocations, decoded source pixels, neural x4 tiles, final PNG encoding, macOS swap, and generated semantic tiles can coexist temporarily. Re-run preflight immediately before loading the model because available space may have changed during reconstruction.

Low system-disk space can still cause memory pressure or a restart even when job files are external. Keep the report visible, stop on actual disk-full or MPS out-of-memory errors, and never delete user files automatically. There is no fixed free-space cutoff.

## Dependencies

The bundled runtime still requires a compatible Python environment with PyTorch, Pillow, NumPy, and OpenCV. Missing dependencies are blocking errors. Core ML Tools 9.0 is required for the ANE route but not for Real-ESRGAN or MPS-only VOSR2.
