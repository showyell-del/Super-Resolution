# Apple Silicon MPS runtime contract

This workflow uses the path validated in production: Real-ESRGAN inference through PyTorch MPS on an Apple M-series Mac. Core ML Tools is not required. There is no CPU or CUDA fallback.

## Mandatory preflight

Run `scripts/apple_preflight.py --check-mps` before any operation that can expand storage or unified-memory pressure. Check every distinct volume used by:

- the source image;
- the tile workspace;
- scratch or temporary files;
- the Real-ESRGAN runtime and weights;
- the final output.

Requirements:

- macOS;
- `arm64` architecture;
- CPU brand containing `Apple M`;
- strictly more than 50 GiB free on every involved native volume;
- PyTorch built with MPS and `torch.backends.mps.is_available()` returning true.

At exactly 50 GiB or below, stop. Do not begin and hope swap remains bounded. Do not offer a lower-resolution fallback, CPU inference, CUDA, Core ML conversion, ordinary interpolation, or another output size.

## Verified MPS executor

Use `scripts/apple_mps_upscale.py`. Its minimal Python runtime contains only the RRDBNet and tiled Real-ESRGAN inference path required by `RealESRGAN_x4plus.pth`; training code and non-MPS accelerator kernels are intentionally excluded. The command requires a hash-bound semantic-master approval created by `scripts/approve_semantic_master.py`; it must reject missing approvals, approvals for another file, and masters changed after review.

- Set `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing PyTorch.
- Require `torch.device("mps")`; abort if MPS is unavailable.
- Use the x4 RRDBNet model with tiled inference. The validated baseline is tile size 256, tile padding 24, no half precision, and no pre-padding.
- The x4 model performs neural reconstruction first. When the requested delivery scale is between native model scales, Real-ESRGAN may downsample the x4 neural result to the exact target, matching the validated workflow used for the prior 12K delivery.
- Preserve the source aspect ratio exactly. Reject target dimensions that do not match it.
- Record model hash, runtime, PyTorch version, MPS device, disabled fallback state, tile settings, source hash, output hash, dimensions, and elapsed time.

## Storage and stability

Unified memory, MPS allocations, decoded source pixels, neural x4 tiles, final PNG encoding, macOS swap, and generated semantic tiles can coexist temporarily. Re-run preflight immediately before loading the model because available space may have changed during reconstruction.

If free space crosses the threshold during a long job, stop at the next safe boundary and resume only after storage is freed. Do not delete user files automatically. The MPS executor must not be launched when preflight fails because prior low-space execution caused memory pressure severe enough to restart the Mac.

## Dependencies

The bundled runtime still requires a compatible Python environment with PyTorch, Pillow, NumPy, and OpenCV. Missing dependencies are blocking errors. Do not install Core ML Tools for this workflow.
