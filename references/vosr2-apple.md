# VOSR2 photographic reconstruction on Apple MPS

Use this route only when the visual contract permits **creative** photographic detail reconstruction. It is not identity recovery, exact text restoration, or a substitute for source-supported geometry. The verified scale is 4x. Route selection is made before inference; this is not a fallback after another backend fails.

## Tested configuration

- Apple M2, 16 GiB unified memory; PyTorch 2.8.0 and torchvision 0.23.0 on MPS, with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Official [VOSR](https://github.com/cswry/VOSR) revision `516f292b99cf23c76fdc33351e86dc4f97711fe8`, [VOSR2](https://huggingface.co/CSWRY/VOSR/tree/main/VOSR2) 1.4B one-step checkpoint, Qwen-Image 2D VAE, and local [DINOv2](https://github.com/facebookresearch/dinov2) revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8` with ViT-L weights.
- `diffusers==0.35.0`, `timm==1.0.11`, `einops==0.8.0`; no Core ML Tools, CUDA, or CPU inference.
- 1536×1024 synthetic photographic source → 6144×4096 PNG. Model processing: 672.26 seconds; preflight-to-file wall-clock upper bound: 756 seconds. A different machine, image, or model may differ.

MPS uses the Apple GPU. This run does not demonstrate Apple Neural Engine execution.

## Prepare on the work volume

Run `apple_preflight.py --check-mps` on the source, runtime, scratch, and output paths. It reports both macOS swap-volume and work-volume free space; there is no fixed space cutoff. Keep runtime, caches, temporary files, and results on the work volume. Do not start a full image while another high-memory render is active.

Clone the pinned upstream checkouts on that volume. Apply [`patches/vosr2-mps.patch`](../patches/vosr2-mps.patch) to VOSR and, when using Python 3.9, [`patches/dinov2-python39.patch`](../patches/dinov2-python39.patch) to the local DINOv2 checkout. Both patches must apply cleanly; do not silently skip a hunk or substitute another revision. Place the DINOv2 checkout at `VOSR/preset/ckpts/torch_cache/facebookresearch_dinov2_main` and its official `dinov2_vitl14_pretrain.pth` at `VOSR/preset/ckpts/torch_cache/checkpoints/`. Download only `VOSR2/*` and `Qwen-Image-vae-2d/*` from `CSWRY/VOSR` into `VOSR/preset/ckpts/`. Do not download the entire model repository.

The VOSR patch forces MPS, disables CUDA/CPU selection and `torch.compile` on this path, uses the local DINOv2 checkout, frees the temporary checkpoint after loading, and removes the DiT/DINOv2 allocations from MPS before Qwen VAE decode. The last two changes are necessary on the tested 16 GiB Mac: without them, full-image decode drove heavy swapping and did not finish within the user's acceptable work time.

Create the Python environment on the work volume. The tested environment used Python 3.9, torch 2.8.0, torchvision 0.23.0, Pillow, NumPy, safetensors, huggingface_hub 0.36.2, plus the three pinned packages above. Confirm imports and MPS availability before model loading. The first model download is several gigabytes and is a one-time cost; keep Hugging Face cache on the work volume.

## Run only after the crop gate

Inspect one native-pixel crop that includes a salient face **and** the troublesome background or material. This test used a 512×512 source crop and checked its 2048×2048 result against the source and earlier rejected output. Reject a crop with painted skin, smeared construction, invented hard edges, or a discontinuous focus plane. A passing sharpness score is insufficient.

For the approved full image, from inside the patched VOSR checkout run:

```bash
PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
HF_HOME=/work-volume/vosr/.cache/huggingface \
TMPDIR=/work-volume/vosr/.cache/tmp \
python inference_vosr_onestep.py \
  -c preset/ckpts/VOSR2 \
  -i /absolute/approved-source.png \
  -o /absolute/empty-final-directory \
  -u 4 --tile_size 512 --tile_overlap 64 \
  --vae_tile_size 1024 --vae_tile_overlap 128
```

The output is saved under the input basename directly in the output directory. Keep exactly one full-resolution candidate there. At 100% native pixels inspect all critical faces, visible hands, wardrobe construction, glass and other repeated lines, foliage, focus falloff, and tile boundaries. Run `build_contact_sheet.py --stage final` for critical people, then complete `final-review.json` and `finalize_delivery.py` only if the candidate passes. The contact sheet's blur and duplicate metrics are prechecks, not an appearance verdict.

## Evidence from this test

The VOSR 0.5B one-step 6K candidate was rejected for coarse skin and glass artifacts. VOSR2's completed 6K candidate passed this test's three-face, material, geometry, and depth checks. This is one synthetic-scene test, not a universal model-quality claim. Keep future failures specific to their image and stage; do not turn a single artifact into a global prompt rule.
