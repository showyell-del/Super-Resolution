# Super-Resolution

Semantic reconstruction and verified neural enlargement for Apple Silicon.

The pipeline fixes missing meaning before adding pixels: malformed faces, smeared clothing, empty materials, broken lines, repeated synthetic detail, and weak environments are rejected before the expensive upscale. Real-ESRGAN then runs once on Apple MPS, followed by one registered high-density repair composite when needed.

## Requirements

- Apple M-series Mac (`macOS`, `arm64`)
- More than 50 GiB free on every volume used by the job
- Python 3.9+
- PyTorch with MPS available

CUDA, CPU inference, Core ML conversion, interpolation-only delivery, and resolution fallback are intentionally unsupported.

## Install

```bash
git clone https://github.com/showyell-del/Super-Resolution.git \
  ~/.codex/skills/super-resolution
cd ~/.codex/skills/super-resolution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Workflow

### 1. Preflight and normalize

```bash
python scripts/apple_preflight.py --check-mps --include-codex-imagegen \
  --runtime /safe/volume/sr-runtime \
  --report /safe/volume/job/preflight.json \
  /absolute/input.png /safe/volume/job /absolute/output.png

python scripts/stage_runtime.py /safe/volume/sr-runtime

python scripts/normalize_canvas.py input.png normalized.png \
  --target-width 3840 --target-height 2160
```

The normalizer crops to the exact ratio and never stretches geometry.

### 2. Make one repair plan

Inspect the normalized image at native pixels. Record every critical person and every defective anatomy, garment, material, line system, reflection, and focal environment region before generation.

Use the whole image once only when broad reconstruction is necessary. Faces below 80 pixels in the semantic master require a contextual group or individual repair; do not lower the threshold and do not shrink the accepted repair back into the master.

```bash
python scripts/prepare_regions.py normalized.png assets/regions.json work/repair
python scripts/record_region.py work/repair/regions-manifest.json region-name \
  generated-region.png --prompt-file region-prompt.txt \
  --review-note "Native-pixel anatomy and integration passed."
```

### 3. Approve semantics before enlargement

```bash
python scripts/build_contact_sheet.py semantic-master.png \
  subject-manifest.json work/people --stage semantic

python scripts/approve_semantic_master.py semantic-master.png native-review.json \
  --subject-report work/people/contact-sheet-report.json \
  --repair-manifest work/repair/regions-manifest.json
```

The contact sheet measures face density, blur, and likely duplication, but never approves eyes, mouths, skin, identity, anatomy, or clothing. Those items are mandatory structured checks in `native-review.json`.

### 4. Enlarge once and composite once

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/apple_mps_upscale.py \
  semantic-master.png mps-output.png \
  --target-width 7680 --target-height 4320 \
  --approval semantic-master.png.approval.json \
  --runtime /safe/volume/sr-runtime

python scripts/composite_regions.py work/repair/regions-manifest.json final.png \
  --base mps-output.png --approval semantic-master.png.approval.json

python scripts/finalize_delivery.py final.png delivery.json \
  --width 7680 --height 4320 \
  --operation-report final.png.composite-report.json
```

If no repair pack is needed, finalize the MPS output directly. A local failure retries only that region; it never restarts the whole pipeline.

## Resolution

Registered tiling has no fixed conceptual canvas limit, so the pipeline can theoretically produce arbitrarily large outputs. Practical limits remain storage, unified memory, model context, file formats, runtime, and cross-region consistency. More pixels never substitute for semantic truth.

See [SKILL.md](SKILL.md) for the execution contract.

## License

Project-authored files are licensed under [Apache-2.0](LICENSE). Third-party models and adapted components retain their own licenses and attribution.
