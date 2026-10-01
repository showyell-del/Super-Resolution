# Super-Resolution

Style-routed semantic reconstruction and verified neural enlargement for Apple Silicon.

The pipeline identifies the source and requested visual style, chooses the least editing work that meets the quality target, then selects a verified Apple Silicon route. Real-ESRGAN enlarges approved content; VOSR2 can creatively reconstruct missing photographic detail on MPS or a tested MPS–Neural Engine hybrid. Native-pixel review catches malformed faces, smeared clothing, false texture, broken lines, and style drift before delivery.

## Requirements

- Apple M-series Mac (`macOS`, `arm64`)
- A work volume for model files, caches, scratch, and output; preflight reports available space on it and the macOS swap volume
- Python 3.9+
- PyTorch with MPS available

CUDA, CPU-only image inference, interpolation-only delivery, and resolution fallback are intentionally unsupported. The ANE hybrid requires macOS 15+, Core ML Tools 9.0, and separately converted pinned model packages.

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

python scripts/normalize_canvas.py input.png normalized.png \
  --target-width 3840 --target-height 2160
```

The normalizer crops to the exact ratio and never stretches geometry. Run `stage_runtime.py /safe/volume/sr-runtime` only when that selected runtime needs a verified model copy; an already verified staged model is reused.

### 2. Diagnose style and make one repair plan

Inspect the normalized image at native pixels. Classify its look as photograph, render, illustration, graphic, or mixed, and record the requested target look and truth mode. Distinguish missing detail from intentional soft focus or artistic simplification. Record critical people and defective anatomy, garments, materials, lines, reflections, and focal environment regions before generation. See [visual routing](references/visual-routing.md) and the [layered prompt contract](references/prompt-system.md).

Compare a clean pass, sparse local repairs, and a broad whole-image edit before generating. The region manifest reports count and gross crop coverage to expose redundant work; it is a workload proxy, not an exact provider price. Test one representative crop only when the result could change the expensive plan. Reject smeared, pasted-on, or style-inconsistent detail before committing to a whole-image generation. Photographic faces below 80 pixels in the semantic master require a contextual group or individual repair; do not lower the threshold and do not shrink the accepted repair back into the master.

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

The review records the source and target look and passes checks for style, material construction, detail hierarchy, and optics or mark-making. The contact sheet measures face density, blur, and likely duplication, but never approves eyes, mouths, skin, identity, anatomy, or clothing. Those items are mandatory structured checks in `native-review.json` for photographic people.

### 4. Run the selected neural route once

For an approved master whose semantics are already sound, use Real-ESRGAN:

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
  --final-review final-review.json \
  --operation-report final.png.composite-report.json
```

Inspect the exact final pixels and fill [final-review.json](assets/final-review.json) with the final file's SHA-256, appearance passes, and inspected regions before delivery. If no repair pack is needed, review and finalize the MPS output directly. A local failure retries only that region; it never restarts the whole pipeline.

For a photographic target that permits invented detail, use VOSR2 after a passing 4x crop trial. Prefer the [MPS–ANE hybrid](references/vosr2-ane.md) when its packages and target-device compute plan are verified; the [MPS-only route](references/vosr2-apple.md) remains a separate choice, never a silent fallback. Large upstream weights live on the work volume, not in this repository. One M2/16 GiB test of a synthetic three-person scene reached 6144×4096 in 463 seconds wall time with the hybrid and passed native-pixel review; MPS-only took 672 seconds of model processing on the same scene. Backend execution and image quality are separate verdicts: an ANE compute plan proves neither photographic realism nor that a visual defect was caused by ANE. This does not establish universal image quality or full-ANE execution. Do not run Real-ESRGAN after VOSR2 or use VOSR2 to recreate exact identities, text, or logos.

## Resolution

Registered tiling has no fixed conceptual canvas limit, so the pipeline can theoretically produce arbitrarily large outputs. Practical limits remain storage, unified memory, model context, file formats, runtime, and cross-region consistency. More pixels never substitute for semantic truth.

See [SKILL.md](SKILL.md) for the execution contract.

## License

Project-authored files are licensed under [Apache-2.0](LICENSE). Third-party models and adapted components retain their own licenses and attribution.
