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
SR_WORK=/Volumes/work/Ai/Super-Resolution-runtime
mkdir -p "$SR_WORK/tmp" "$SR_WORK/pip-cache"
python3 -m venv "$SR_WORK/.venv"
source "$SR_WORK/.venv/bin/activate"
TMPDIR="$SR_WORK/tmp" PIP_CACHE_DIR="$SR_WORK/pip-cache" \
  python -m pip install -r requirements.txt
python scripts/stage_runtime.py "$SR_WORK"
```

Change `SR_WORK` to your work volume. In Codex, invoke `$super-resolution` with an image, target size, and whether invented detail is allowed. The skill routes the image, records native-pixel reviews, and runs only the selected backend. Real-ESRGAN's weight is bundled; VOSR2 and its ANE packages require the separate setup below.

## Workflow

For an approved master, `--preset 6k` or `--preset 12k` sets the long edge to 6144 or 12288 pixels and keeps the source composition. The selected image must be large enough for a neural scale of at most 4×: for a 3:2 landscape, at least 1536×1024 for 6K or 3072×2048 for 12K. No smaller-resolution substitute is produced.

```bash
python scripts/apple_mps_upscale.py approved.png result-6k.png \
  --preset 6k --approval approved.png.approval.json --runtime "$SR_WORK"
python scripts/apple_mps_upscale.py approved-3k.png result-12k.png \
  --preset 12k --approval approved-3k.png.approval.json --runtime "$SR_WORK"
```

These commands perform a single MPS neural pass and write a hash-bound runtime report. The 6K and 12K presets are output sizes, not guarantees that missing faces or materials can be recovered; finish with native-pixel review before delivery. For creative photographic reconstruction, use the separately gated VOSR2 route below.
Photographic people, or photographic enlargement above 2x, require a passing crop review before a full Real-ESRGAN run. Use `--trial-box X Y W H` to test an approved-source crop at 4x, inspect the actual pixels, then supply `--trial-review crop-review.json` for the full run. See the [crop review example](references/apple-runtime.md#photographic-crop-gate). A failed trial stops the render.
The 12K route requires a separately approved 3K-or-larger semantic master. Do not use a previous VOSR2 output as the input to another VOSR2 pass: repeating generative enlargement increases pixels without proving more authentic detail.

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
  --preset 6k \
  --approval semantic-master.png.approval.json \
  --runtime /safe/volume/sr-runtime

python scripts/composite_regions.py work/repair/regions-manifest.json final.png \
  --base mps-output.png --approval semantic-master.png.approval.json

python scripts/finalize_delivery.py final.png delivery.json \
  --width 6144 --height 3456 \
  --final-review final-review.json \
  --operation-report final.png.composite-report.json
```

Inspect the exact final pixels and fill [final-review.json](assets/final-review.json) with the final file's SHA-256, appearance passes, and inspected regions before delivery. If no repair pack is needed, review and finalize the MPS output directly. A local failure retries only that region; it never restarts the whole pipeline.

For a photographic target that permits invented detail, use VOSR2 after a passing 4x crop trial. Prefer the [MPS–ANE hybrid](references/vosr2-ane.md) when its packages and target-device compute plan are verified; the [MPS-only route](references/vosr2-apple.md) remains a separate choice, never a silent fallback. Large upstream weights live on the work volume, not in this repository. One M2/16 GiB test of a synthetic three-person scene reached 6144×4096 in 463 seconds wall time with the hybrid and passed native-pixel review; MPS-only took 672 seconds of model processing on the same scene. Backend execution and image quality are separate verdicts: an ANE compute plan proves neither photographic realism nor that a visual defect was caused by ANE. This does not establish universal image quality or full-ANE execution. Do not run Real-ESRGAN after VOSR2 or use VOSR2 to recreate exact identities, text, or logos.

The 12K backend has completed a 12288×8192 run on M2, but an independent real-photo crop test failed facial fidelity and material detail. Universal photographic 12K quality is not validated. A fresh public clone, isolated Python environment, dependency check, script regression, and real MPS inference were verified on M2/macOS 15.7.5; this installation check does not certify image appearance.

## Resolution

Registered tiling has no fixed conceptual canvas limit, so the pipeline can theoretically produce arbitrarily large outputs. Practical limits remain storage, unified memory, model context, file formats, runtime, and cross-region consistency. More pixels never substitute for semantic truth.

See [SKILL.md](SKILL.md) for the execution contract.

## License

Project-authored files are licensed under [Apache-2.0](LICENSE). Third-party models and adapted components retain their own licenses and attribution.
