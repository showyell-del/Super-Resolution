# Super-Resolution

Style-aware semantic reconstruction and verified neural enlargement for Apple Silicon.

The pipeline separates **plausible detail generation** from **pixel enlargement**. It classifies the requested look, repairs only necessary regions, approves the actual pixels, then runs one selected neural route. More pixels are not proof of recovered detail or photographic realism. For nonphotographic 12K-detail work, use the [effective-detail acceptance standard](references/12k-quality.md): source information, projected repair density, native material structure, paired comparisons and every join are reviewed separately from backend completion.


For explicitly requested balanced creative enhancement, use [creative-balanced.md](references/creative-balanced.md): preserve facial structure, expression, composition and focus while accepting coherent invented texture. Keep strict photographic/effective-detail reviews as separate historical verdicts; do not present creative approval as authentic recovered detail.

## Resolution routing

The approved target look selects the limit: `camera_photo` supports a long edge up to **6144 pixels (6K)**; `render`, `illustration`, `graphic` and nonphotographic `mixed` support up to **12288 pixels (12K)**. A generated image with photographic style is also `camera_photo`. A scene with photographic subjects and graphic overlays follows the photographic route. Classify from actual pixels; do not relabel a photograph to bypass the limit.

Oversized requests are rejected before inference; explicit width/height, registered composites and final delivery obey the same limit. No automatic resize or route substitution. See [resolution routing](references/resolution-routing.md).

## Requirements

- Apple M-series Mac, macOS, Python 3.9+, working PyTorch MPS
- A work volume for runtimes, models, caches, scratch and results
- For the VOSR2 ANE hybrid: macOS 15+, Core ML Tools 9.0 and pinned converted packages

Preflight reports work-volume and system swap-volume space without a fixed threshold. CUDA, CPU-only image inference, silent backend/resolution substitution and interpolation-only enhancement are unsupported. Supported working modes are RGB, RGBA, L and LA; convert other modes explicitly with color management.

## Install

```bash
git clone https://github.com/showyell-del/Super-Resolution.git ~/.codex/skills/super-resolution
cd ~/.codex/skills/super-resolution
SR_WORK=/Volumes/work/Ai/Super-Resolution-runtime
mkdir -p "$SR_WORK/tmp" "$SR_WORK/pip-cache"
python3 -m venv "$SR_WORK/.venv"
source "$SR_WORK/.venv/bin/activate"
TMPDIR="$SR_WORK/tmp" PIP_CACHE_DIR="$SR_WORK/pip-cache" \
  python -m pip install -r requirements.txt
python scripts/stage_runtime.py "$SR_WORK"
```

Set `SR_WORK` to your work volume. In Codex, invoke `$super-resolution` with the image, target resolution and whether invented detail is permitted. Real-ESRGAN's weight is bundled; ANE setup is separate.

## Usage

1. **Inspect and route.** Preserve the intended medium, identities, wording, geometry and focus. Choose clean, local, global or evidence-preserving work. Initialize the small [job ledger](references/job-ledger.md) before trials or generation.
2. **Repair selectively.** Generate only documented failures. Keep native repair density for the final canvas. People repairs require output-local subject evidence with faces at least 120px; geometry and IDs must match the replacement target.
3. **Approve semantics.** Fill [native-review.json](assets/native-review.json), including the inspected image's SHA-256. For critical people, add a semantic subject report and explicit eyes/mouth/skin/anatomy/material checks. Scores do not approve appearance.
4. **Enlarge once.** Choose approved Real-ESRGAN MPS, or creative VOSR2 MPS–ANE after a passing 4× crop trial. Never cascade neural enlargement.
5. **Review final bytes.** Check every critical person, material, join and focus region at native pixels. Finalization checks execution lineage and all approved repairs. Each delivery directory must contain exactly one image.

For an approved Real-ESRGAN master:

```bash
python scripts/approve_semantic_master.py master.png native-review.json
python scripts/apple_mps_upscale.py master.png /work/job/delivery/final.png \
  --preset 6k --approval master.png.approval.json --runtime "$SR_WORK"
python scripts/finalize_delivery.py /work/job/delivery/final.png /work/job/delivery/delivery.json \
  --width 6144 --height 4096 --approval master.png.approval.json \
  --final-review /work/job/final-review.json \
  --operation-report /work/job/delivery/final.png.mps-report.json
```

The example assumes a 3:2 master. The 6K preset uses a 6144-pixel long edge; the 12K preset is limited to nonphotographic targets. Both preserve aspect to the nearest pixel. Source density must support at most 4× enlargement: at least a 1536-pixel long edge for 6K or 3072 for nonphotographic 12K. These are execution/source-density floors, not effective-detail guarantees. The commands do not generate the manual review.

Photographic people or photographic enlargement above 2× require a passing `--trial-review` before the full MPS run; see the [crop gate](references/apple-runtime.md#photographic-crop-gate). Add subject/repair evidence to approval and finalization when applicable. For repairs, use a separate neural base and `composite_regions.py --base --base-report --approval`, then pass both operation reports to finalization.

## Routes and limits

- **Real-ESRGAN MPS:** enlarges an approved semantic master; it cannot guarantee faithful new facial or material detail.
- **VOSR2 MPS–ANE hybrid:** reconstructs plausible photographic detail. Requires source approval, passing same-source crop review, pinned conversion evidence and a same-device package plan. It is not exact identity, logo or text restoration. See [ANE setup](references/vosr2-ane.md).
- **MPS-only VOSR2:** retained as a separate [diagnostic control](references/vosr2-apple.md), not a current hash-gated delivery entrypoint.

Compute-plan preferred devices are not execution traces or quality proof. Historical scene-specific successes and failures are recorded in [Agent.md](Agent.md); neither establishes universal realism or a promised runtime.

Registered repairs must support the selected delivery scale at native density. The style-specific delivery limits apply regardless of source size or tile count.

For nonphotographic targets, when both 12K and 6K are requested, derive the smaller version from an approved parent with `downsample_delivery.py`, then review and finalize it independently. See [delivery evidence](references/delivery-evidence.md) and [registered repairs](references/sparse-region-repair.md).

## Verify

```bash
mkdir -p /work/tests
SR_TEST_ROOT=/work/tests python -m unittest discover -s tests -v
python scripts/smoke_test.py /work/tests/new-empty-smoke-directory
```

Lightweight tests validate contracts and failure behavior, not photographic quality or actual ANE placement. Source approvals use version 4 and delivery reports version 2; older unbound reports must not be relabeled as current evidence.

## License

Project-authored files: [Apache-2.0](LICENSE). Third-party models and adapted components retain their licenses and attribution.
