# Super-Resolution

Structure-first semantic reconstruction and verified neural super-resolution for Apple Silicon.

Super-Resolution repairs what ordinary upscalers cannot: malformed geometry, empty materials, tiny synthetic faces, cloned people, broken lines, and other semantic defects that remain visible even in a high-resolution file. It then performs verified Real-ESRGAN enlargement on Apple MPS.

## What it provides

- Whole-image tiling and sparse local repair with registration and feathered compositing
- A hard face pixel-density gate and per-person subject manifest
- Native-pixel face contact sheets, blur checks, and likely-duplicate detection
- Structure-first prompting for people, architecture, products, art, and mixed scenes
- Deterministic preservation of text, logos, diagrams, and interfaces
- Exact aspect-ratio normalization without stretching
- SHA-bound semantic approval and a unified final delivery report
- Runtime staging when the system volume lacks safe working space

## Requirements

- Apple M-series Mac (`macOS`, `arm64`)
- More than 50 GiB free on every volume used for source, output, temporary data, runtime, weights, or generated images
- Python 3.9+
- PyTorch with Apple MPS available

CPU, CUDA, Core ML conversion, and interpolation-only fallbacks are intentionally unsupported.

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

Run commands from the repository root with absolute paths.

### 1. Preflight every storage path

```bash
python scripts/apple_preflight.py --check-mps --include-codex-imagegen \
  --runtime /safe/volume/sr-runtime \
  --report /safe/volume/job/preflight.json \
  /absolute/input.png /safe/volume/job /absolute/output.png

python scripts/stage_runtime.py /safe/volume/sr-runtime
```

### 2. Normalize the target canvas

```bash
python scripts/normalize_canvas.py input.png normalized.png \
  --target-width 3840 --target-height 2160
```

This performs an exact center crop only. It never stretches or silently changes geometry.

### 3. Choose reconstruction scope

Use overlapping tiles for broad defects:

```bash
python scripts/prepare_tiles.py normalized.png workspace/tiles \
  --cols 4 --rows 4 --overlap 192
```

Use local regions for small people or isolated defects:

```bash
python scripts/prepare_regions.py normalized.png assets/regions.json workspace/regions
```

For acceptance-critical people, copy `assets/subject-manifest.json`, define every subject and face box, and follow [camera-real people reconstruction](references/people-camera-realism.md). Faces narrower than 80 pixels require local repair; below 48 pixels they require individual or small-group creative reconstruction unless a sharper identity reference exists.

Record accepted tile stages with `record_tile.py`, or accepted local regions with `record_region.py`. Register and assemble them with `stitch_tiles.py` or `composite_regions.py`.

### 4. Approve at native pixels

```bash
python scripts/build_contact_sheet.py semantic-master.png \
  subject-manifest.json workspace/people-review

python scripts/approve_semantic_master.py semantic-master.png native-review.json \
  --subject-report workspace/people-review/contact-sheet-report.json
```

The automated report does not replace manual anatomy review. A failing report blocks approval.

### 5. Enlarge on Apple MPS

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/apple_mps_upscale.py \
  semantic-master.png output.png \
  --target-width 7680 --target-height 4320 \
  --approval semantic-master.png.approval.json \
  --runtime /safe/volume/sr-runtime \
  --tile 256 --tile-pad 24
```

### 6. Finalize the exact delivered bytes

After all local composites and exports are complete:

```bash
python scripts/finalize_delivery.py output.png delivery.json \
  --width 7680 --height 4320 \
  --artifact semantic-master=semantic-master.png \
  --operation-report output.png.mps-report.json
```

See [SKILL.md](SKILL.md) for the complete decision contract and [delivery evidence](references/delivery-evidence.md) for required proof.

## Theoretically unbounded resolution

The bounded-tile architecture has no fixed conceptual canvas limit: larger outputs can be produced by processing more registered regions. Practical limits remain storage, unified memory, runtime, file-format limits, model context, and cross-region consistency.

Unbounded pixels do not mean unbounded semantic truth. A 12K image can still contain a 20-pixel face with no recoverable identity. Semantic detail must be reconstructed at sufficient local subject scale, explicitly reviewed, and never misrepresented as source evidence.

## License

Project-authored files are licensed under [Apache-2.0](LICENSE). Third-party models and adapted components retain their own licenses and attribution.
