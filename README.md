# Super-Resolution

Structure-first semantic reconstruction and verified neural super-resolution for Apple Silicon.

Super-Resolution is a reusable Codex skill and Python pipeline for images that are already large but still break down under close inspection. It separates structural repair, semantic detail reconstruction, native-pixel approval, and final Real-ESRGAN enlargement so malformed content is not merely sharpened.

## Highlights

- Repairs geometry, perspective, repetition, occlusion, and material structure before enlargement.
- Tracks overlapping tiles through separate structural and semantic review stages.
- Preserves exact text, logos, diagrams, and interfaces through deterministic restoration.
- Binds approval to the semantic master by path and SHA-256 hash.
- Runs Real-ESRGAN on PyTorch MPS with CPU and CUDA fallback disabled.
- Produces registration, model, dimension, and checksum evidence for final verification.

## Requirements

- macOS on an Apple M-series (`arm64`) Mac
- More than 50 GiB free on every volume used by the job
- Python 3.9 or newer
- PyTorch with Apple MPS available

Core ML Tools is not required. Intel Macs, CPU inference, and CUDA are not supported.

## Installation

```bash
git clone https://github.com/showyell-del/Super-Resolution.git \
  ~/.codex/skills/super-resolution
cd ~/.codex/skills/super-resolution

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository includes the minimal MPS inference runtime and verified `RealESRGAN_x4plus.pth` weight required by the pipeline.

## Usage

Run all commands from the repository root with the virtual environment active. Replace the example paths and dimensions with values for your image.

### 1. Preflight and tile preparation

```bash
python scripts/apple_preflight.py --check-mps \
  /absolute/path/input.png \
  /absolute/path/workspace \
  /absolute/path/output.png

python scripts/prepare_tiles.py \
  /absolute/path/input.png \
  /absolute/path/workspace/tiles \
  --cols 4 --rows 4 --overlap 192
```

Complete the generated `scene_anchor.json` before editing. Increase tile density where small subjects, complex geometry, repeated structures, or material detail require more local resolution.

### 2. Reconstruct each tile

Create a structure-only edit first, inspect it at native pixels, and record the accepted result:

```bash
python scripts/record_tile.py \
  /absolute/path/workspace/tiles/manifest.json r1c1 \
  /absolute/path/generated/r1c1-structure.png \
  --stage structure \
  --prompt-file /absolute/path/prompts/r1c1-structure.txt \
  --review-note "Geometry, perspective, topology, and occlusion passed."
```

Then reconstruct semantic and material detail from the accepted structure:

```bash
python scripts/record_tile.py \
  /absolute/path/workspace/tiles/manifest.json r1c1 \
  /absolute/path/generated/r1c1-semantic.png \
  --stage semantic \
  --prompt-file /absolute/path/prompts/r1c1-semantic.txt \
  --review-note "Material construction and lighting response passed."
```

Repeat both stages for every tile. Use the [prompt system](references/prompt-system.md), [defect-repair rules](references/defect-repair.md), and only the relevant [scene modules](references/scene-modules.md). Exact text, logos, diagrams, and interfaces should be masked and restored with `scripts/restore_protected.py`, not regenerated.

### 3. Register and stitch

```bash
python scripts/stitch_tiles.py \
  /absolute/path/workspace/tiles/manifest.json \
  /absolute/path/workspace/semantic-master.png \
  --scratch-dir /absolute/path/workspace/scratch
```

The stitcher rejects unapproved tile stages, changed tile files, weak registration, excessive translation, and scale drift.

### 4. Approve the semantic master

Inspect representative regions at 100% native pixels, create the [required review JSON](references/defect-repair.md#approval-schema), then bind approval to the exact master file:

```bash
python scripts/approve_semantic_master.py \
  /absolute/path/workspace/semantic-master.png \
  /absolute/path/workspace/native-review.json
```

### 5. Run neural super-resolution

Target dimensions must be larger than the source and preserve its aspect ratio exactly.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/apple_mps_upscale.py \
  /absolute/path/workspace/semantic-master.png \
  /absolute/path/output.png \
  --target-width 12288 \
  --target-height 8192 \
  --approval /absolute/path/workspace/semantic-master.png.approval.json \
  --tile 256 --tile-pad 24
```

### 6. Verify the delivery

```bash
python scripts/verify_output.py \
  /absolute/path/output.png \
  /absolute/path/review-regions.json \
  /absolute/path/verification \
  --width 12288 --height 8192
```

Review the exported crops at 100%. If a region fails, repair its semantic tile and rebuild the master instead of hiding the defect with sharpening, grain, blur, or compression.

Tile editing is model-agnostic and is not bundled with this repository. The pipeline provides the prompting contract, review gates, protected-region handling, registration, MPS enlargement, and verification tools around that editing step.

## Theoretically unbounded resolution

The tile-based reconstruction workflow has no conceptual fixed canvas limit. By increasing the tile grid, processing bounded regions independently, and assembling them with overlap and registration, it can theoretically scale to arbitrarily large output dimensions—effectively “unlimited resolution” at the pipeline-design level.

This does not mean infinite pixels on a single machine. Practical output size is bounded by available storage, unified memory, processing time, image-format and library limits, model context, and cross-tile consistency. Higher resolution also does not create semantic truth automatically: every added region still requires structurally valid reconstruction and native-pixel review.

## Documentation

- [Skill contract](SKILL.md)
- [Structure-first defect repair](references/defect-repair.md)
- [Prompt system](references/prompt-system.md)
- [Scene modules](references/scene-modules.md)
- [Apple MPS runtime](references/apple-runtime.md)
- [Third-party notices](THIRD_PARTY.md)

## License

Project-authored files are licensed under [Apache-2.0](LICENSE). Adapted components and model assets retain their original licenses and attribution.
