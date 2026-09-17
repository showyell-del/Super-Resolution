# Super-Resolution

A reusable Codex skill and deterministic image pipeline for structure-first semantic reconstruction followed by Apple-Silicon neural super-resolution.

High pixel counts do not guarantee meaningful detail. This project separates the job into three enforced stages:

1. repair malformed topology, geometry, repeated systems, emitters, organic support, depth, and occlusion;
2. reconstruct scene-appropriate semantic and material detail;
3. approve the semantic master at native pixels, then run Real-ESRGAN through PyTorch MPS.

The MPS runner has no CPU, CUDA, Core ML, or interpolation-only fallback. It rejects an unapproved semantic master and invalidates approval when the master changes.

## Requirements

- macOS on an Apple M-series (`arm64`) Mac
- strictly more than 50 GiB free on every volume used for input, workspace, runtime, temporary files, and output
- Python 3.9 or newer
- PyTorch with `torch.backends.mps.is_available()` returning `True`

The validated environment used Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, Pillow 11.3.0, and OpenCV 5.0.0.

## Install as a Codex skill

```bash
git clone https://github.com/showyell-del/Super-Resolution.git ~/.codex/skills/super-resolution
cd ~/.codex/skills/super-resolution
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository includes a minimal MPS inference runtime adapted from pinned BasicSR and Real-ESRGAN revisions plus the official `RealESRGAN_x4plus.pth` weight. It intentionally excludes training code, CUDA extensions, C++ kernels, MATLAB tools, tests, examples, and unrelated models. Core ML Tools is not used or required.

## Workflow

Run all commands from the repository root with the virtual environment active.

### 1. Preflight

```bash
python scripts/apple_preflight.py --check-mps \
  /absolute/path/input.png \
  /absolute/path/workspace \
  /absolute/path/output
```

The command stops unless the machine, MPS runtime, and free-space requirements pass.

### 2. Prepare overlapping tiles

```bash
python scripts/prepare_tiles.py \
  /absolute/path/input.png \
  /absolute/path/workspace/tiles \
  --cols 4 --rows 4 --overlap 192
```

Complete `scene_anchor.json`. Use [the structure-first repair system](references/defect-repair.md), [the universal prompt scaffold](references/prompt-system.md), and only the relevant [scene modules](references/scene-modules.md).

### 3. Record the structural pass

After generating and reviewing a structure-only edit for a tile:

```bash
python scripts/record_tile.py \
  /absolute/path/workspace/tiles/manifest.json r1c1 \
  /absolute/path/generated/r1c1-structure.png \
  --stage structure \
  --prompt-file /absolute/path/prompts/r1c1-structure.txt \
  --review-note "Line topology, perspective, repetition, and occlusion passed at native pixels."
```

### 4. Record the semantic/material pass

The script refuses this stage until the structural pass is accepted:

```bash
python scripts/record_tile.py \
  /absolute/path/workspace/tiles/manifest.json r1c1 \
  /absolute/path/generated/r1c1-semantic.png \
  --stage semantic \
  --prompt-file /absolute/path/prompts/r1c1-semantic.txt \
  --review-note "Material construction and lighting response passed at native pixels."
```

Repeat both stages for every tile. Exact text, logos, diagrams, interfaces, and other protected regions should be masked and restored deterministically with `scripts/restore_protected.py`, not regenerated.

### 5. Register and stitch

```bash
python scripts/stitch_tiles.py \
  /absolute/path/workspace/tiles/manifest.json \
  /absolute/path/workspace/semantic-master.png \
  --scratch-dir /absolute/path/workspace/scratch
```

Each tile is registered against its source crop. Correlation, scale drift, and translation drift are checked before disk-backed feathering.

### 6. Approve the semantic master

Inspect representative regions at 100% native pixels and create the review JSON described in [the approval schema](references/defect-repair.md#approval-schema):

```bash
python scripts/approve_semantic_master.py \
  /absolute/path/workspace/semantic-master.png \
  /absolute/path/workspace/native-review.json
```

The resulting approval is bound to the exact master path and SHA-256 hash.

### 7. Run MPS neural super-resolution

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/apple_mps_upscale.py \
  /absolute/path/workspace/semantic-master.png \
  /absolute/path/output.png \
  --target-width 12288 \
  --target-height 8192 \
  --approval /absolute/path/workspace/semantic-master.png.approval.json \
  --tile 256 --tile-pad 24
```

Target dimensions must preserve the source aspect ratio. The runner records model identity, weight hash, MPS device, disabled fallback state, source and output hashes, tile settings, dimensions, and elapsed time.

### 8. Verify native-pixel crops

```bash
python scripts/verify_output.py \
  /absolute/path/output.png \
  /absolute/path/review-regions.json \
  /absolute/path/verification \
  --width 12288 --height 8192
```

Do not approve a result from a fit-to-screen preview. Repair failed semantic tiles and rebuild the master instead of hiding failures with blur, grain, sharpening, or compression.

## What this project does not do

- It does not treat upscaling as semantic or geometric repair.
- It does not fabricate evidence in documents, medical images, scientific imagery, or forensic material.
- It does not ask generative models to reproduce exact text, logos, diagrams, or interfaces.
- It does not run on Intel Macs or silently use CPU when MPS is unavailable.
- It does not continue when any involved volume has 50 GiB or less free space.

## Repository layout

- `SKILL.md` — Codex skill entry point and hard workflow contract
- `references/` — prompt system, defect-repair modules, scene modules, and Apple runtime contract
- `scripts/` — preflight, tile state, registration, protected-region restoration, approval, minimal MPS inference, and verification
- `runtime/weights/` — the verified x4 model weight
- `third_party/licenses/` — preserved upstream license texts
- `agents/openai.yaml` — Codex UI metadata

## License and third-party software

Project-authored files are licensed under Apache-2.0. Adapted third-party code and the model retain their licenses and attribution; see [THIRD_PARTY.md](THIRD_PARTY.md) and `third_party/licenses/`.
