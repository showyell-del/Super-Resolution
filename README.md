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

## Workflow

1. Run the Apple Silicon, MPS, and storage preflight.
2. Split the source into overlapping tiles and define the global scene anchor.
3. Repair structural defects, then reconstruct semantic and material detail.
4. Register and stitch accepted tiles; restore protected regions deterministically.
5. Inspect the semantic master at native pixels and create a hash-bound approval.
6. Run the approved master through Real-ESRGAN on MPS.
7. Verify exact dimensions and representative 100% crops.

```bash
python scripts/apple_preflight.py --check-mps \
  /absolute/path/input.png \
  /absolute/path/workspace \
  /absolute/path/output.png

python scripts/prepare_tiles.py \
  /absolute/path/input.png \
  /absolute/path/workspace/tiles \
  --cols 4 --rows 4 --overlap 192

PYTORCH_ENABLE_MPS_FALLBACK=0 python scripts/apple_mps_upscale.py \
  /absolute/path/workspace/semantic-master.png \
  /absolute/path/output.png \
  --target-width 12288 \
  --target-height 8192 \
  --approval /absolute/path/workspace/semantic-master.png.approval.json
```

Tile editing is model-agnostic and is not bundled with this repository. The pipeline provides the prompting contract, review gates, protected-region handling, registration, MPS enlargement, and verification tools around that editing step.

## Documentation

- [Skill contract](SKILL.md)
- [Structure-first defect repair](references/defect-repair.md)
- [Prompt system](references/prompt-system.md)
- [Scene modules](references/scene-modules.md)
- [Apple MPS runtime](references/apple-runtime.md)
- [Third-party notices](THIRD_PARTY.md)

## License

Project-authored files are licensed under [Apache-2.0](LICENSE). Adapted components and model assets retain their original licenses and attribution.
