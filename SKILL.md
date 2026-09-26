---
name: super-resolution
description: Reconstruct missing semantic detail and deliver verified 4K, 6K, 8K, 12K, or larger raster images on an Apple M-series Mac using PyTorch MPS and Real-ESRGAN. Use when nominal resolution is high but people, materials, lines, or environments fail at native pixels; not for resize-only work or non-Apple-Silicon computers.
---

# Super-Resolution

Separate semantic correctness from pixel enlargement. Real-ESRGAN may enlarge only an approved master; it must never be used to discover or repair malformed content.

## Hard gate

- Require macOS on arm64 Apple Silicon and PyTorch MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Require strictly more than 50 GiB free on every native volume used by source, output, scratch, temporary files, runtime, weights, or generated images. Run `apple_preflight.py` before model loading. Use `stage_runtime.py` when needed; never delete user data.
- CUDA, CPU inference, Core ML conversion, interpolation-only output, lower resolution, and backend substitution are not supported.
- Read [apple-runtime.md](references/apple-runtime.md) before execution.

## Truth mode

Choose exactly one:

- **Creative reconstruction:** plausible new detail while preserving declared invariants.
- **Faithful restoration:** only source- or same-subject-reference-supported detail.
- **Evidence preserving:** reversible, non-generative enhancement only.

Never call an invented face a recovered identity.

## One-plan workflow

### 1. Inspect once

Normalize the exact aspect ratio with `normalize_canvas.py`; never stretch. At 100% record:

- composition, camera, perspective, count, pose, lighting, focus, and protected content;
- every acceptance-critical person in a subject manifest;
- every failed face, body, garment, material, line system, repeated structure, reflection, foliage group, and focal environment region.

Create one repair plan before any expensive enlargement. Do not discover these regions one at a time during later reruns.

### 2. Choose the shortest lane

- **Clean lane:** if geometry, subjects, and materials already pass, approve the master and enlarge once.
- **Reconstruction lane:** allow at most one whole-image semantic edit. Then create all necessary high-density local repairs as one registered repair pack.

For people, read [people-camera-realism.md](references/people-camera-realism.md). The semantic-stage face-width floors are fixed:

- 120 px or more: preferred;
- 80–119 px: permitted only when anatomy is already stable;
- below 80 px: mandatory contextual group or individual repair;
- below 48 px: creative reconstruction unless a sharper same-person reference exists.

Never lower a threshold to pass a report. Never shrink an accepted high-density face repair back into the low-resolution master. Bind it to the master as a repair pack and composite it once onto the final enlarged canvas.

### 3. Reconstruct semantics

Use one compact contract from [prompt-system.md](references/prompt-system.md): locked invariants, failed structures, required physical detail, and forbidden defects. Group nearby subjects only when every included face reaches the required working density; otherwise use individual regions. Independent groups may run in parallel.

Fix invalid topology before texture, but do not force a separate pass when structure is already valid. Read [defect-repair.md](references/defect-repair.md) only for actual structural failures. Register accepted regions with [sparse-region-repair.md](references/sparse-region-repair.md).

### 4. Gate before MPS

Run `build_contact_sheet.py --stage semantic` for acceptance-critical people. Its blur and duplicate metrics are prechecks only; they cannot approve facial semantics.

Create one native-pixel review containing:

- scene checks and representative regions;
- for people: explicit passes for eyes/gaze, mouth/teeth, skin texture, identity/distinctness, hair/hands/anatomy, and wardrobe/material integration;
- one status, delivery strategy, and concrete note for every subject.

`approve_semantic_master.py` rejects missing manual checks. A subject that fails density, blur, or duplicate precheck must use an accepted `final_registered_repair` in `--repair-manifest`.

Block MPS on empty or asymmetric eyes, black mouth cavities, wax or globally blurred skin, cloned faces, fused anatomy, smeared fabric, textureless materials, melted lines, pseudo-text, broken perspective, or unregistered repairs. Sharpness never overrides a semantic failure.

### 5. Enlarge and finish once

Run `apple_mps_upscale.py` once. If a repair pack exists, apply it once with:

```bash
python scripts/composite_regions.py repair/regions-manifest.json final.png \
  --base mps-output.png --approval semantic-master.png.approval.json
```

Then inspect the exact final image at 100%, run `build_contact_sheet.py --stage final` for critical people, and generate the final report with `finalize_delivery.py`. Keep only the evidence listed in [delivery-evidence.md](references/delivery-evidence.md).

## Cost and retry rules

- Maximum one whole-image semantic generation and one MPS invocation per approved plan.
- Generate all planned local repairs before MPS; composite them in one final operation.
- A failed region invalidates only that region. Retry it once after correcting the diagnosed relationship; never rerun the whole image to fix a local defect.
- If the retry still fails, stop and report the exact region and defect. Do not add quality adjectives, cascade new passes, or claim completion.
- Do not create reports, crops, or prompt variants that do not drive a hard gate or final evidence.

## Acceptance

The final pixels must preserve locked composition and content; show coherent topology, perspective, repetition, light, materials, depth falloff, and protected elements; and contain distinct anatomically valid people without beauty-filter skin, blank eyes, black mouth holes, duplicated templates, halos, or cutout edges. A higher pixel count is not acceptance.
