---
name: super-resolution
description: Diagnose image style and missing detail, reconstruct only what the requested look supports, and deliver verified high-resolution raster images on Apple Silicon with PyTorch MPS and Real-ESRGAN. Use when faces, materials, lines, or environments fail at native pixels despite high nominal resolution.
---

# Super-Resolution

Decide the intended visual language before writing a prompt. Semantic correctness, convincing appearance, and pixel enlargement are separate gates. Real-ESRGAN may enlarge only an approved master.

## Hard gate

- Require macOS on arm64 Apple Silicon and PyTorch MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Require strictly more than 50 GiB free on every native volume used by source, output, scratch, temporary files, runtime, weights, or generated images. Run `apple_preflight.py` before model loading. Use `stage_runtime.py` when needed; never delete user data.
- CUDA, CPU inference, Core ML conversion, interpolation-only output, lower resolution, and backend substitution are not supported.
- Read [apple-runtime.md](references/apple-runtime.md) before execution.

## 1. Classify on receipt

Read [visual-routing.md](references/visual-routing.md). From the image, user request, and references, fill the visual contract in [native-review.json](assets/native-review.json) before writing any generation prompt:

- source and requested look: camera photograph, render, illustration, graphic, or mixed; identify the photographic subtype or artistic medium only when visible;
- truth mode: faithful restoration (source or same-subject evidence), creative reconstruction (plausible new detail), or evidence preserving (reversible enhancement only);
- source-supported structure and detail versus missing, smeared, compressed, deliberately simplified, or optically blurred areas;
- locked composition, identities, wording, geometry, color intent, focus, and other protected content.

If the user did not request a change of look, preserve the source look. Never claim an invented face or object detail was recovered. Do not force photorealism onto art, sharpen intended depth blur, or add generic noise as a substitute for material detail. Normalize the exact aspect ratio with `normalize_canvas.py` only after checking the crop; never stretch or silently discard important content.

## 2. Map risk and choose work

At 100%, inspect a representative sharp region and the worst salient region for people, anatomy, fabric, architecture, foliage, reflections, background, text, and boundaries where present. Record every acceptance-critical person in a subject manifest. Make one prioritized defect map: topology and protected content first, focal subjects and materials next, distant detail last. Preserve physically expected loss of detail from focus, distance, motion, haze, and illumination.

Before any generation, compare viable plans by expected image-edit calls, total submitted crop area, repeated review work, and MPS passes. `prepare_regions.py` records region count and gross crop coverage; this is a workload proxy, not a price quote. Choose the least work that still resolves every acceptance-critical defect. Do not start with local edits and later discover that a whole-image edit was necessary.

- **Clean lane:** source already has coherent semantics; approve and enlarge once.
- **Local lane:** defects are sparse and the global look is stable; batch only affected regions. Merge nearby defects when they share context and all faces retain working density. Avoid duplicate crop overlap.
- **Global lane:** defects or style conversion affect most of the composition, or many local edits would repeat the same context; at most one whole-image edit, then only indispensable high-density repairs.
- **Evidence-preserving lane:** when exact identity or detail is required but no supporting pixels or references exist, report that limit; do not invent a claimed restoration.

Use a crop trial only when its answer can change an expensive plan: uncertain style conversion, high-risk people/materials, or an untested prompt. Skip it for clean or straightforward isolated repairs. Compare the trial at intended viewing scale against the source/reference; correct the failed prompt layer once. If it remains implausible, stop before enlargement. A trial is diagnostic, not a deliverable.

For acceptance-critical photographic people, read [people-camera-realism.md](references/people-camera-realism.md). The semantic-stage face-width floors are fixed:

- 120 px or more: preferred;
- 80–119 px: permitted only when anatomy is already stable;
- below 80 px: mandatory contextual group or individual repair;
- below 48 px: creative reconstruction unless a sharper same-person reference exists.

Never lower a threshold to pass a report. Never shrink an accepted high-density face repair back into the low-resolution master. Bind it to the master as a repair pack and composite it once onto the final enlarged canvas.

## 3. Reconstruct with layered prompts

Use [prompt-system.md](references/prompt-system.md) in priority order: immutable content → camera and geometry → subject structure → material construction → style-specific optics or mark-making → only relevant exclusions. Use one look module and only the scene cues visible in the crop from [scene-modules.md](references/scene-modules.md). Prompt at the scale actually visible: pores or stitching belong only where the projected size supports them. Group nearby subjects only when every included face reaches the required working density; otherwise use individual regions.

Fix invalid topology before texture, but do not force a separate pass when structure is already valid. Read [defect-repair.md](references/defect-repair.md) only for actual structural failures. Register accepted regions with [sparse-region-repair.md](references/sparse-region-repair.md).

## 4. Gate before MPS

Run `build_contact_sheet.py --stage semantic` for acceptance-critical people. Its blur and duplicate metrics are prechecks only; they cannot approve facial semantics.

Create one native-pixel review using [native-review.json](assets/native-review.json), containing:

- source look, requested look, truth mode, evidence for the choice, and explicit appearance passes for style consistency, material construction, spatial detail hierarchy, and optics or mark-making;
- scene checks and representative regions;
- for people: explicit passes for eyes/gaze, mouth/teeth, skin texture, identity/distinctness, hair/hands/anatomy, and wardrobe/material integration;
- one status, delivery strategy, and concrete note for every subject.

`approve_semantic_master.py` rejects missing manual checks. A subject that fails density, blur, or duplicate precheck must use an accepted `final_registered_repair` in `--repair-manifest`.

Block MPS on malformed anatomy, smeared or repeated texture, material detail unrelated to form, pseudo-text, broken perspective, lost intentional blur, unsupported style conversion, or unregistered repairs. For photographic people, also block empty eyes, black mouth cavities, wax skin, and cloned faces. Sharpness never overrides a semantic failure.

## 5. Enlarge and finish once

Run `apple_mps_upscale.py` once. If a repair pack exists, apply it once with:

```bash
python scripts/composite_regions.py repair/regions-manifest.json final.png \
  --base mps-output.png --approval semantic-master.png.approval.json
```

Then inspect the exact final image at 100%, comparing the same risk regions to the approved master and intended look. Run `build_contact_sheet.py --stage final` for critical photographic people. Fill [final-review.json](assets/final-review.json) with passes and concrete notes for the delivered bytes; `finalize_delivery.py --final-review` rejects missing, failed, or stale reviews. Keep only the evidence listed in [delivery-evidence.md](references/delivery-evidence.md).

## Cost and retry rules

- Spend in this order: read-only inspection and plan → necessary crop trial → one batch of semantic edits → approval → one MPS run → final review. Fail fast at each gate.
- Reuse the source/target visual contract across regions; each prompt adds only the local defect and relevant constraints. Do not resend the full task history or multiple full-size previews for every crop.
- Maximum one whole-image semantic generation and one MPS invocation per approved plan. A failed crop trial does not justify a whole-image run.
- Generate all planned local repairs before MPS; composite them in one final operation.
- A failed region invalidates only that region. Retry it once after correcting the diagnosed relationship; never rerun the whole image to fix a local defect.
- If the retry still fails, stop and report the exact region and defect. Do not add quality adjectives, cascade new passes, or claim completion.
- Keep the defect map, trial, and review as compact as the actual risk allows; avoid variants that do not change a decision.

## Acceptance

The final pixels must preserve locked composition and content, match the requested look, and show coherent topology, material construction, scale-dependent detail, light, and depth. Photographic work must retain plausible optics and natural local variation; illustrated work must retain its own mark-making. A higher pixel count is not acceptance.
