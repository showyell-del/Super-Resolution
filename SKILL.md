---
name: semantic-detail-upscale
description: Reconstruct scene-appropriate semantic detail in AI-generated, compressed, scanned, or over-smoothed raster images before delivering 6K, 8K, 12K, or larger outputs on an Apple M-series Mac with PyTorch MPS and Real-ESRGAN. Use when a nominally high-resolution image still lacks meaningful subject, material, or style detail under close inspection; not for ordinary resize-only requests or non-Apple-Silicon computers.
---

# Semantic Detail Upscale

Treat resolution and detail as separate problems. A larger pixel grid is not completion when the source lacks meaningful information.

## Hard runtime gate

- Run only on macOS with an arm64 Apple M-series processor.
- Final neural super-resolution must use the verified Real-ESRGAN x4 model through PyTorch MPS and Apple Metal acceleration. Set `PYTORCH_ENABLE_MPS_FALLBACK=0`; do not substitute CUDA, generic CPU inference, Core ML conversion, interpolation-only enlargement, or another backend.
- Before tile creation, registration, restoration, or model loading, run `scripts/apple_preflight.py` against the source, output, scratch, and temporary-storage locations.
- Every distinct native volume involved must have strictly more than 50 GiB free. If any volume is at or below 50 GiB, stop before allocating tiles or loading a model. This prevents unified-memory pressure and swap growth from destabilizing or restarting the Mac.
- Read [references/apple-runtime.md](references/apple-runtime.md) before executing the workflow.

## Required outcome

- Preserve the user's composition, camera position, perspective, geometry, lighting intent, protected text, logos, and blank areas.
- Reconstruct locally appropriate semantic detail: subject structure, material behavior, and style-consistent marks, not generic sharpening noise.
- Deliver the requested exact dimensions and verify representative crops at native pixels.
- Do not claim success from a whole-image thumbnail.

## Workflow

1. Run the hard Apple-Silicon and storage preflight. Do not proceed on failure.
2. Inspect the source at native scale. Classify its visual domain and intent: photograph, product render, portrait, food, vehicle, architecture/interior, landscape/wildlife, document/archive, illustration, painting, anime, or mixed media. Do not force photographic texture onto non-photographic work.
3. Choose the operating mode: creative semantic reconstruction, faithful restoration, or evidence-preserving enhancement. Evidence-preserving work must not use generative reconstruction.
4. Identify weak semantic regions by subject, material, and style. Distinguish genuinely missing information from intended smoothness, shallow depth of field, motion blur, atmospheric haze, painterly simplification, or graphic flat color.
5. Record a global scene anchor and invariants that must not change: identity, pose, anatomy, product geometry, composition, perspective, object count, lighting logic, depth of field, stylistic mark-making, protected text, logos, and intentional blank areas.
6. Create explicit masks for exact text, logos, interfaces, diagrams, and other protected regions. Restore those source pixels deterministically with `scripts/restore_protected.py`; do not ask the image model to reproduce exact content.
7. Choose tile dimensions from model input size, final viewing scale, and local semantic complexity. Use overlapping tiles with enough contextual margin to preserve continuity; do not select a grid only from the requested output resolution.
8. Audit the source for structural defects before prompting. Read [references/defect-repair.md](references/defect-repair.md) whenever lines, repeated modules, luminous geometry, foliage, distant structures, anatomy, or occlusions can fail.
9. Build prompts from [references/prompt-system.md](references/prompt-system.md), then load only the relevant domain modules from [references/scene-modules.md](references/scene-modules.md). Do not use quality adjectives as a substitute for named construction relationships.
10. Run a structure-first editing pass on every tile. Where no defect is present, preserve the inspected structure unchanged. Establish valid topology, geometry, perspective, repetition, occlusion, and emitter shapes before requesting material microdetail. Reject the structural pass if a line or object is merely sharper but still malformed. Record it with `scripts/record_tile.py --stage structure --review-note ...`.
11. Run a separate semantic-material pass only on structurally accepted tiles. Use the same global scene anchor and record it with `scripts/record_tile.py --stage semantic --review-note ...`; the script must reject a semantic tile without an accepted structural predecessor.
12. Register every edited tile against its source crop before feathering. Reject drift beyond the configured geometry and correlation thresholds; never hide misregistration with blur.
13. Inspect the semantic master at native pixels. Review the focal subject, highest-risk structure, repeated pattern, transition or occlusion, luminous region when present, far field, and every protected region. Create a hash-bound approval with `scripts/approve_semantic_master.py`.
14. Apply neural super-resolution only after approval, using the verified Apple MPS executor described in [references/apple-runtime.md](references/apple-runtime.md). Pass the approval file to `scripts/apple_mps_upscale.py`; it must reject an unapproved or changed master. The x4 neural reconstruction may be reduced to the exact target size after inference, matching the validated workflow.
15. Inspect the final image at 100% in the same high-risk regions. Repair failed tiles and rebuild rather than hiding defects with global blur, grain, texture overlays, or compression.

Use `scripts/prepare_tiles.py` to create an overlapping, resumable tile manifest and global scene anchor. Record accepted outputs with `scripts/record_tile.py`. Use `scripts/stitch_tiles.py` for mandatory registration, drift rejection, disk-backed assembly, alpha/ICC restoration, and a stitch report. Restore protected pixels with `scripts/restore_protected.py`. Approve the inspected semantic master with `scripts/approve_semantic_master.py`, run Real-ESRGAN on Apple MPS with `scripts/apple_mps_upscale.py --approval ...`, and export the final evidence bundle with `scripts/verify_output.py`.

## Tile density

- Start from the editing model's effective input dimensions and keep the focal subject large enough to resolve its internal structure.
- Increase local tile density where subject scale, identity, anatomy, text adjacency, patterns, or material complexity require it.
- Use fewer tiles for intentionally smooth or defocused regions and more for focal detail. A 4x4 grid is a common large-image starting point, not a universal minimum.

If a tile remains vague at native size, reduce its physical coverage and regenerate it. Upscaling the vague tile is not a solution.

## Acceptance checks

- The focal subject retains identity, silhouette, proportions, pose, object count, and scene role.
- Added detail follows the source domain: photographic material cues for photos, production detail for products, anatomical detail for living subjects, and style-consistent marks for artwork.
- Materials remain distinguishable through their own structure and response to light rather than one shared noise pattern.
- Intended smoothness, blur, bokeh, haze, negative space, graphic flats, and painterly simplification remain intentional instead of being indiscriminately textured.
- Highlights roll off naturally without clipped glowing worms; shadows retain texture without HDR halos.
- Repeated structures and transitions do not warp across tile seams. No doubled edges, broken contours, duplicate subjects, or discontinuous patterns and reflections.
- Every visible line has a physically justified origin, continuation, junction, occlusion, or termination. No melted, kinked, forked, floating, tapering, or arbitrarily interrupted edges.
- Parallel and repeated line families converge consistently under the locked projection. Repeated modules preserve count, cadence, thickness, alignment, and distance-dependent size progression rather than becoming approximate decorative marks.
- Luminous regions preserve the geometry of the emitter, housing, reflected light, and low-frequency bloom as separate phenomena. Bloom must not invent or erase structural edges.
- Organic structures preserve support topology: trunks support branches, branches support smaller growth, and leaf or fur groups attach plausibly instead of forming fused blobs or repeated stamps.
- Far-field detail stays scale- and atmosphere-consistent. It must not become pseudo-text, glyph-like windows, false people, or crisp foreground texture.
- Every overlap has one clear edge owner and plausible depth order. No doubled contours, melted tangencies, transparent solids, or impossible shared edges.
- Camera texture or artistic surface texture is restrained and subordinate to semantic detail. Grain, canvas, paper, halftone, or brush texture must never disguise empty content.
- Final dimensions, color mode, file size, and representative native-pixel crops are verified.
- The Apple preflight report, tile manifest, per-tile prompts and hashes, stitch registration report, protection masks, hash-bound semantic-master approval, Real-ESRGAN model identity, MPS report, and final checksum are retained with the deliverable.

## Stop conditions

Do not deliver while any focal crop still shows meaningless synthetic texture, malformed anatomy or objects, broken line topology, inconsistent repeated structures, invalid light emitters, fused organic forms, ambiguous edge ownership, tile seams, changed identity or geometry, invented text, style drift, or inconsistent perspective. Regenerate the affected tile with a narrower structure-first prompt, the correct domain module, and smaller coverage, then restitch and recheck. Never use Real-ESRGAN as a repair stage: it may reconstruct local pixels, but it cannot prove or restore scene topology.
