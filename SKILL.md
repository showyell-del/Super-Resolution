---
name: super-resolution
description: Reconstruct scene-appropriate semantic detail in AI-generated, compressed, scanned, or over-smoothed raster images before delivering 4K, 6K, 8K, 12K, or larger outputs on an Apple M-series Mac with PyTorch MPS and Real-ESRGAN. Use when a nominally high-resolution image still lacks meaningful subjects, materials, or style detail at native pixels; not for resize-only requests or non-Apple-Silicon computers.
---

# Super-Resolution

Treat pixel count, semantic detail, and photographic credibility as separate acceptance problems. Enlarging vague or malformed content is not completion.

## Hard gate

- Run only on macOS with an arm64 Apple M-series processor.
- Neural enlargement must use the verified Real-ESRGAN x4 model through PyTorch MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Do not substitute CUDA, CPU inference, Core ML conversion, interpolation-only enlargement, or another backend.
- Before allocating images or loading the model, run `apple_preflight.py` against source, output, scratch, temporary storage, runtime, weights, and Codex generated-image storage. Every distinct native volume must have strictly more than 50 GiB free.
- Stage the verified runtime onto a safe volume with `stage_runtime.py` when its original volume fails preflight. Never delete user files automatically.
- Read [apple-runtime.md](references/apple-runtime.md) before execution.

## Choose the truth mode

- **Creative semantic reconstruction:** synthesize plausible detail while preserving declared invariants. Use for generated imagery and creative work.
- **Faithful restoration:** add only detail supported by the source or same-subject references. Do not invent identity, anatomy, text, objects, or history.
- **Evidence-preserving enhancement:** no generative reconstruction. Use reversible enhancement and disclose every operation.

For people, distinguish real identity restoration, creative people reconstruction, and background crowd continuity. Never describe an invented face as recovered identity.

## Route by semantic density

Inspect the source at 100% and inventory every acceptance-critical subject. For people, create a subject manifest from `assets/subject-manifest.json` and read [people-camera-realism.md](references/people-camera-realism.md).

- Face width **120–200 px** in the semantic editor is preferred.
- At **80–119 px**, whole-frame editing is allowed only for secondary subjects with stable anatomy.
- At **48–79 px**, contextual group or individual region repair is mandatory.
- Below **48 px**, individual or small-group creative reconstruction is mandatory; identity restoration requires a sharper reference.

Do not proceed with a whole-frame pass when this gate requires local repair. If the user asks only for more resolution, explain that the limiting factor is subject pixel density and use the sparse-region route.

## Decision path

1. Normalize the source to the exact requested aspect ratio with `normalize_canvas.py`; never stretch. Stop if the permitted crop limit is exceeded.
2. Record the global scene anchor: composition, camera, perspective, geometry, count, identity/role, pose, lighting, depth of field, style, and protected content.
3. Mask exact text, logos, interfaces, diagrams, and blank areas. Restore them deterministically with `restore_protected.py`.
4. Audit topology and geometry using [defect-repair.md](references/defect-repair.md). Run structure-first repair before material detail.
5. Choose reconstruction scope:
   - adequate subject density and broad defects: overlapping whole-image tiles;
   - small complex subjects or isolated failures: [sparse region repair](references/sparse-region-repair.md);
   - evidence-preserving work: non-generative enhancement only.
6. Build prompts with [prompt-system.md](references/prompt-system.md) and only relevant [scene modules](references/scene-modules.md). Prompts must name construction relationships and invariants, not just quality adjectives.
7. Register every edited tile or region against its source. Reject perspective, translation, scale, count, identity, or topology drift; do not hide it with blur.
8. Inspect the semantic master at native pixels. When people matter, run `build_contact_sheet.py` and manually verify anatomy, identity distinction, count, lighting, and integration. A failing report blocks approval.
9. Create a hash-bound approval with `approve_semantic_master.py`; pass `--subject-report` when `subject_report_required` is true.
10. Run `apple_mps_upscale.py` with the approval and staged runtime. Real-ESRGAN may be reduced after x4 inference to exact target dimensions.
11. Inspect the final at 100%. If only sparse regions fail, repair them without resizing, register and composite them, then regenerate all affected reports and checksums.
12. Run `finalize_delivery.py` after the final modification. Read [delivery-evidence.md](references/delivery-evidence.md).

## Acceptance

- Composition, camera, perspective, object/person count, roles, poses, protected content, and intentional blank or defocused regions are unchanged.
- Lines have valid origins, continuations, junctions, occlusions, and terminations. Repeated modules share a consistent projection, cadence, thickness, and depth progression.
- Emitters, housings, reflected light, and bloom remain separate. Organic forms preserve support topology. Far-field detail remains scale- and atmosphere-consistent.
- Materials differ through construction and light response, not a shared noise overlay. Grain, pores, fabric, foliage, and microtexture are restrained and scale-correct.
- Acceptance-critical people are individually distinct, anatomically coherent, integrated with the same camera and lighting, and free of wax skin, blank eyes, repeated facial templates, fused hands, halos, or cutout edges.
- The final exact bytes have a complete evidence chain and representative native-pixel crops.

## Stop conditions

Do not deliver while any focal crop has vague or cloned subjects, invented identity or text, malformed anatomy, broken topology, inconsistent repetition, invalid emitters, fused organic forms, ambiguous edge ownership, seams, registration drift, style drift, or merely sharpened synthetic texture. Repair the smallest sufficient region, rebuild downstream approvals, and verify again. Real-ESRGAN is an enlargement stage, not proof of semantic correctness.
