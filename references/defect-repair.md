# Structure-first defect repair

Read this reference before semantic reconstruction when close inspection reveals malformed lines, repetitions, lights, anatomy, organic forms, distant objects, or occlusions. The goal is not to make defects sharper. It is to restore physically or stylistically coherent relationships before adding surface detail.

## Mandatory two-pass order

### Pass A: structural repair

Repair only topology and construction. Preserve the global scene anchor, crop, camera, object count, lighting intent, color relationships, focus, and protected content. Do not add pores, grain, scratches, foliage texture, facade noise, material aging, or cinematic effects in this pass.

Use this core prompt and append only relevant modules below:

> Perform a conservative structural repair of this exact crop. Do not redesign it and do not add microtexture yet. Preserve the locked composition, camera or projection, object count, identity, silhouette, depth order, lighting intent, focus behavior, and protected regions. Reconstruct only malformed topology and geometry from physically supported endpoints and neighboring context. Every line, contour, junction, opening, repeated unit, support, and occlusion must have a coherent origin, continuation, thickness, depth owner, and termination. Keep source-supported imperfections; remove only synthetic warping, melting, duplication, arbitrary forks, discontinuities, and impossible tangencies. Output the same crop and aspect ratio.

Pass A fails if a defect becomes cleaner or sharper without becoming structurally valid.

### Pass B: semantic and material reconstruction

Run only after Pass A passes at native pixels. Add scene-appropriate construction and material cues that follow the repaired geometry. Use the normal universal scaffold and relevant scene modules. Detail must attach to a valid surface, body part, mark, or object; it must not float across boundaries or define new geometry.

### Pass C: neural super-resolution

Run only after the semantic master has a hash-bound approval. Real-ESRGAN has no prompt and is not a geometry repair model. It must not receive a master with known structural defects.

## Structural prompt modules

Select modules present in the crop. State the measurable relationship to preserve; avoid “perfect lines,” “ultra-detailed,” “8K,” “12K,” and “photorealistic” as repair instructions.

### Line topology and edge ownership

> Trace each major contour from a source-supported start to a source-supported end. Preserve continuous thickness and curvature appropriate to the object. At every crossing or overlap, assign one foreground edge owner and one occluded background continuation. Remove doubled contours, melted tangencies, floating fragments, arbitrary forks, pinched widths, sudden kinks, and edges that terminate without a joint, boundary, shadow, or occluder.

### Perspective and line families

> Preserve the locked camera and projection. Group edges that represent the same world direction; each family must converge consistently toward its shared vanishing region. Preserve vertical behavior, horizon relationship, foreshortening, and depth-dependent spacing. Do not straighten intentional lens curvature, but do not invent local bends that contradict the surrounding line family.

### Repeated structures

> Treat the repeated elements as one measured system rather than individually invented marks. Preserve the source-supported count, phase, interval, width, alignment, row and column relationships, perspective convergence, and size progression with depth. Keep plausible small variation in contents or reflections while the supporting grid remains coherent. Add no extra bays, slats, windows, stitches, tiles, spokes, teeth, buttons, or limbs.

### Luminous geometry

> Separate the physical emitter, its housing or mounting surface, direct reflection, indirect spill, and optical bloom. The emitter keeps a stable source-supported shape and attachment; the housing remains visible where exposure allows. Bloom is smooth, low-frequency, and subordinate to the emitter. It may soften contrast but must not become a white worm, erase a structural edge, bridge separate lights, or define new geometry.

### Organic support topology

> Reconstruct hierarchical support before texture: body to limb to joint, trunk to bough to twig to leaf cluster, stem to branch to petal, or root to strand to tip. Every subordinate form attaches plausibly to a larger support and obeys gravity, growth direction, pose, and occlusion. Preserve negative gaps through the form. Avoid fused black masses, broccoli clusters, repeated stamps, detached leaves, impossible joints, and unsupported detail.

### Far-field and atmospheric detail

> Match information density to projected pixel size, distance, haze, focus, and illumination. Preserve large silhouettes, floor or object cadence, and restrained light groupings; do not manufacture pseudo-text, glyph-like windows, tiny faces, false people, crisp interiors, or foreground-scale texture. Distant detail should become simpler and lower-contrast while remaining structurally consistent.

### Reflections and transparent surfaces

> Keep the physical boundary of the reflective or transparent surface distinct from reflected content. Reflections must follow surface orientation, curvature, roughness, and plausible scene sources; transmission must preserve depth order. Do not treat reflections as solid objects, duplicate bright forms, or continue reflected lines through frames and occluders.

## Tile defect audit

Before accepting Pass A, inspect the tile at 100% and compare it with adjacent context:

1. Follow every dominant line and contour end to end. Reject unexplained forks, gaps, kinks, width changes, and double edges.
2. Compare each parallel family and repeated system. Reject inconsistent convergence, count, cadence, alignment, or depth progression.
3. Separate bright emitters from bloom and reflection. Reject lights whose glow replaces their physical construction.
4. Inspect organic forms as support hierarchies. Reject detached, fused, duplicated, or unsupported parts.
5. Inspect foreground/background junctions. Reject ambiguous edge ownership and impossible transparency.
6. Inspect the far field at native pixels. Reject pseudo-symbols and foreground-scale invented texture.
7. Compare overlap margins with neighboring tiles. Reject phase changes and topology that cannot continue across the seam.

If any item fails, reduce tile coverage so the failed structure occupies more pixels, provide more contextual overlap, and regenerate Pass A with only the relevant module. Do not compensate by adding more adjectives or by asking the upscaler to repair it.

## Approval schema

Create a review JSON for `scripts/approve_semantic_master.py` after native-pixel inspection:

```json
{
  "review_scale": "100% native pixels",
  "global_checks": {
    "geometry_and_perspective": "pass",
    "line_topology_and_edge_ownership": "pass",
    "repeated_structures": "pass",
    "emitters_reflections_and_bloom": "not_applicable",
    "organic_support_topology": "not_applicable",
    "far_field_and_depth_falloff": "pass",
    "protected_content": "pass"
  },
  "regions": [
    {
      "name": "highest-risk structure",
      "box": [100, 200, 800, 600],
      "status": "pass",
      "note": "Lines traced end to end; cadence and depth order checked."
    }
  ],
  "reviewer_note": "No known structural defect remains before neural super-resolution."
}
```

Use `not_applicable` only when the category is genuinely absent from the image. An approval is invalid if every structural category is marked `not_applicable`.
