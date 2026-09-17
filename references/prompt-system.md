# Universal semantic reconstruction prompt system

Use this scaffold for every domain. Add only the relevant modules from [scene-modules.md](scene-modules.md). When the input already contains malformed geometry, repetition, emitters, organic topology, or occlusion, run the separate structure-first pass in [defect-repair.md](defect-repair.md) before using this semantic-detail scaffold. Do not paste an exhaustive prompt library into every tile.

## Global scene anchor

Before writing tile prompts, complete `scene_anchor.json`. Every tile must repeat or inherit the same identity, object count, composition, lens or projection, lighting direction, palette, focus and motion behavior, atmospheric state, style language, protected content, and forbidden changes. A tile prompt may add local detail but must not override the anchor.

## Prompt assembly

Build each tile prompt in this order:

1. **Task and fidelity contract** — what is being improved and what cannot change.
2. **Tile inventory** — the visible subjects, materials, depth layers, and intentional effects in this crop.
3. **Missing semantic detail** — concrete structures, objects, material cues, or style marks that should become legible.
4. **Continuity** — perspective, lighting, focus, motion, scale, occlusion, pattern, and overlap behavior.
5. **Rendering language** — photographic realism or the source's existing illustration, painting, print, or graphic style.
6. **Negative constraints** — likely failure modes for this tile, not a generic wall of adjectives.

Record the final tile prompt with `scripts/record_tile.py`; the manifest is the resume checkpoint and audit trail.

## Universal scaffold

Adapt the bracketed fields; omit irrelevant clauses.

> Edit this exact image crop as a faithful semantic reconstruction, not a redesign. Preserve [identity / pose / anatomy / product geometry / composition / perspective / object count / lighting / depth of field / motion / stylistic language / protected blank and text areas]. The crop contains [subjects, materials, depth layers, and intentional effects]. Reconstruct only the genuinely missing information: [specific subject structures, material behavior, small objects, surface construction, or style-consistent marks]. Keep scale, occlusion, edge continuity, light direction, shadow logic, focus falloff, motion behavior, color relationships, and neighboring-tile continuity consistent with the source. Render in [photographic / product / archival / painterly / illustrated / anime / graphic] language already established by the image. Do not add, remove, duplicate, relabel, restyle, beautify, age, damage, or rearrange content unless explicitly requested. No invented text, logos, symbols, anatomy, objects, reflections, highlights, or background structures. No geometry drift, identity drift, style drift, repeated texture stamps, procedural noise, halos, oversharpening, or detail outside the source's depth-of-field and motion limits. Output the exact same crop and aspect ratio.

## How to describe detail

Use nouns, construction, and physical or stylistic relationships:

- Prefer “individual woven threads following the garment folds, seam stitching, restrained pilling at friction points” over “ultra-detailed fabric.”
- Prefer “fine eyelashes emerging from the eyelid margin, natural pores following facial planes, subtle peach fuzz against rim light” over “8K skin.”
- Prefer “machined concentric marks, brushed grain aligned with the panel, fastener recesses, faint fingerprints near touch points” over “realistic metal texture.”
- Prefer “short directional brush strokes following the painted form, pigment buildup at overlaps, visible canvas only in thin paint” over “high-detail painting.”

Tie every detail to scale, structure, lighting, wear, anatomy, or mark-making. If a detail cannot be located or justified, do not request it.

“Ultra-detailed,” “12K,” “masterpiece,” “photorealistic,” and similar quality labels are not structural instructions. They can increase contrast and synthetic surface activity while preserving or amplifying malformed objects. Name the construction relationship, support, interval, edge owner, emitter, material response, or depth behavior that must be repaired or reconstructed.

## Preserve intentional softness

Semantic reconstruction must respect information that is intentionally absent.

- Do not sharpen bokeh, motion-blurred objects, atmospheric haze, reflections, translucent surfaces, or out-of-focus depth planes into crisp objects.
- Do not add pores to deliberately airbrushed beauty work, paper grain to clean vector graphics, or photographic imperfections to polished 3D or product rendering unless the user requests realism conversion.
- Do not fill intentional negative space, graphic flat colors, minimalist backgrounds, or painterly simplification with arbitrary texture.

## Continuity rules

- A feature crossing a tile edge must preserve position, angle, thickness, phase, color, light response, and depth ordering.
- Repeated elements must keep their original count and rhythm without cloning identical microtextures.
- Surface detail must follow form: skin planes, fabric folds, wood grain direction, metal machining, hair flow, perspective, brush direction, or print screen angle.
- Detail density should decrease naturally with distance, defocus, haze, motion, transparency, and low illumination.
- Reflections must correspond to plausible scene content and surface curvature; they are not decorative highlights.

## Negative constraints by risk

Select only relevant risks:

- **Identity/anatomy:** no changed face, age, ethnicity, expression, gaze, hand pose, limb count, body shape, fur pattern, or species.
- **Product/vehicle:** no changed silhouette, dimensions, panel gaps, controls, ports, labels, wheel geometry, trim, packaging, or colorway.
- **Scene/architecture:** no altered layout, horizon, vanishing points, openings, terrain, vegetation placement, furniture count, or structural geometry.
- **Text/graphics:** no invented letters, pseudo-text, altered logo, watermark, interface element, icon, chart, or label.
- **Artwork/style:** no conversion to photography, 3D rendering, oil paint, anime, watercolor, vector, or another artist language unless explicitly requested.
- **Texture:** no universal grain, random speckles, fake cracks, excessive grime, repeated stamps, crunchy edges, clarity halos, or sharpened noise.

## Repair loop

When a tile fails, diagnose the failure rather than adding more quality adjectives.

- Wrong identity or geometry: shrink tile coverage, strengthen invariants, remove creative terms, and supply more surrounding context.
- Broken lines, repetitions, emitters, organic support, or occlusion: return to Pass A in [defect-repair.md](defect-repair.md). Do not combine the repair with texture enrichment.
- Empty detail: name missing parts and their relationships; do not merely increase resolution language.
- Hallucinated clutter: restrict additions to existing regions and original object count.
- Repeated texture: request structural variation tied to support geometry and forbid stamps.
- Excessive aging or damage: ask for clean material construction plus restrained use cues; remove weathered, gritty, and cinematic-realism language.
- Over-sharpened background: explicitly preserve depth-of-field, haze, motion, and distance-dependent detail falloff.
- Tile seam: regenerate the less faithful tile with larger overlap and explicit edge continuity; do not blur the final image globally.

## Native-pixel review

Export review crops without resizing. Choose crops based on the image, not a fixed architectural checklist:

1. The primary identity or focal subject.
2. The most failure-prone material, anatomy, or style-mark region.
3. A transition: overlap seam, reflection boundary, depth change, patterned surface, hair edge, translucent edge, or motion boundary.
4. Any protected text, logo, interface, diagram, or exact geometry.

Reject any version that gains surface activity by changing meaning, identity, construction, or visual language.
