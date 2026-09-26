# Compact semantic prompt

Use one contract per whole image or region:

1. **Lock:** composition, camera, perspective, count, identity or role, pose, lighting, focus, palette, crop, protected content.
2. **Repair:** name the failed structures and their physical relationships.
3. **Resolve:** name material or subject detail tied to scale, form, light, and depth.
4. **Forbid:** list only failure modes relevant to this crop.

Template:

> Edit this exact crop, not a redesign. Preserve [locked invariants]. Repair only [specific malformed topology, anatomy, material, or environment]. Resolve [construction detail] so it follows [surface form, support, perspective, light, wear, focus, and depth]. Preserve intentional blur, haze, reflections, and negative space. Output the same crop and aspect ratio. No [relevant semantic defects], geometry drift, identity drift, pseudo-text, repeated texture stamps, halos, or global sharpening.

Concrete relationships beat quality adjectives:

- fabric: weave follows folds; seams join panels; highlights follow fiber and tension;
- skin: tonal variation follows facial planes; pores remain restrained; no global blur;
- architecture: line families share vanishing behavior; repeated bays keep count and cadence;
- foliage: trunk-to-branch-to-leaf support remains visible; no fused blobs;
- metal/glass: grain, roughness, reflections, and edge highlights follow surface orientation;
- distance: detail and contrast fall with projected size, haze, focus, motion, and illumination.

Do not use “8K,” “12K,” “masterpiece,” or “ultra-detailed” as a substitute for construction. On failure, correct the specific relationship and retry only that region once.
