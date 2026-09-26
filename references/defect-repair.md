# Structural repair

Use this only when the source has invalid geometry or topology. Do not force an extra pass for a structurally valid image.

Repair the smallest region that contains the defect and enough context. Preserve camera, perspective, identity, count, silhouette, depth order, light, focus, crop, and protected content.

Check:

- contours have an owner, origin, continuation, junction, occlusion, and termination;
- shared world directions keep consistent vanishing behavior;
- repeated systems preserve count, phase, interval, thickness, alignment, and depth progression;
- emitters remain distinct from housing, reflection, spill, and bloom;
- bodies, limbs, branches, stems, and leaves keep valid support hierarchy;
- reflections and transparency respect surface orientation and occlusion;
- far-field information simplifies with distance rather than becoming pseudo-text or foreground-scale detail.

Prompt:

> Conservatively repair only [named defect]. Preserve all locked invariants. Reconstruct valid topology from supported endpoints and neighboring context. Keep line ownership, projection, repetition, attachment, occlusion, and depth coherent. Do not add microtexture or redesign content. No melting, duplication, arbitrary forks, floating fragments, impossible tangencies, or sharpened invalid geometry.

After structure passes, material detail may be added in the same region. A failed retry stops that region; it does not restart the image.
