# Layered semantic prompt

Build one compact prompt per image or crop. Higher layers override lower layers. Resolve contradictions before generation; do not pile on quality adjectives or a long universal negative list.

1. **Immutable:** exact composition, content, protected text/logo, identities or roles, count, crop, color intent, and reference roles. Say which image supplies each element.
2. **Geometry:** viewpoint, vanishing behavior, silhouette, support, junction, occlusion, repetition, and object placement. Repair only named failures.
3. **Subject:** anatomy, expression, garment or object construction, attachment, and physically possible interaction.
4. **Material:** surface behavior tied to form, joins, folds, roughness, reflectance, local wear, and lighting. Resolve missing mid-scale construction before microtexture.
5. **Look:** select one visual route from `visual-routing.md`. For photo targets, specify lens/focus/motion/exposure/grain only if supported by the scene; for art, specify medium-consistent mark-making. Preserve intentional softness and negative space.
6. **Exclusions:** only defects likely in that crop; never forbid a feature that is intentional in the target look.

Prompt pattern:

> Edit this exact [image/crop] for [target look and truth mode]. Keep [immutable content]. Repair [specific structural failure] while preserving [camera/geometry/depth]. Resolve [subject and material construction] where projected size supports it. Match [observed target optics or mark-making] and retain [intentional softness]. Exclude [three or fewer relevant defects]. Return the same crop and aspect ratio.

For a crop trial, choose the highest-risk region that contains enough context to judge integration. Compare the trial against the source/reference at its intended final scale. Reject it for invented geometry, style drift, texture pasted independently of form, uniform sharpening, or lost optical depth. Revise the specific failed layer once; do not respond by adding more “realistic”, “8K”, or “ultra-detailed” language.
