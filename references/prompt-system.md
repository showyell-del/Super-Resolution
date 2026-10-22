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

For a crop trial, choose the highest-risk region that contains enough context to judge integration. Compare the trial against the source/reference at its intended final scale. Reject invented geometry, style drift, texture pasted independently of form, uniform sharpening, or lost optical depth. Change only the failed layer using a concrete method or better evidence, following the user's iteration budget. Do not repeat an unchanged prompt or add more “realistic”, “8K”, or “ultra-detailed” language; use the main skill's stopping conditions.

## Explicit balanced creative target

When selected by the user, add this compact look instruction after immutable/geometry constraints:

> Natural, coherent creative enhancement. Keep recognizable features, expression and focus. Resolve useful skin, grouped hair and garment texture according to form and light; accept restrained stylization. Prioritize the complete image over microscopic realism. Avoid a floating fiber veil, repetitive pores and sharpening halos. Added detail is creative, not authenticated restoration.

Do not ask every coat to reveal individual fibers or every face to reveal pores; that instruction can overdrive surface texture. This target never permits changed expression, malformed anatomy, new wording, seams or loss of focus continuity. Use the quality target in the review, and keep any earlier strict-photographic verdict separate.
