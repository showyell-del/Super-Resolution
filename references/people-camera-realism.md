# Camera-real people reconstruction

Use this module whenever a face or person must withstand native-pixel inspection. Resolution alone cannot recover identity or anatomy that occupies too few source pixels.

## Semantic density gate

Measure face width in the image entering the semantic editor.

- **120–200 px:** preferred working range for a face that must look individually photographed.
- **80–119 px:** whole-frame editing is allowed only when the person is secondary and the native crop already contains stable anatomy. Inspect every face separately.
- **48–79 px:** use a contextual group or individual region. Whole-frame editing is prohibited for acceptance-critical faces.
- **Below 48 px:** individual or small-group reconstruction is mandatory. Treat the output as creative reconstruction, not restoration of the real person's identity.

If a face cannot be brought into the preferred range without losing pose and scene context, use multiple passes: contextual group repair first, then a narrower individual face-and-body repair.

## Truth modes

- **Identity restoration:** allowed only when a sharper reference of the same person exists. Preserve the reference identity; do not infer it from a tiny face.
- **Creative people reconstruction:** create distinct, plausible people while preserving count, role, pose, wardrobe, age range, expression, gaze, lighting, and placement. Do not claim recovered identity.
- **Crowd/background continuity:** preserve head count, silhouettes, depth, and action. Do not force portrait detail into people that should remain optically unresolved.

Record every acceptance-critical person in `assets/subject-manifest.json`. Give each a stable ID and visible distinguishing traits. Prompts must reference those IDs rather than saying only “different faces”.

## Prompt construction

State the capture conditions and physical relationships, not quality slogans:

1. Preserve the exact camera position, focal length impression, stage geometry, person count, blocking, poses, wardrobe, and warm key/fill/rim-light directions.
2. Reconstruct each listed subject as a distinct individual with different facial proportions, eye spacing, brow shape, nose bridge, lip shape, jawline, hairline, skin tone variation, age cues, and expression.
3. Resolve two anatomically complete eyes with aligned gaze, natural eyelids and catchlights; a coherent nose and philtrum; closed lip contours; ears, jaw, hairline, neck, hands, and fingers appropriate to distance and pose.
4. Preserve natural skin translucency, fine tonal variation, restrained pores, fabric weave, seams, folds, and believable hair groups. Keep detail subordinate to focus and depth of field.
5. Preserve photographic imperfections: modest sensor noise, lens softness away from focus, highlight roll-off, chromatic restraint, and low-frequency atmospheric depth.

Negative constraints: no cloned faces, beauty-filter skin, wax figures, doll eyes, painted eyelashes, duplicated smiles, blank eyes, fused fingers, extra limbs, melted instruments, repeated hair templates, excessive pores, cutout edges, HDR halos, or synthetic microcontrast.

## Native-pixel acceptance

Run `build_contact_sheet.py` on the exact semantic master. The automated report is necessary but not sufficient. At 100% inspect:

- count and ID correspondence;
- two valid eyes, aligned gaze, eyelids, teeth/lips, ears, jaw, hairline, neck, hands, and limb anatomy;
- distinct facial geometry across people, not only different hair or clothing;
- lighting direction, skin response, depth of field, and edge integration;
- absence of doubled contours, pasted faces, halos, or inconsistent scale.

Reject the master when any acceptance-critical subject fails. Do not rely on Real-ESRGAN to repair semantic anatomy.
