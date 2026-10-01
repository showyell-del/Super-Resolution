# Visual routing before any prompt

Infer the look from visible evidence, then reconcile it with the user's explicit request. Record a short source-to-target visual contract in the native review before generation; review it again before neural enlargement. When references disagree, the user's stated target and the designated content reference take priority for their respective roles.

| Source / target look | Preserve and resolve | Typical false detail to reject |
| --- | --- | --- |
| Camera photograph | Lens perspective, depth of field, motion, sensor/grain character, exposure roll-off, local asymmetry, believable materials and contact | Wax skin, plastic foliage, uniform microcontrast, repeated windows, edge halos, identical grain everywhere |
| Architectural or product render | Designed geometry, material response, intentional clean surfaces and lighting; add physical wear only if requested or evidenced | Random grime, invented joints, inconsistent reflections, photographic noise painted over clean geometry |
| Illustration or painting | Stroke/line logic, medium, pigment, contour rhythm, deliberate simplification and layering | Photographic pores, synthetic sharpening, inconsistent brush scale, textures detached from painted forms |
| Graphic, text, or UI | Exact glyphs, logos, layout, flat areas, antialiasing, and color boundaries | Generated lettering, textured text, deformed icons, halos |
| Mixed scene | Route each region by its own medium and preserve its boundary; e.g. photographed stage plus an exact graphic screen | Applying one style to the whole image, screen content spilling onto people |

For a requested style conversion, name the target evidence and which content remains locked. If no conversion was requested, target equals source. A render can be made more photographic only when the user asks; then build plausible camera response and material variation from visible structure instead of adding undirected grain.

Choose one short look clause for the prompt, then add only the relevant scene cues:

- **Photo:** “Keep the observed viewpoint, exposure, focus plane, motion, and natural falloff. Resolve real material and mid-scale construction with irregular but coherent local variation.”
- **Render:** “Keep the designed geometry and clean material language. Resolve joints, surface response, and reflections consistently with the existing lights.”
- **Illustration:** “Continue the original stroke weight, pigment behavior, shape simplification, and edge rhythm; resolve ambiguous forms in the same medium.”
- **Graphic:** “Preserve exact text, marks, alignments, flat color, and crisp antialiasing; restore these deterministically from reference assets.”
- **Mixed:** Assign one of these clauses per region. Protect the boundaries between media and the occluding foreground.

## Native-pixel diagnosis

Decide for each important region whether detail is: (1) valid and already visible, (2) missing from compression or generation, (3) deliberately absent due to style or optics, or (4) unknowable without a reference. Only (2) is a reconstruction target. Faithful restoration may use (1) and same-subject references; creative reconstruction may plausibly synthesize (4) but must label it as invented. Evidence-preserving work cannot synthesize it.

Inspect repeated structures, silhouette intersections, material boundaries, human anatomy, focal texture, shadow/reflection ownership, and distance falloff. Rank by visibility at delivery size. For photo targets, judge camera realism across the whole spatial frequency range: large forms, mid-scale construction, and small detail must agree. Extra high-frequency texture cannot rescue wrong forms or smeared mid-scale detail.
