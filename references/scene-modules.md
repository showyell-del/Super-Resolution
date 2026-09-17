# Scene and subject modules

Select only the modules present in the current tile. Combine subject and material modules when needed.

## Operating modes

- **Creative semantic reconstruction:** May synthesize plausible subordinate detail while preserving the global scene anchor and explicit invariants. Use for AI-generated imagery, concept art, renders, and creative photographs when the user wants richer detail.
- **Faithful restoration:** Enhance only structures supported by the source. Repair compression, fading, noise, scratches, or soft capture without inventing identity, text, anatomy, objects, or historical features.
- **Evidence-preserving enhancement:** For documents, diagrams, screenshots, scientific, medical, legal, journalistic, forensic, or archival evidence. Do not use generative reconstruction. Use only reversible, non-generative enhancement and disclose every operation. If the request requires invented missing information, stop and ask for a creative copy separate from the evidence master.

## People and portraits

Lock identity, age, expression, gaze, hairstyle, anatomy, pose, wardrobe, makeup, and lighting. Reconstruct facial planes, eyelid and lip edges, individual hair groups, believable skin variation, fine vellus hair, garment construction, seams, and accessories only where visible. Preserve retouching style and depth of field. Avoid pore tiling, wax skin, sharpened makeup, changed teeth, iris patterns, hands, or facial proportions.

## Animals and wildlife

Lock species, individual markings, anatomy, pose, gaze, limb count, and environment. Resolve fur or feather groups from their growth direction and body form, whiskers, claws, scales, eye moisture, and habitat interaction. Avoid repeated fur stamps, decorative feathers, human-like eyes, changed markings, or anatomically impossible joints.

## Products and packaging

Lock silhouette, proportions, colorway, controls, ports, seams, labels, logo zones, and packaging dielines. Resolve manufacturing detail such as molding seams, machined edges, fasteners, glass thickness, material junctions, controlled reflections, print registration, and restrained handling marks. Preserve premium clean surfaces when intended. Restore exact labels separately rather than generating pseudo-text.

## Vehicles and machinery

Lock model-defining geometry, stance, wheel diameter, track, panel gaps, glass, lights, trim, controls, and mechanical relationships. Resolve tire tread, brake components, fasteners, seals, welds, hoses, vents, machined parts, paint behavior, and scale-correct wear. Avoid altered wheel spokes, impossible reflections, fake vents, random bolts, or excessive grime.

## Food and beverages

Lock portion, plating, container, garnish placement, and doneness. Resolve ingredient structure, crumb, fibers, bubbles, condensation, sauce viscosity, oil sheen, steam, translucency, browning, and moisture according to the food. Avoid plastic surfaces, duplicated garnish, artificial saturation, excessive gloss, added ingredients, or crispy texture on soft food.

## Fashion, fabric, and accessories

Lock garment cut, drape, pattern alignment, color, closures, branding zones, and body relationship. Resolve weave or knit scale, yarn direction, seams, stitching, hems, folds, tension, embroidery, leather grain, hardware, and restrained wear at plausible contact points. Avoid moire, invented pattern repeats, changed tailoring, random wrinkles, or texture floating across folds.

## Architecture and interiors

Lock vanishing points, facade and room geometry, floor heights, openings, window grids, furniture placement, roads, railings, and lighting. Before material enrichment, apply the line-family, repeated-structure, luminous-geometry, far-field, reflection, and edge-ownership modules from [defect-repair.md](defect-repair.md). Resolve scale-consistent fixtures, furniture, hardware, joints, mullions, panel seams, gaskets, glass thickness, ceiling systems, restrained occupants, and reflections only within existing regions. Window contents may vary, but their frames and floor cadence must remain one perspective-consistent system. Keep the emitter, fixture, wall reflection, glass reflection, and bloom separate. Avoid bent grids, tapered mullions without perspective cause, broken railings, altered floor count, duplicate furniture, random rooms, invented signage, glyph-like distant windows, decorative structural changes, or bloom that replaces construction.

## Landscape, plants, and terrain

Lock horizon, terrain, waterline, vegetation placement, weather, season, and atmospheric depth. Resolve species-consistent plant structure, branch-supported leaf clusters, bark, groundcover, soil, rock strata, erosion, water behavior, and distance-dependent detail. Avoid broccoli foliage, repeated leaf stamps, identical rocks, sharpened haze, new mountains, or changed weather.

## Urban scenes and crowds

Lock street layout, vehicle and pedestrian placement, storefront zones, traffic logic, and depth. Resolve plausible street furniture, facade construction, clothing groups, road markings, curb detail, restrained window contents, and layered atmospheric perspective. Keep distant people indistinct; avoid cloned faces, pseudo-signage, extra vehicles, and crowd multiplication.

## Documents, archives, and old photographs

Lock all semantic content, handwriting, typography, borders, faces, dates, stamps, and historical features. Restore legibility, tonal separation, paper fibers, emulsion grain, crease continuity, and damage boundaries without inventing missing words or facial features. Treat colorization, damage removal, and content reconstruction as separate user choices. Never replace documentary evidence with plausible fabrication.

## Interfaces, diagrams, maps, and technical imagery

Treat text, icons, measurements, coordinates, chart marks, wiring, topology, and geometry as exact protected content. Use evidence-preserving enhancement unless the user explicitly requests a redesigned creative copy. Do not generatively reconstruct unreadable labels, numbers, medical findings, scientific observations, or engineering details.

## Illustration, anime, comics, and vector-like art

Lock line language, character design, proportions, palette, cel shading, brush shape, edge hierarchy, screentone, print texture, and intentional flats. Resolve clean line continuity, controlled corners, pattern rhythm, style-consistent hatching, brush or fill boundaries, and production artifacts appropriate to the medium. Do not add skin pores, camera grain, photographic reflections, 3D materials, extra linework, or realism that changes the style.

## Painting and traditional media

Lock composition, palette, subject shapes, brush language, paint thickness, canvas or paper behavior, and degree of abstraction. Resolve strokes following form, pigment buildup, glazing, dry-brush breakup, edge variety, underdrawing, paper tooth, or canvas weave only where the original medium supports it. Avoid photographic detail, uniform canvas overlays, invented cracks, over-restoration, or smoothing away the artist's marks.

## 3D renders and game art

First determine whether the user wants a cleaner render or conversion toward live-action realism. For render preservation, lock asset design, topology silhouette, UV patterns, lighting, material intent, and art direction; improve physically coherent surface response without introducing real-world damage. For realism conversion, state that goal explicitly and add construction, material, optical, and environmental imperfections selectively. Do not mix both modes implicitly.

## Material modules

- **Skin:** pores vary with facial region and stretch; fine lines follow expression and age; highlights follow moisture and oil distribution.
- **Hair/fur:** groups emerge from roots and flow; density, occlusion, flyaways, and highlight breakup follow form.
- **Wood:** grain follows cut and construction; end grain, pores, joints, finish, and wear differ by part.
- **Metal:** machining or brushed direction, edge wear, fasteners, oxidation, fingerprints, and highlights match alloy and finish.
- **Glass/liquid:** thickness, refraction, meniscus, bubbles, condensation, reflection layering, and transparency remain physically related.
- **Stone/concrete/ceramic:** aggregate, pores, mineral variation, joints, glaze, chips, dampness, and dirt accumulation remain scale-correct.
- **Plastic/rubber:** molding seams, parting lines, matte or gloss response, compression, scuffs, and dust match manufacturing and use.
- **Paper/print:** fibers, embossing, ink spread, halftone, registration, folds, and edge wear must not corrupt exact content.
- **Water/wet surfaces:** ripples, puddle boundaries, absorption, droplets, flow direction, and broken reflections follow gravity and surface shape.
- **Snow/sand/smoke/clouds:** preserve particulate or volumetric behavior, wind direction, occlusion, translucency, and distance falloff; do not sharpen them into solid objects.
