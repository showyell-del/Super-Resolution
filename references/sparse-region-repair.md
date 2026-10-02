# Registered repair pack

Use local regions for small subjects or isolated semantic failures.

1. List every required region before the selected neural run. Keep `target_box` tight around the defect; added crop context is for pose, light, and registration and must remain unchanged in the composite. Compare the manifest's `planning` count and gross coverage with one whole-image edit; merge redundant overlapping crops when density and registration remain valid.
2. Add subject_ids to every people region.
3. Extract with prepare_regions.py.
4. Generate each region at high density, inspect it at native pixels, and record it with record_region.py.
5. Bind the accepted manifest during semantic approval.
6. Run the selected neural route once.
7. Composite the full repair pack once onto the enlarged base with composite_regions.py --base.
8. Run one final native-pixel review and final checksum.

Do not shrink repair outputs into the semantic master. Do not replace a whole image for a local failure. Do not hide registration drift with blur, grain, glow, or compression.

## High-density semantic canvas

When detail is missing throughout a creatively reconstructable scene, generate registered tiles around one approved composition anchor. Choose tile framing that preserves aspect ratio and enough overlap to compare geometry, material construction and focus; do not reduce overlap to one pixel merely to fit a generator's aspect ratio. Place seams away from eyes, mouths, necklines and dominant structural edges where practical.

Record accepted native outputs with `record_tile.py`; an already valid structure can be recorded without an unnecessary structural generation. Stitch with `stitch_tiles.py manifest.json semantic-master.png --output-scale SCALE`. Choose SCALE from actual generated tile density, not the desired final resolution. The command preserves generated detail on that larger canvas and rejects enlargement beyond the smallest native tile support (allowing one pixel of boundary rounding). Its resized source is a geometry reference, not new detail evidence.

Inspect every tile join at native pixels even when registration correlation passes. Check texture construction and focus on both sides, not just edge alignment. Correct documented join defects with registered narrow targets before neural enlargement. Reserve separate high-density face/material packs when the stitched master cannot support the final viewing scale; never downsample those packs into the master. If final review finds a new local material failure, replace that pack and recompose onto the existing neural base, then review the new final bytes.
