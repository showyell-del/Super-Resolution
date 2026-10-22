# Registered repair pack

Use local regions for small subjects or isolated semantic failures.

1. List every required region before the selected neural run. Keep `target_box` tight around the defect; added crop context is for pose, light, and registration and must remain unchanged in the composite. Compare the manifest's `planning` count and gross coverage with one whole-image edit; merge redundant overlapping crops when density and registration remain valid.
2. Add subject_ids to every people region.
3. Extract with prepare_regions.py.
4. Generate each region at sufficient native density for its final projected crop, including its entire context crop. Use the [projected-density equations](12k-quality.md); calculate both dimensions before generation. Requested generator resolution is not its actual output size. Native density is necessary but still requires material/detail review. For people, build `--stage repair` evidence on the generated output with crop-local face boxes and pass `record_region.py --subject-report`; every repaired face requires at least 120px. An ID alone does not establish target coverage.
5. Bind the accepted manifest during semantic approval.
6. Run the selected neural route once.
7. Composite once with `composite_regions.py --base --base-report --approval`. The report must identify the exact approved neural base. Mode, alpha and ICC are preserved; float accumulation uses the work volume and retains fractional precision across overlapping repairs.
8. Run one final native-pixel review and final checksum.

Do not shrink repair outputs into the semantic master. Do not replace a whole image for a local failure. Do not hide registration drift with blur, grain, glow, or compression. Preparation requires a new empty task directory. Recorders serialize manifest updates and install hash-named revisions atomically; partial copies cannot destroy accepted bytes. Keep prior revisions in the repair workspace, not the delivery directory.

## High-density semantic canvas

When detail is missing throughout a creatively reconstructable scene, generate registered tiles around one approved composition anchor. Choose tile framing that preserves aspect ratio and enough overlap to compare geometry, material construction and focus; do not reduce overlap to one pixel merely to fit a generator's aspect ratio. Place seams away from eyes, mouths, necklines and dominant structural edges where practical. Confirm actual overlap, generator context and subject density before `prepare_tiles.py --context-reviewed`; 1–2px overlaps are rejected, but a wider overlap is not automatic quality approval.

Record accepted native outputs with `record_tile.py`; an already valid structure can be recorded without an unnecessary structural generation. Stitch with `stitch_tiles.py manifest.json semantic-master.png --output-scale SCALE`. Choose SCALE from actual generated tile density, not the desired final resolution. The command preserves generated detail on that larger canvas and rejects enlargement beyond the smallest native tile support (allowing one pixel of boundary rounding). Its resized source is a geometry reference, not new detail evidence.

Inspect every tile join at native pixels even when registration correlation passes. Check texture construction and focus on both sides, not just edge alignment. Correct documented join defects with registered narrow targets before neural enlargement. Reserve separate high-density face/material packs when the stitched master cannot support the final viewing scale; never downsample those packs into the master. If final review finds a new local failure, replace only that repair and create a new approval filename binding the current manifest. When source bytes, source review and subject evidence are unchanged, the neural contract hash allows reuse of the existing base; changed source or review does not qualify. Recompose once and review the new final bytes. Never edit a runtime report to claim base reuse.
