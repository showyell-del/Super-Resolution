# Registered repair pack

Use local regions for small subjects or isolated semantic failures.

1. List every required region before MPS. Include enough body or scene context for pose, scale, perspective, light, and registration.
2. Add subject_ids to every people region.
3. Extract with prepare_regions.py.
4. Generate each region at high density, inspect it at native pixels, and record it with record_region.py.
5. Bind the accepted manifest during semantic approval.
6. Run MPS once.
7. Composite the full repair pack once onto the enlarged base with composite_regions.py --base.
8. Run one final native-pixel review and final checksum.

Do not shrink repair outputs into the semantic master. Do not replace a whole image for a local failure. Do not hide registration drift with blur, grain, glow, or compression.
