# Sparse region repair

Use sparse repair when a whole-frame semantic pass cannot allocate enough pixels to small complex subjects, or when final native-pixel inspection reveals isolated semantic failures.

1. Define only failed regions with `assets/regions.json`; include enough surrounding body, architecture, or stage context to preserve scale and lighting.
2. Run `prepare_regions.py` to extract context crops and a manifest.
3. Reconstruct one region at a time using the global scene anchor and relevant subject IDs. The accepted output must keep the source crop's exact aspect ratio.
4. Inspect the region at native pixels, then record it with `record_region.py` and a concrete review note.
5. Run `composite_regions.py`. It registers every region to the source, rejects excessive translation or scale drift, and feather-blends only accepted regions.
6. Rebuild the contact sheet and all downstream approvals from the new composite. Any earlier SHA-bound report is invalid.

Preferred order is semantic repair before neural enlargement. A final-size sparse repair is allowed only when it does not resize the canvas and when its region, registration, mask, composite report, and new final checksum are included in delivery evidence.

Never replace an entire image to fix a few small people. Never hide registration errors with blur, grain, glow, or compression.
