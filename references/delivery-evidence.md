# Delivery evidence

Keep only evidence that drives a gate or identifies the exact delivered bytes:

- Apple/storage preflight and exact canvas report;
- locked scene review, protected-content data, and subject manifest when applicable;
- one subject precheck plus structured manual review;
- accepted repair manifest and final composite report when applicable;
- semantic approval, MPS report, final dimensions, mode, byte size, and SHA-256.

Run finalize_delivery.py after the last composite or export. A prior checksum is stale. Evidence proves file identity and completed gates; it does not replace visual inspection.
