#!/usr/bin/env python3
"""Create a hash-bound native-pixel approval for a semantic master."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from apple_preflight import require_apple_silicon


REQUIRED_CHECKS = {
    "geometry_and_perspective",
    "line_topology_and_edge_ownership",
    "repeated_structures",
    "emitters_reflections_and_bloom",
    "organic_support_topology",
    "far_field_and_depth_falloff",
    "protected_content",
}
VALID_RESULTS = {"pass", "not_applicable"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.master.with_suffix(args.master.suffix + ".approval.json")
    require_apple_silicon([args.master, args.review, output.parent])
    if not args.master.is_file():
        parser.error(f"Missing semantic master: {args.master}")
    if not args.review.is_file():
        parser.error(f"Missing native-pixel review: {args.review}")

    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("review_scale") != "100% native pixels":
        parser.error("review_scale must be exactly '100% native pixels'")
    checks = review.get("global_checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS:
        parser.error(f"global_checks must contain exactly: {sorted(REQUIRED_CHECKS)}")
    invalid = {name: result for name, result in checks.items() if result not in VALID_RESULTS}
    if invalid:
        parser.error(f"Every global check must be pass or not_applicable; invalid: {invalid}")
    structural = REQUIRED_CHECKS - {"protected_content"}
    if not any(checks[name] == "pass" for name in structural):
        parser.error("At least one structural category must be inspected and passed")

    with Image.open(args.master) as image:
        width, height = image.size
        mode = image.mode
    regions = review.get("regions")
    if not isinstance(regions, list) or not regions:
        parser.error("At least one native-pixel review region is required")
    names = set()
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            parser.error(f"Region {index} must be an object")
        name = str(region.get("name", "")).strip()
        note = str(region.get("note", "")).strip()
        box = region.get("box")
        if not name or name in names:
            parser.error(f"Region {index} needs a unique non-empty name")
        names.add(name)
        if region.get("status") != "pass" or not note:
            parser.error(f"Region {name!r} must have status pass and a non-empty note")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, int) for value in box):
            parser.error(f"Region {name!r} box must be [x,y,width,height] integers")
        x, y, region_width, region_height = box
        if x < 0 or y < 0 or region_width < 1 or region_height < 1:
            parser.error(f"Region {name!r} has invalid bounds")
        if x + region_width > width or y + region_height > height:
            parser.error(f"Region {name!r} exceeds the semantic master")
    if not str(review.get("reviewer_note", "")).strip():
        parser.error("reviewer_note is required")

    approval = {
        "approval_version": 1,
        "approved": True,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "master": str(args.master.resolve()),
        "master_sha256": sha256(args.master),
        "master_size": [width, height],
        "master_mode": mode,
        "review": review,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
