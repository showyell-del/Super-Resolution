#!/usr/bin/env python3
"""Create a hash-complete final delivery report across semantic and MPS stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from apple_preflight import require_apple_silicon

REQUIRED_APPEARANCE_CHECKS = {
    "style_consistency", "material_construction", "spatial_detail_hierarchy", "optics_or_markmaking"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Artifacts must use role=/absolute/path")
    role, raw_path = value.split("=", 1)
    if not role.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Artifacts need a non-empty role and path")
    return role.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_image", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--final-review", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], type=parse_artifact)
    parser.add_argument("--operation-report", action="append", default=[], type=Path)
    args = parser.parse_args()

    paths = [args.final_image, args.final_review, args.report.parent]
    paths.extend(path for _, path in args.artifact)
    paths.extend(args.operation_report)
    require_apple_silicon(paths)
    if not args.final_image.is_file():
        parser.error(f"Missing final image: {args.final_image}")
    with Image.open(args.final_image) as image:
        if image.size != (args.width, args.height):
            parser.error(f"Expected {(args.width, args.height)}, got {image.size}")
        final_metadata = {
            "path": str(args.final_image.resolve()),
            "sha256": sha256(args.final_image),
            "bytes": args.final_image.stat().st_size,
            "size": list(image.size),
            "mode": image.mode,
            "icc_profile": bool(image.info.get("icc_profile")),
        }

    if not args.final_review.is_file():
        parser.error(f"Missing final native-pixel review: {args.final_review}")
    review = json.loads(args.final_review.read_text(encoding="utf-8"))
    if review.get("review_scale") != "100% native pixels":
        parser.error("Final review must inspect 100% native pixels")
    if review.get("image_sha256") != final_metadata["sha256"]:
        parser.error("Final review does not match the delivered image bytes")
    checks = review.get("appearance_checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_APPEARANCE_CHECKS:
        parser.error(f"Final appearance_checks must contain exactly: {sorted(REQUIRED_APPEARANCE_CHECKS)}")
    if any(value != "pass" for value in checks.values()):
        parser.error("Final appearance checks have not all passed")
    regions = review.get("regions")
    if not isinstance(regions, list) or not regions:
        parser.error("Final review needs at least one representative native-pixel region")
    for item in regions:
        box = item.get("box") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, int) for value in box):
            parser.error("Final review regions need integer [x,y,width,height] boxes")
        x, y, w, h = box
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > args.width or y + h > args.height:
            parser.error("Final review region exceeds the delivered canvas")
        if item.get("status") != "pass" or not str(item.get("note", "")).strip():
            parser.error("Every final review region needs a pass and concrete note")
    if not str(review.get("reviewer_note", "")).strip():
        parser.error("Final review needs a concrete reviewer_note")

    artifacts = []
    for role, path in args.artifact:
        if not path.is_file():
            parser.error(f"Missing artifact for {role}: {path}")
        artifacts.append({"role": role, "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size})
    operations = []
    for path in args.operation_report:
        if not path.is_file():
            parser.error(f"Missing operation report: {path}")
        operations.append({"path": str(path.resolve()), "sha256": sha256(path), "content": json.loads(path.read_text(encoding="utf-8"))})

    report = {
        "delivery_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final": final_metadata,
        "final_review": {"path": str(args.final_review.resolve()), "sha256": sha256(args.final_review)},
        "artifacts": artifacts,
        "operation_reports": operations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
