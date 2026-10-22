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
from pipeline_contract import (load_approval, verify_neural_report, verify_subject_report,
                               PEOPLE_CHECKS, contains, atomic_json, new_output, validate_resolution)

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


def validate_native_review(review: dict, image_hash: str, size: tuple[int, int]) -> None:
    """Validate the same byte-bound review for a trial or final image."""
    if review.get("review_scale") != "100% native pixels":
        raise ValueError("Review must inspect 100% native pixels")
    if review.get("image_sha256") != image_hash:
        raise ValueError("Review does not match the image bytes")
    checks = review.get("appearance_checks")
    if not isinstance(checks, dict) or set(checks) != REQUIRED_APPEARANCE_CHECKS:
        raise ValueError(f"appearance_checks must contain exactly: {sorted(REQUIRED_APPEARANCE_CHECKS)}")
    if any(value != "pass" for value in checks.values()):
        raise ValueError("Appearance checks have not all passed")
    global_checks = review.get("global_checks", {})
    if not isinstance(global_checks, dict) or any(v not in {"pass", "not_applicable"} for v in global_checks.values()):
        raise ValueError("Global/protected content checks failed")
    people = review.get("people_review")
    if people is not None:
        if not isinstance(people, dict) or set(people.get("checks", {})) != PEOPLE_CHECKS:
            raise ValueError("People review requires all facial/material checks")
        if any(v != "pass" for v in people["checks"].values()):
            raise ValueError("People checks failed")
        for person in people.get("subjects", []):
            if person.get("status") != "pass" or not str(person.get("note", "")).strip():
                raise ValueError("Final subject failed or lacks a concrete note")
    regions = review.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Review needs at least one representative native-pixel region")
    for item in regions:
        box = item.get("box") if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4 or not all(type(value) is int for value in box):
            raise ValueError("Review regions need integer [x,y,width,height] boxes")
        x, y, w, h = box
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > size[0] or y + h > size[1]:
            raise ValueError("Review region exceeds the image canvas")
        if item.get("status") != "pass" or not str(item.get("note", "")).strip():
            raise ValueError("Every review region needs a pass and concrete note")
    if not str(review.get("reviewer_note", "")).strip():
        raise ValueError("Review needs a concrete reviewer_note")


def validate_final_requirements(review, approval, image, subject_path):
    with Image.open(image) as pixels, Image.open(approval["master"]) as original:
        validate_resolution(approval, pixels.size)
        if pixels.mode != original.mode or pixels.info.get("icc_profile") != original.info.get("icc_profile"):
            raise ValueError("Final mode or ICC profile changed from the approved master")
    expected = approval["review"]
    for key, value in expected["global_checks"].items():
        if value == "pass" and review.get("global_checks", {}).get(key) != "pass":
            raise ValueError(f"Final review must pass locked check {key}")
    ids = [item["id"] for item in expected.get("people_review", {}).get("subjects", [])]
    if expected.get("subject_report_required") or ids:
        if subject_path is None:
            raise ValueError("Final photographic people require --subject-report")
        report = json.loads(subject_path.read_text())
        subjects = verify_subject_report(report, image, "final", ids)
        reviewed = review.get("people_review", {}).get("subjects", [])
        if len(reviewed) != len(ids) or {item.get("id") for item in reviewed} != set(ids):
            raise ValueError("Final review must cover every approved subject ID exactly")
        for item in subjects.values():
            if not any(contains(region["box"], item["face_box"]) for region in review["regions"]):
                raise ValueError("Final review regions do not cover every critical face")


def validate_execution_chain(approval_path, approval, operations, final_hash):
    if not operations:
        raise ValueError("Final delivery requires a neural execution chain")
    neural = operations[0]["content"]
    current = neural.get("output_sha256")
    verify_neural_report(neural, approval_path, approval, current)
    repair = approval.get("repair_manifest")
    repaired = False
    for operation in operations[1:]:
        item = operation["content"]
        if item.get("operation") == "registered_composite":
            if item.get("base_sha256") != current or item.get("source_sha256") != approval["master_sha256"]:
                raise ValueError("Composite source/base hash chain changed")
            if not repair or item.get("manifest_sha256") != repair["sha256"]:
                raise ValueError("Composite is not bound to the approved repair pack")
            manifest = json.loads(Path(repair["path"]).read_text())
            expected = {r["name"]: r["output_sha256"] for r in manifest["regions"]}
            actual = {r["name"]: r["output_sha256"] for r in item["regions"]}
            if actual != expected or len(item["regions"]) != len(expected):
                raise ValueError("Composite does not contain every approved repair exactly once")
            repaired = True
        elif item.get("operation") == "approved_downsample":
            parent = Path(item["parent_delivery"])
            from downsample_delivery import validate_parent_delivery
            delivery = validate_parent_delivery(parent)
            if (sha256(parent) != item.get("parent_delivery_sha256")
                    or delivery.get("delivery_version") != 2 or delivery["final"]["sha256"] != current
                    or item.get("source_sha256") != current):
                raise ValueError("Downsample is not derived from an approved parent delivery")
        else:
            raise ValueError("Unsupported operation in final execution chain")
        current = item.get("output_sha256")
    if repair and not repaired:
        raise ValueError("Approved repair pack has not been composited")
    if current != final_hash:
        raise ValueError("Execution chain does not end at the delivered image hash")


def validate_delivery_directory(image):
    formats = set(Image.registered_extensions()) | {".heic", ".heif", ".jxl", ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf"}
    candidates = [path for path in image.parent.iterdir() if path.suffix.lower() in formats and path.is_file()]
    if len(candidates) != 1 or candidates[0].resolve() != image.resolve():
        raise ValueError("Delivery directory must contain exactly one image; keep trials and bases in scratch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_image", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--final-review", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--subject-report", type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=parse_artifact)
    parser.add_argument("--operation-report", action="append", required=True, type=Path)
    args = parser.parse_args()

    paths = [args.final_image, args.final_review, args.report.parent]
    paths.extend(path for _, path in args.artifact)
    paths.extend(args.operation_report)
    require_apple_silicon(paths)
    new_output(args.report)
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
    try:
        validate_native_review(review, final_metadata["sha256"], (args.width, args.height))
        approval = load_approval(args.approval)
        validate_delivery_directory(args.final_image)
        validate_final_requirements(review, approval, args.final_image, args.subject_report)
    except ValueError as error:
        parser.error(str(error))

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

    try:
        validate_execution_chain(args.approval, approval, operations, final_metadata["sha256"])
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    report = {
        "delivery_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final": final_metadata,
        "final_review": {"path": str(args.final_review.resolve()), "sha256": sha256(args.final_review)},
        "artifacts": artifacts,
        "operation_reports": operations,
        "approval": {"path": str(args.approval.resolve()), "sha256": sha256(args.approval)},
        "subject_report": ({"path": str(args.subject_report.resolve()), "sha256": sha256(args.subject_report)}
                           if args.subject_report else None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(args.report)


if __name__ == "__main__":
    main()
