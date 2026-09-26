#!/usr/bin/env python3
"""Create a hash-bound approval after structured native-pixel review."""

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
REQUIRED_PEOPLE_CHECKS = {
    "eyes_and_gaze",
    "mouth_and_teeth",
    "skin_texture",
    "identity_and_distinctness",
    "hair_hands_and_anatomy",
    "wardrobe_and_material_integration",
}
VALID_RESULTS = {"pass", "not_applicable"}
VALID_SUBJECT_STRATEGIES = {"semantic_master", "final_registered_repair"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_repair_manifest(path: Path, master: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if Path(str(data.get("source", ""))).resolve() != master.resolve():
        raise ValueError("Repair manifest belongs to another semantic master")
    if data.get("source_sha256") != sha256(master):
        raise ValueError("Semantic master changed after repair regions were prepared")
    covered: set[str] = set()
    regions = data.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Repair manifest must contain at least one region")
    for item in regions:
        if item.get("status") != "accepted":
            raise ValueError(f"Repair region {item.get('name', '<unnamed>')} is not accepted")
        output = path.parent / str(item.get("output", ""))
        if not output.is_file() or sha256(output) != item.get("output_sha256"):
            raise ValueError(f"Repair region {item.get('name', '<unnamed>')} output is missing or changed")
        subject_ids = item.get("subject_ids", [])
        if not isinstance(subject_ids, list):
            raise ValueError(f"Repair region {item.get('name', '<unnamed>')} has invalid subject_ids")
        covered.update(str(subject_id).strip() for subject_id in subject_ids if str(subject_id).strip())
    return covered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--subject-report",
        type=Path,
        help="subject precheck report when people or faces are acceptance-critical",
    )
    parser.add_argument(
        "--repair-manifest",
        type=Path,
        help="accepted high-density repair pack for subjects that cannot pass in the semantic master",
    )
    args = parser.parse_args()

    output = args.output or args.master.with_suffix(args.master.suffix + ".approval.json")
    paths = [args.master, args.review, output.parent]
    if args.subject_report:
        paths.append(args.subject_report)
    if args.repair_manifest:
        paths.append(args.repair_manifest)
    require_apple_silicon(paths)
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

    repair_subjects: set[str] = set()
    if args.repair_manifest:
        if not args.repair_manifest.is_file():
            parser.error(f"Missing repair manifest: {args.repair_manifest}")
        try:
            repair_subjects = load_repair_manifest(args.repair_manifest, args.master)
        except ValueError as exc:
            parser.error(str(exc))

    subject_report = None
    if review.get("subject_report_required") is True and not args.subject_report:
        parser.error("This review requires --subject-report")
    if args.subject_report:
        if not args.subject_report.is_file():
            parser.error(f"Missing subject report: {args.subject_report}")
        subject_report = json.loads(args.subject_report.read_text(encoding="utf-8"))
        if subject_report.get("schema_version") != 2 or subject_report.get("stage") != "semantic":
            parser.error("Subject report must use schema_version 2 and stage semantic")
        if subject_report.get("manual_semantic_review_required") is not True:
            parser.error("Subject report does not enforce manual semantic review")
        report_image = Path(str(subject_report.get("image", "")))
        if report_image.resolve() != args.master.resolve():
            parser.error("Subject report belongs to another semantic master")
        if subject_report.get("image_sha256") != sha256(args.master):
            parser.error("Semantic master changed after the subject report was created")

        people_review = review.get("people_review")
        if not isinstance(people_review, dict):
            parser.error("people_review is required when a subject report is supplied")
        people_checks = people_review.get("checks")
        if not isinstance(people_checks, dict) or set(people_checks) != REQUIRED_PEOPLE_CHECKS:
            parser.error(f"people_review.checks must contain exactly: {sorted(REQUIRED_PEOPLE_CHECKS)}")
        if any(value != "pass" for value in people_checks.values()):
            parser.error("Every people_review check must be pass")

        report_entries = subject_report.get("subjects")
        if not isinstance(report_entries, list) or not report_entries:
            parser.error("Subject report must contain subjects")
        report_subjects = {str(entry.get("id", "")).strip(): entry for entry in report_entries}
        if "" in report_subjects or len(report_subjects) != len(report_entries):
            parser.error("Subject report contains empty or duplicate IDs")
        reviewed = people_review.get("subjects")
        if not isinstance(reviewed, list):
            parser.error("people_review.subjects must be a list")
        reviewed_by_id = {str(item.get("id", "")).strip(): item for item in reviewed}
        if len(reviewed_by_id) != len(reviewed) or set(reviewed_by_id) != set(report_subjects) or "" in reviewed_by_id:
            parser.error("people_review.subjects must match every subject-report ID exactly")
        for subject_id, entry in report_subjects.items():
            item = reviewed_by_id[subject_id]
            strategy = item.get("delivery_strategy")
            if item.get("status") != "pass" or not str(item.get("note", "")).strip():
                parser.error(f"Subject {subject_id!r} needs status pass and a concrete note")
            if strategy not in VALID_SUBJECT_STRATEGIES:
                parser.error(f"Subject {subject_id!r} has invalid delivery_strategy")
            needs_repair = bool(entry.get("automated_failures"))
            if needs_repair and strategy != "final_registered_repair":
                parser.error(f"Subject {subject_id!r} failed precheck and must use final_registered_repair")
            if strategy == "final_registered_repair" and subject_id not in repair_subjects:
                parser.error(f"Subject {subject_id!r} is not covered by the accepted repair manifest")

    approval = {
        "approval_version": 2,
        "approved": True,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "master": str(args.master.resolve()),
        "master_sha256": sha256(args.master),
        "master_size": [width, height],
        "master_mode": mode,
        "review": review,
        "subject_report": (
            {"path": str(args.subject_report.resolve()), "sha256": sha256(args.subject_report)}
            if args.subject_report else None
        ),
        "repair_manifest": (
            {"path": str(args.repair_manifest.resolve()), "sha256": sha256(args.repair_manifest)}
            if args.repair_manifest else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
