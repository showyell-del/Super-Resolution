#!/usr/bin/env python3
"""Deterministic smoke test for canvas, sparse-region, subject, and delivery gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def run(*arguments: str, expect: int = 0) -> None:
    result = subprocess.run([sys.executable, *arguments], text=True, capture_output=True)
    if result.returncode != expect:
        raise RuntimeError(
            f"Expected exit {expect}, got {result.returncode}: {' '.join(arguments)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def visual_review() -> dict:
    return {
        "visual_contract": {
            "source_look": "graphic",
            "target_look": "graphic",
            "truth_mode": "evidence_preserving",
            "source_evidence": "Synthetic sharp-edged graphic fixture.",
            "target_reason": "Preserve the source graphic look.",
            "detail_policy": "Keep exact edges; do not add camera texture.",
        },
        "appearance_checks": {
            "style_consistency": "pass",
            "material_construction": "pass",
            "spatial_detail_hierarchy": "pass",
            "optics_or_markmaking": "pass",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path, help="empty test directory on a volume with >50 GiB free")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    scripts = root / "scripts"
    args.workspace.mkdir(parents=True, exist_ok=True)

    width, height = 641, 360
    array = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    array[..., 0] = (xx * 3 + yy) % 256
    array[..., 1] = (xx + yy * 5) % 256
    array[..., 2] = (xx * 7 + yy * 2) % 256
    source = args.workspace / "source.png"
    Image.fromarray(array).save(source)

    normalized = args.workspace / "normalized.png"
    run(str(scripts / "normalize_canvas.py"), str(source), str(normalized), "--target-width", "16", "--target-height", "9")
    with Image.open(normalized) as image:
        if image.size != (640, 360):
            raise RuntimeError(f"Unexpected normalized size: {image.size}")

    regions_file = args.workspace / "regions.json"
    regions_file.write_text(json.dumps([{"name": "center", "x": 160, "y": 90, "width": 320, "height": 180}]), encoding="utf-8")
    regions_dir = args.workspace / "regions"
    run(str(scripts / "prepare_regions.py"), str(normalized), str(regions_file), str(regions_dir), "--context", "0")
    crop = regions_dir / "center-input.png"
    prompt = args.workspace / "region-prompt.txt"
    prompt.write_text("Preserve the exact deterministic fixture.", encoding="utf-8")
    run(str(scripts / "record_region.py"), str(regions_dir / "regions-manifest.json"), "center", str(crop), "--prompt-file", str(prompt), "--review-note", "Exact deterministic crop accepted.")
    normalized_review = args.workspace / "normalized-review.json"
    normalized_review.write_text(json.dumps({
        **visual_review(),
        "review_scale": "100% native pixels",
        "subject_report_required": False,
        "global_checks": {
            "geometry_and_perspective": "pass",
            "line_topology_and_edge_ownership": "not_applicable",
            "repeated_structures": "not_applicable",
            "emitters_reflections_and_bloom": "not_applicable",
            "organic_support_topology": "not_applicable",
            "far_field_and_depth_falloff": "not_applicable",
            "protected_content": "not_applicable"
        },
        "regions": [{
            "name": "fixture", "box": [0, 0, 640, 360], "status": "pass",
            "note": "Deterministic fixture inspected."
        }],
        "reviewer_note": "Deterministic fixture accepted."
    }), encoding="utf-8")
    normalized_approval = args.workspace / "normalized-approval.json"
    run(
        str(scripts / "approve_semantic_master.py"), str(normalized), str(normalized_review),
        "--repair-manifest", str(regions_dir / "regions-manifest.json"),
        "--output", str(normalized_approval),
    )
    composite = args.workspace / "composite.png"
    run(str(scripts / "composite_regions.py"), str(regions_dir / "regions-manifest.json"), str(composite))
    enlarged_base = args.workspace / "enlarged-base.png"
    Image.open(normalized).resize((1280, 720), Image.Resampling.LANCZOS).save(enlarged_base)
    enlarged_composite = args.workspace / "enlarged-composite.png"
    run(
        str(scripts / "composite_regions.py"),
        str(regions_dir / "regions-manifest.json"),
        str(enlarged_composite),
        "--base", str(enlarged_base),
        "--approval", str(normalized_approval),
    )
    with Image.open(enlarged_composite) as image:
        if image.size != (1280, 720):
            raise RuntimeError(f"Unexpected enlarged composite size: {image.size}")

    people = Image.new("RGB", (400, 200), "#777777")
    draw = ImageDraw.Draw(people)
    draw.ellipse((30, 30, 129, 129), fill="#c58d65", outline="black", width=3)
    draw.ellipse((260, 30, 359, 129), fill="#8d5f43", outline="white", width=3)
    for offset in (0, 230):
        draw.ellipse((55 + offset, 60, 65 + offset, 70), fill="black")
        draw.ellipse((92 + offset, 60, 102 + offset, 70), fill="black")
    people_path = args.workspace / "people.png"
    people.save(people_path)
    subjects = args.workspace / "subjects.json"
    subjects.write_text(json.dumps({"subjects": [
        {"id": "a", "face_box": [45, 45, 60, 60], "identity": {"distinguishing_features": "round face"}},
        {"id": "b", "face_box": [255, 25, 110, 110], "identity": {"distinguishing_features": "angular face"}}
    ]}), encoding="utf-8")
    run(
        str(scripts / "build_contact_sheet.py"), str(people_path), str(subjects),
        str(args.workspace / "legacy-contact"), "--min-edit-face-width", "30", expect=2,
    )
    run(
        str(scripts / "build_contact_sheet.py"), str(people_path), str(subjects),
        str(args.workspace / "contact"), "--stage", "semantic", "--min-sharpness", "1",
        "--max-hash-distance", "0", expect=2,
    )
    subject_report = json.loads((args.workspace / "contact" / "contact-sheet-report.json").read_text())
    if "pass" in subject_report or subject_report.get("manual_semantic_review_required") is not True:
        raise RuntimeError("Automated subject metrics must not approve facial semantics")

    people_regions = args.workspace / "people-regions.json"
    people_regions.write_text(json.dumps([{
        "name": "repair-a", "category": "people", "subject_ids": ["a"],
        "x": 0, "y": 0, "width": 180, "height": 180
    }]), encoding="utf-8")
    people_repairs = args.workspace / "people-repairs"
    run(str(scripts / "prepare_regions.py"), str(people_path), str(people_regions), str(people_repairs), "--context", "0")
    repair_prompt = args.workspace / "people-repair-prompt.txt"
    repair_prompt.write_text("Preserve the fixture and reconstruct subject a.", encoding="utf-8")
    run(
        str(scripts / "record_region.py"), str(people_repairs / "regions-manifest.json"),
        "repair-a", str(people_repairs / "repair-a-input.png"),
        "--prompt-file", str(repair_prompt), "--review-note", "Subject a repair inspected.",
    )

    review = args.workspace / "native-review.json"
    review.write_text(json.dumps({
        **visual_review(),
        "review_scale": "100% native pixels",
        "subject_report_required": True,
        "global_checks": {
            "geometry_and_perspective": "pass",
            "line_topology_and_edge_ownership": "not_applicable",
            "repeated_structures": "not_applicable",
            "emitters_reflections_and_bloom": "not_applicable",
            "organic_support_topology": "not_applicable",
            "far_field_and_depth_falloff": "not_applicable",
            "protected_content": "not_applicable"
        },
        "regions": [{"name": "people", "box": [0, 0, 400, 200], "status": "pass", "note": "Two distinct fixture subjects reviewed."}],
        "reviewer_note": "Synthetic smoke-test fixture accepted.",
        "people_review": {
            "checks": {
                "eyes_and_gaze": "pass",
                "mouth_and_teeth": "pass",
                "skin_texture": "pass",
                "identity_and_distinctness": "pass",
                "hair_hands_and_anatomy": "pass",
                "wardrobe_and_material_integration": "pass"
            },
            "subjects": [
                {"id": "a", "status": "pass", "delivery_strategy": "final_registered_repair", "note": "Fixture A repair inspected."},
                {"id": "b", "status": "pass", "delivery_strategy": "semantic_master", "note": "Fixture B inspected."}
            ]
        }
    }), encoding="utf-8")
    missing_style_review = args.workspace / "missing-style-review.json"
    missing_style_review.write_text(json.dumps({
        key: value for key, value in json.loads(review.read_text(encoding="utf-8")).items()
        if key not in {"visual_contract", "appearance_checks"}
    }), encoding="utf-8")
    run(str(scripts / "approve_semantic_master.py"), str(people_path), str(missing_style_review), expect=2)
    failed_appearance = args.workspace / "failed-appearance-review.json"
    failed_data = json.loads(review.read_text(encoding="utf-8"))
    failed_data["appearance_checks"]["material_construction"] = "fail"
    failed_appearance.write_text(json.dumps(failed_data), encoding="utf-8")
    run(str(scripts / "approve_semantic_master.py"), str(people_path), str(failed_appearance), expect=2)
    invalid_conversion = args.workspace / "invalid-conversion-review.json"
    invalid_data = json.loads(review.read_text(encoding="utf-8"))
    invalid_data["visual_contract"]["target_look"] = "camera_photo"
    invalid_conversion.write_text(json.dumps(invalid_data), encoding="utf-8")
    run(str(scripts / "approve_semantic_master.py"), str(people_path), str(invalid_conversion), expect=2)
    run(str(scripts / "approve_semantic_master.py"), str(people_path), str(review), expect=2)
    run(
        str(scripts / "approve_semantic_master.py"), str(people_path), str(review),
        "--subject-report", str(args.workspace / "contact" / "contact-sheet-report.json"), expect=2,
    )
    run(
        str(scripts / "approve_semantic_master.py"), str(people_path), str(review),
        "--subject-report", str(args.workspace / "contact" / "contact-sheet-report.json"),
        "--repair-manifest", str(people_repairs / "regions-manifest.json"),
    )

    final_report = args.workspace / "delivery.json"
    final_review = args.workspace / "final-review.json"
    final_review.write_text(json.dumps({
        "review_scale": "100% native pixels",
        "image_sha256": hashlib.sha256(composite.read_bytes()).hexdigest(),
        "appearance_checks": visual_review()["appearance_checks"],
        "regions": [{"name": "fixture", "box": [0, 0, 640, 360], "status": "pass", "note": "Exact fixture inspected at delivery size."}],
        "reviewer_note": "Final graphic fixture accepted."
    }), encoding="utf-8")
    stale_review = args.workspace / "stale-final-review.json"
    stale_data = json.loads(final_review.read_text(encoding="utf-8"))
    stale_data["image_sha256"] = "0" * 64
    stale_review.write_text(json.dumps(stale_data), encoding="utf-8")
    run(str(scripts / "finalize_delivery.py"), str(composite), str(final_report), "--width", "640", "--height", "360", "--operation-report", str(composite) + ".composite-report.json", expect=2)
    run(str(scripts / "finalize_delivery.py"), str(composite), str(final_report), "--width", "640", "--height", "360", "--final-review", str(stale_review), expect=2)
    run(str(scripts / "finalize_delivery.py"), str(composite), str(final_report), "--width", "640", "--height", "360", "--final-review", str(final_review), "--operation-report", str(composite) + ".composite-report.json")
    print(json.dumps({"pass": True, "workspace": str(args.workspace.resolve())}, indent=2))


if __name__ == "__main__":
    main()
