#!/usr/bin/env python3
"""Deterministic smoke test for canvas, sparse-region, subject, and delivery gates."""

from __future__ import annotations

import argparse
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
    composite = args.workspace / "composite.png"
    run(str(scripts / "composite_regions.py"), str(regions_dir / "regions-manifest.json"), str(composite))

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
        {"id": "a", "face_box": [25, 25, 110, 110], "identity": {"distinguishing_features": "round face"}},
        {"id": "b", "face_box": [255, 25, 110, 110], "identity": {"distinguishing_features": "angular face"}}
    ]}), encoding="utf-8")
    run(str(scripts / "build_contact_sheet.py"), str(people_path), str(subjects), str(args.workspace / "contact"), "--min-sharpness", "1", "--max-hash-distance", "0")

    review = args.workspace / "native-review.json"
    review.write_text(json.dumps({
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
        "reviewer_note": "Synthetic smoke-test fixture accepted."
    }), encoding="utf-8")
    run(str(scripts / "approve_semantic_master.py"), str(people_path), str(review), expect=2)
    run(str(scripts / "approve_semantic_master.py"), str(people_path), str(review), "--subject-report", str(args.workspace / "contact" / "contact-sheet-report.json"))

    final_report = args.workspace / "delivery.json"
    run(str(scripts / "finalize_delivery.py"), str(composite), str(final_report), "--width", "640", "--height", "360", "--operation-report", str(composite) + ".composite-report.json")
    print(json.dumps({"pass": True, "workspace": str(args.workspace.resolve())}, indent=2))


if __name__ == "__main__":
    main()
