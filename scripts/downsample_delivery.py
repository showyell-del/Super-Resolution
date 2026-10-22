#!/usr/bin/env python3
"""Derive a smaller candidate from an already approved neural delivery."""

import argparse
import json
from pathlib import Path

from PIL import Image

from pipeline_contract import atomic_json, load_approval, new_output, sha256


def validate_parent_delivery(path):
    from finalize_delivery import validate_native_review, validate_final_requirements, validate_execution_chain
    data = json.loads(path.read_text())
    if data.get("delivery_version") != 2:
        raise ValueError("Require a version 2 approved parent delivery")
    image = Path(data["final"]["path"])
    review_path = Path(data["final_review"]["path"])
    approval_path = Path(data["approval"]["path"])
    if (sha256(image) != data["final"]["sha256"]
            or sha256(review_path) != data["final_review"]["sha256"]
            or sha256(approval_path) != data["approval"]["sha256"]):
        raise ValueError("Approved parent image or evidence changed")
    approval = load_approval(approval_path)
    review = json.loads(review_path.read_text())
    with Image.open(image) as pixels:
        if list(pixels.size) != data["final"]["size"] or pixels.mode != data["final"]["mode"]:
            raise ValueError("Parent image metadata changed")
        validate_native_review(review, data["final"]["sha256"], pixels.size)
    subject = data.get("subject_report")
    if subject and sha256(Path(subject["path"])) != subject["sha256"]:
        raise ValueError("Parent subject report changed")
    validate_final_requirements(review, approval, image, Path(subject["path"]) if subject else None)
    for operation in data["operation_reports"]:
        actual = Path(operation["path"])
        if sha256(actual) != operation["sha256"] or json.loads(actual.read_text()) != operation["content"]:
            raise ValueError("Parent operation report changed")
    validate_execution_chain(approval_path, approval, data["operation_reports"], data["final"]["sha256"])
    return data


def downsample(parent, output, target):
    report = output.with_suffix(output.suffix + ".downsample-report.json")
    new_output(output)
    new_output(report)
    delivery = validate_parent_delivery(parent)
    with Image.open(delivery["final"]["path"]) as original:
        if min(target) < 1 or target[0] >= original.width or target[1] >= original.height:
            raise ValueError("Approved derivation must be smaller, not another enlargement")
        if target[0] * original.height != target[1] * original.width:
            raise ValueError("Downsample must preserve the exact approved aspect")
        options = {"icc_profile": original.info["icc_profile"]} if original.info.get("icc_profile") else {}
        output.parent.mkdir(parents=True, exist_ok=True)
        original.resize(target, Image.Resampling.LANCZOS).save(output, **options)
    atomic_json(report, {"operation": "approved_downsample", "parent_delivery": str(parent.resolve()),
                        "parent_delivery_sha256": sha256(parent), "source_sha256": delivery["final"]["sha256"],
                        "output_sha256": sha256(output), "target": list(target)})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_delivery", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    try:
        print(downsample(args.parent_delivery, args.output, (args.width, args.height)))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
