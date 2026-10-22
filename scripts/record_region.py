#!/usr/bin/env python3
"""Record a byte-bound native-density region under a serialized manifest lock."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image
from pipeline_contract import local_file, locked_manifest, sha256, verify_subject_report, immutable_artifact


def record(data, args):
    if sha256(Path(data["source"])) != data["source_sha256"]:
        raise ValueError("Source changed after preparation")
    matches = [item for item in data["regions"] if item["name"] == args.region]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate region")
    item = matches[0]
    with Image.open(args.output) as image:
        width, height = image.size
    _, _, cw, ch = item["crop_box"]
    error = abs((width / height) / (cw / ch) - 1)
    if error > args.max_aspect_error or width < cw or height < ch:
        raise ValueError("Region aspect or native density is insufficient")
    destination = local_file(args.manifest.parent, item["output"])
    if destination.resolve() == Path(data["source"]).resolve():
        raise ValueError("Repair destination cannot overwrite source")
    report = None
    if item.get("subject_ids"):
        if not args.subject_report:
            raise ValueError("People regions require --subject-report from the repair output")
        report = json.loads(args.subject_report.read_text())
        verify_subject_report(report, args.output, "repair", item["subject_ids"])
    prompt = args.prompt_file.read_text()
    destination = immutable_artifact(args.manifest.parent, item["name"], args.output)
    item.update(status="accepted", output=destination.name, prompt=prompt, review_note=args.review_note.strip(),
                output_size=[width, height], aspect_error=error, output_sha256=sha256(destination),
                subject_report=report)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("region")
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--subject-report", type=Path)
    parser.add_argument("--max-aspect-error", type=float, default=0.002)
    args = parser.parse_args()
    if not math.isfinite(args.max_aspect_error) or not 0 <= args.max_aspect_error <= 0.002:
        parser.error("Aspect tolerance must be finite and between 0 and 0.002")
    if not args.review_note.strip() or not args.prompt_file.read_text().strip():
        parser.error("Prompt and native-pixel review note must be non-empty")
    try:
        with locked_manifest(args.manifest) as data:
            destination = record(data, args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(destination)


if __name__ == "__main__":
    main()
