#!/usr/bin/env python3
"""Record a reviewed sparse-region edit in its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("region")
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--max-aspect-error", type=float, default=0.002)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = [item for item in data["regions"] if item["name"] == args.region]
    if len(matches) != 1:
        parser.error(f"Unknown or duplicate region: {args.region}")
    if not args.output.is_file() or not args.prompt_file.is_file():
        parser.error("Output image and prompt file must exist")
    if not args.review_note.strip():
        parser.error("A native-pixel review note is required")

    item = matches[0]
    _, _, expected_width, expected_height = item["crop_box"]
    with Image.open(args.output) as image:
        width, height = image.size
    aspect_error = abs((width / height) / (expected_width / expected_height) - 1)
    if aspect_error > args.max_aspect_error:
        parser.error(f"Region aspect changed by {aspect_error:.4%}")

    destination = args.manifest.parent / item["output"]
    if args.output.resolve() != destination.resolve():
        destination.write_bytes(args.output.read_bytes())
    item["status"] = "accepted"
    item["prompt"] = args.prompt_file.read_text(encoding="utf-8")
    item["review_note"] = args.review_note.strip()
    item["output_size"] = [width, height]
    item["aspect_error"] = aspect_error
    item["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    temporary = args.manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.manifest)
    print(destination)


if __name__ == "__main__":
    main()
