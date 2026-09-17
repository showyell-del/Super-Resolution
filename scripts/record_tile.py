#!/usr/bin/env python3
"""Atomically record an accepted structural or semantic tile stage."""

import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tile")
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--stage", choices=("structure", "semantic"), required=True)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--max-aspect-error", type=float, default=0.002)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = [tile for tile in data["tiles"] if tile["name"] == args.tile]
    if len(matches) != 1:
        parser.error(f"Unknown or duplicate tile name: {args.tile}")
    if not args.output.is_file():
        parser.error(f"Missing output tile: {args.output}")
    if not args.prompt_file.is_file():
        parser.error(f"Missing prompt file: {args.prompt_file}")
    if not args.review_note.strip():
        parser.error("A native-pixel review note is required")

    tile = matches[0]
    with Image.open(args.output) as image:
        output_width, output_height = image.size
    expected_aspect = tile["width"] / tile["height"]
    output_aspect = output_width / output_height
    relative_error = abs(output_aspect / expected_aspect - 1.0)
    if relative_error > args.max_aspect_error:
        parser.error(
            f"Tile aspect ratio changed by {relative_error:.4%}; regenerate instead of distorting it"
        )

    if args.stage == "semantic" and tile.get("structural_status") != "accepted":
        parser.error(f"Tile {args.tile} has no accepted structural pass")
    destination_key = "structure_output" if args.stage == "structure" else "output"
    destination = args.manifest.parent / tile[destination_key]
    if args.output.resolve() != destination.resolve():
        destination.write_bytes(args.output.read_bytes())
    prompt = args.prompt_file.read_text(encoding="utf-8")
    output_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    if args.stage == "structure":
        tile["structural_status"] = "accepted"
        tile["structure_prompt"] = prompt
        tile["structure_review_note"] = args.review_note.strip()
        tile["structure_output_sha256"] = output_hash
        tile["structure_output_size"] = [output_width, output_height]
        tile["structure_aspect_error"] = relative_error
        tile["semantic_status"] = "pending"
        tile["semantic_prompt"] = None
        tile["semantic_review_note"] = None
        tile["output_sha256"] = None
    else:
        tile["semantic_status"] = "accepted"
        tile["semantic_prompt"] = prompt
        tile["semantic_review_note"] = args.review_note.strip()
        tile["output_sha256"] = output_hash
        tile["output_size"] = [output_width, output_height]
        tile["aspect_error"] = relative_error

    temporary = args.manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.manifest)
    print(destination)


if __name__ == "__main__":
    main()
