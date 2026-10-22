#!/usr/bin/env python3
"""Serialize accepted tile stages without losing concurrent manifest updates."""

import argparse
import math
from pathlib import Path

from PIL import Image
from pipeline_contract import locked_manifest, local_file, sha256, immutable_artifact


def record(data, args):
    if sha256(Path(data["source"])) != data["source_sha256"]:
        raise ValueError("Source changed after preparation")
    matches = [tile for tile in data["tiles"] if tile["name"] == args.tile]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate tile")
    tile = matches[0]
    with Image.open(args.output) as image:
        width, height = image.size
    error = abs((width / height) / (tile["width"] / tile["height"]) - 1)
    if error > args.max_aspect_error or width < tile["width"] or height < tile["height"]:
        raise ValueError("Tile aspect or native density is insufficient")
    if args.stage == "semantic" and tile.get("structural_status") != "accepted":
        raise ValueError("Semantic tile requires an accepted structural stage")
    key = "structure_output" if args.stage == "structure" else "output"
    destination = local_file(args.manifest.parent, tile[key])
    if destination.resolve() == Path(data["source"]).resolve():
        raise ValueError("Tile cannot overwrite source")
    prompt = args.prompt_file.read_text()
    destination = immutable_artifact(args.manifest.parent, f"{tile['name']}-{args.stage}", args.output)
    tile[key] = destination.name
    if args.stage == "structure":
        tile.update(structural_status="accepted", structure_prompt=prompt,
                    structure_review_note=args.review_note.strip(), structure_output_sha256=sha256(destination),
                    structure_output_size=[width, height], structure_aspect_error=error,
                    semantic_status="pending", semantic_prompt=None, semantic_review_note=None, output_sha256=None)
    else:
        tile.update(semantic_status="accepted", semantic_prompt=prompt,
                    semantic_review_note=args.review_note.strip(), output_sha256=sha256(destination),
                    output_size=[width, height], aspect_error=error)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tile")
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--stage", choices=("structure", "semantic"), required=True)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--max-aspect-error", type=float, default=0.002)
    args = parser.parse_args()
    if not math.isfinite(args.max_aspect_error) or not 0 <= args.max_aspect_error <= 0.002:
        parser.error("Invalid aspect tolerance")
    if not args.review_note.strip() or not args.prompt_file.read_text().strip():
        parser.error("Prompt and native-pixel review note are required")
    try:
        with locked_manifest(args.manifest) as data:
            destination = record(data, args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(destination)


if __name__ == "__main__":
    main()
