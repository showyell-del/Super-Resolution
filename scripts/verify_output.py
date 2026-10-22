#!/usr/bin/env python3
"""Verify delivery metadata and export native-pixel review crops."""

import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

from apple_preflight import require_apple_silicon
from pipeline_contract import local_file, new_task_directory, atomic_json, box


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("regions", type=Path, help="JSON list of name,x,y,width,height objects")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    require_apple_silicon([args.image, args.output_dir.parent])
    image = Image.open(args.image)
    if image.size != (args.width, args.height):
        parser.error(f"Expected {(args.width, args.height)}, got {image.size}")
    regions = json.loads(args.regions.read_text(encoding="utf-8"))
    if not isinstance(regions, list) or not regions:
        parser.error("Regions file must contain a non-empty JSON list")
    try:
        names = set()
        for region in regions:
            local_file(args.output_dir, region["name"])
            if region["name"] in names:
                raise ValueError("Review region names must be unique")
            names.add(region["name"])
            box([region[key] for key in ("x", "y", "width", "height")], image.size)
        new_task_directory(args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    crops = []
    for region in regions:
        name = region["name"]
        x, y, width, height = (int(region[key]) for key in ("x", "y", "width", "height"))
        if width < 1 or height < 1 or x < 0 or y < 0 or x + width > image.width or y + height > image.height:
            parser.error(f"Invalid crop region {name!r}")
        path = local_file(args.output_dir, f"{name}.png")
        image.crop((x, y, x + width, y + height)).save(path, compress_level=2)
        crops.append({"name": name, "box": [x, y, width, height], "path": str(path)})
    report = {
        "image": str(args.image.resolve()),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "bytes": args.image.stat().st_size,
        "sha256": file_hash(args.image),
        "icc_profile": bool(image.info.get("icc_profile")),
        "crops": crops,
    }
    report_path = args.output_dir / "verification.json"
    atomic_json(report_path, report)
    print(report_path)


if __name__ == "__main__":
    main()
