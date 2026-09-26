#!/usr/bin/env python3
"""Extract sparse high-risk regions for localized semantic reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from apple_preflight import require_apple_silicon


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("regions", type=Path, help="JSON list containing name,x,y,width,height")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--context", type=int, default=64)
    args = parser.parse_args()

    require_apple_silicon([args.image, args.regions, args.output_dir.parent])
    if args.context < 0:
        parser.error("--context must be non-negative")
    image = Image.open(args.image).convert("RGB")
    definitions = json.loads(args.regions.read_text(encoding="utf-8"))
    if not isinstance(definitions, list) or not definitions:
        parser.error("Regions must be a non-empty JSON list")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    names: set[str] = set()
    prepared = []
    for index, item in enumerate(definitions):
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            parser.error(f"Region {index} needs a unique non-empty name")
        names.add(name)
        try:
            x, y, width, height = (int(item[key]) for key in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"Region {name!r} needs integer x,y,width,height: {exc}")
        if x < 0 or y < 0 or width < 1 or height < 1 or x + width > image.width or y + height > image.height:
            parser.error(f"Region {name!r} exceeds the image")
        left = max(0, x - args.context)
        top = max(0, y - args.context)
        right = min(image.width, x + width + args.context)
        bottom = min(image.height, y + height + args.context)
        filename = f"{name}-input.png"
        path = args.output_dir / filename
        image.crop((left, top, right, bottom)).save(path, compress_level=2)
        prepared.append({
            "name": name,
            "category": item.get("category", "semantic"),
            "target_box": [x, y, width, height],
            "crop_box": [left, top, right - left, bottom - top],
            "input": filename,
            "input_sha256": sha256(path),
            "output": f"{name}-output.png",
            "status": "pending",
            "prompt": None,
            "review_note": None,
            "output_sha256": None,
        })

    manifest = {
        "source": str(args.image.resolve()),
        "source_sha256": sha256(args.image),
        "source_size": [image.width, image.height],
        "context": args.context,
        "regions": prepared,
    }
    path = args.output_dir / "regions-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
