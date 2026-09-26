#!/usr/bin/env python3
"""Center-crop an image to an exact rational target aspect without stretching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from apple_preflight import require_apple_silicon


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--max-crop-ratio", type=float, default=0.05)
    args = parser.parse_args()

    require_apple_silicon([args.source, args.output.parent])
    if args.target_width < 1 or args.target_height < 1:
        parser.error("Target dimensions must be positive")
    if not 0 <= args.max_crop_ratio < 1:
        parser.error("--max-crop-ratio must be in [0,1)")

    divisor = math.gcd(args.target_width, args.target_height)
    ratio_width = args.target_width // divisor
    ratio_height = args.target_height // divisor
    source = Image.open(args.source)
    width, height = source.size
    scale = min(width // ratio_width, height // ratio_height)
    if scale < 1:
        parser.error("Source is smaller than one exact target-ratio unit")
    crop_width = ratio_width * scale
    crop_height = ratio_height * scale
    removed_ratio = 1 - (crop_width * crop_height) / (width * height)
    if removed_ratio > args.max_crop_ratio:
        parser.error(
            f"Exact aspect crop would remove {removed_ratio:.2%}; limit is {args.max_crop_ratio:.2%}"
        )
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    box = (left, top, left + crop_width, top + crop_height)
    output_image = source.crop(box)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_args = {"compress_level": 2}
    if source.info.get("icc_profile"):
        save_args["icc_profile"] = source.info["icc_profile"]
    output_image.save(args.output, **save_args)

    report = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "source_size": [width, height],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "output_size": [crop_width, crop_height],
        "target_aspect": [ratio_width, ratio_height],
        "crop_box": [left, top, crop_width, crop_height],
        "removed_area_ratio": removed_ratio,
        "resampled": False,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".canvas-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
