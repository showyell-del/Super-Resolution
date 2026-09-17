#!/usr/bin/env python3
"""Restore exact source pixels inside a protection mask after generative editing."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from apple_preflight import require_apple_silicon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("processed", type=Path)
    parser.add_argument("mask", type=Path, help="White preserves source; black keeps processed")
    parser.add_argument("output", type=Path)
    parser.add_argument("--feather", type=float, default=0.0)
    args = parser.parse_args()

    require_apple_silicon([args.source, args.processed, args.mask, args.output.parent])
    source = Image.open(args.source).convert("RGBA")
    processed_original = Image.open(args.processed)
    processed = processed_original.convert("RGBA")
    if source.size != processed.size:
        parser.error("Source and processed images must have identical dimensions")
    mask = Image.open(args.mask).convert("L")
    if mask.size != source.size:
        parser.error("Protection mask dimensions must match the images")
    if args.feather < 0:
        parser.error("Feather radius must be non-negative")
    if args.feather:
        mask = mask.filter(ImageFilter.GaussianBlur(args.feather))

    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    source_array = np.asarray(source, dtype=np.float32)
    processed_array = np.asarray(processed, dtype=np.float32)
    result = np.clip(source_array * alpha + processed_array * (1.0 - alpha), 0, 255).astype(np.uint8)
    output = Image.fromarray(result)
    icc = processed_original.info.get("icc_profile") or Image.open(args.source).info.get("icc_profile")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_args = {"compress_level": 2}
    if icc:
        save_args["icc_profile"] = icc
    output.save(args.output, **save_args)
    print(args.output)


if __name__ == "__main__":
    main()
