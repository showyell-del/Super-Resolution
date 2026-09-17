#!/usr/bin/env python3
"""Register, validate, and feather edited tiles into a semantic master."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from apple_preflight import require_apple_silicon


def smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def overlap(first: dict, second: dict, axis: str) -> int:
    size = "width" if axis == "x" else "height"
    return max(0, first[axis] + first[size] - second[axis])


def rgb_proxy(source: Image.Image) -> Image.Image:
    if source.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", source.size, (255, 255, 255, 255))
        background.alpha_composite(source.convert("RGBA"))
        return background.convert("RGB")
    return source.convert("RGB")


def register_tile(reference: np.ndarray, edited: np.ndarray, cv2, max_side: int = 1024):
    height, width = reference.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    small_size = (max(32, round(width * scale)), max(32, round(height * scale)))
    ref_small = cv2.resize(reference, small_size, interpolation=cv2.INTER_AREA)
    edit_small = cv2.resize(edited, small_size, interpolation=cv2.INTER_AREA)
    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edit_gray = cv2.cvtColor(edit_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    correlation, warp = cv2.findTransformECC(
        ref_gray,
        edit_gray,
        warp,
        cv2.MOTION_AFFINE,
        criteria,
        None,
        5,
    )
    warp[0, 2] /= scale
    warp[1, 2] /= scale
    aligned = cv2.warpAffine(
        edited,
        warp,
        (width, height),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, float(correlation), warp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--min-correlation", type=float, default=0.60)
    parser.add_argument("--max-translation-ratio", type=float, default=0.03)
    parser.add_argument("--max-scale-error", type=float, default=0.04)
    args = parser.parse_args()

    scratch_root = args.scratch_dir or args.output.parent
    require_apple_silicon([args.manifest, args.output.parent, scratch_root])
    try:
        import cv2
    except ImportError as exc:
        parser.error(f"opencv-python is required for mandatory tile registration: {exc}")

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.parent
    source_path = Path(data["source"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != data["source_sha256"]:
        parser.error("Source image changed after tile preparation")
    source = Image.open(source_path)
    source_rgb = rgb_proxy(source)
    width, height = data["width"], data["height"]
    by_position = {(tile["row"], tile["col"]): tile for tile in data["tiles"]}
    reports = []

    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semantic-stitch-", dir=scratch_root) as temporary:
        temporary = Path(temporary)
        accum = np.memmap(temporary / "accum.f32", dtype=np.float32, mode="w+", shape=(height, width, 3))
        weights = np.memmap(temporary / "weights.f32", dtype=np.float32, mode="w+", shape=(height, width, 1))
        accum[:] = 0
        weights[:] = 0

        for tile in data["tiles"]:
            if tile.get("structural_status") != "accepted":
                parser.error(f"Tile {tile['name']} has no accepted structural pass")
            if tile.get("semantic_status") != "accepted":
                parser.error(f"Tile {tile['name']} has no accepted semantic pass")
            path = base / tile["output"]
            if not path.is_file():
                parser.error(f"Missing edited tile: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != tile.get("output_sha256"):
                parser.error(f"Tile {tile['name']} changed after it was recorded")

            x, y, tw, th = tile["x"], tile["y"], tile["width"], tile["height"]
            edited = Image.open(path).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            edited_array = np.asarray(edited, dtype=np.uint8)
            reference_array = np.asarray(source_rgb.crop((x, y, x + tw, y + th)), dtype=np.uint8)
            try:
                aligned, correlation, warp = register_tile(reference_array, edited_array, cv2)
            except cv2.error as exc:
                parser.error(f"Registration failed for {tile['name']}: {exc}")

            linear = warp[:, :2]
            singular_values = np.linalg.svd(linear, compute_uv=False)
            scale_error = float(np.max(np.abs(singular_values - 1.0)))
            translation_ratio = max(abs(float(warp[0, 2])) / tw, abs(float(warp[1, 2])) / th)
            if correlation < args.min_correlation:
                parser.error(f"Tile {tile['name']} correlation {correlation:.3f} is below the acceptance threshold")
            if scale_error > args.max_scale_error:
                parser.error(f"Tile {tile['name']} geometry scale drift {scale_error:.3%} exceeds the threshold")
            if translation_ratio > args.max_translation_ratio:
                parser.error(f"Tile {tile['name']} translation drift {translation_ratio:.3%} exceeds the threshold")

            wx = np.ones(tw, dtype=np.float32)
            wy = np.ones(th, dtype=np.float32)
            row, col = tile["row"], tile["col"]
            if col > 1:
                amount = overlap(by_position[(row, col - 1)], tile, "x")
                if amount > 0:
                    wx[:amount] = smoothstep(np.linspace(0, 1, amount, dtype=np.float32))
            if col < data["cols"]:
                amount = overlap(tile, by_position[(row, col + 1)], "x")
                if amount > 0:
                    wx[-amount:] = smoothstep(np.linspace(1, 0, amount, dtype=np.float32))
            if row > 1:
                amount = overlap(by_position[(row - 1, col)], tile, "y")
                if amount > 0:
                    wy[:amount] = smoothstep(np.linspace(0, 1, amount, dtype=np.float32))
            if row < data["rows"]:
                amount = overlap(tile, by_position[(row + 1, col)], "y")
                if amount > 0:
                    wy[-amount:] = smoothstep(np.linspace(1, 0, amount, dtype=np.float32))

            weight = (wy[:, None] * wx[None, :])[..., None]
            accum[y:y + th, x:x + tw] += aligned.astype(np.float32) * weight
            weights[y:y + th, x:x + tw] += weight
            reports.append({
                "tile": tile["name"],
                "correlation": correlation,
                "scale_error": scale_error,
                "translation_ratio": translation_ratio,
                "warp": warp.tolist(),
            })

        if np.any(weights == 0):
            parser.error("Tile manifest leaves uncovered pixels")
        result = np.memmap(temporary / "result.u8", dtype=np.uint8, mode="w+", shape=(height, width, 3))
        for start in range(0, height, 256):
            end = min(height, start + 256)
            result[start:end] = np.clip(accum[start:end] / weights[start:end], 0, 255).astype(np.uint8)

        output_image = Image.fromarray(np.asarray(result))
        if data["source_mode"] in {"L", "LA"}:
            output_image = output_image.convert("L")
        if data.get("source_alpha"):
            alpha = Image.open(base / data["source_alpha"]).convert("L")
            output_image.putalpha(alpha)
        icc_profile = (base / data["source_icc"]).read_bytes() if data.get("source_icc") else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_args = {"compress_level": 2}
        if icc_profile:
            save_args["icc_profile"] = icc_profile
        output_image.save(args.output, **save_args)

    report_path = args.output.with_suffix(args.output.suffix + ".stitch-report.json")
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
