#!/usr/bin/env python3
"""Register and feather accepted sparse semantic regions onto a master image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from apple_preflight import require_apple_silicon


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(reference: np.ndarray, edited: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    correlation, warp = cv2.findTransformECC(ref_gray, edit_gray, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    aligned = cv2.warpAffine(
        edited,
        warp,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, float(correlation), warp


def feather_mask(width: int, height: int, feather: int) -> np.ndarray:
    if feather <= 0:
        return np.ones((height, width, 1), dtype=np.float32)
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    edge = np.minimum(y[:, None], x[None, :])
    return np.clip(edge / feather, 0, 1)[..., None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--feather", type=int, default=32)
    parser.add_argument("--min-correlation", type=float, default=0.45)
    parser.add_argument("--max-translation-ratio", type=float, default=0.04)
    parser.add_argument("--max-scale-error", type=float, default=0.05)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_path = Path(data["source"])
    require_apple_silicon([source_path, args.manifest, args.output.parent])
    if sha256(source_path) != data["source_sha256"]:
        parser.error("Source changed after region preparation")
    source_image = Image.open(source_path)
    source = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
    result = source.astype(np.float32)
    reports = []

    for item in data["regions"]:
        if item.get("status") != "accepted":
            parser.error(f"Region {item['name']} is not accepted")
        edit_path = args.manifest.parent / item["output"]
        if not edit_path.is_file() or sha256(edit_path) != item.get("output_sha256"):
            parser.error(f"Region {item['name']} output is missing or changed")
        x, y, width, height = item["crop_box"]
        edited = np.asarray(Image.open(edit_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
        reference = source[y:y + height, x:x + width]
        try:
            aligned, correlation, warp = register(reference, edited)
        except cv2.error as exc:
            parser.error(f"Registration failed for {item['name']}: {exc}")
        singular_values = np.linalg.svd(warp[:, :2], compute_uv=False)
        scale_error = float(np.max(np.abs(singular_values - 1)))
        translation_ratio = max(abs(float(warp[0, 2])) / width, abs(float(warp[1, 2])) / height)
        if correlation < args.min_correlation or scale_error > args.max_scale_error or translation_ratio > args.max_translation_ratio:
            parser.error(
                f"Region {item['name']} registration rejected: correlation={correlation:.3f}, "
                f"scale_error={scale_error:.3%}, translation={translation_ratio:.3%}"
            )
        alpha = feather_mask(width, height, min(args.feather, width // 4, height // 4))
        base = result[y:y + height, x:x + width]
        result[y:y + height, x:x + width] = base * (1 - alpha) + aligned.astype(np.float32) * alpha
        reports.append({
            "name": item["name"],
            "crop_box": item["crop_box"],
            "output_sha256": item["output_sha256"],
            "correlation": correlation,
            "scale_error": scale_error,
            "translation_ratio": translation_ratio,
            "warp": warp.tolist(),
        })

    output_image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_args = {"compress_level": 2}
    if source_image.info.get("icc_profile"):
        save_args["icc_profile"] = source_image.info["icc_profile"]
    output_image.save(args.output, **save_args)
    report = {
        "source": str(source_path.resolve()),
        "source_sha256": data["source_sha256"],
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "feather": args.feather,
        "regions": reports,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".composite-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
