#!/usr/bin/env python3
"""Register native-density repairs with disk-backed, exact float accumulation."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from apple_preflight import require_apple_silicon
from pipeline_contract import (validate_resolution, sha256, local_file, new_output, verify_repairs, load_approval,
                               verify_neural_report, preserve_mode, atomic_json)


def register(reference, edited):
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    correlation, warp = cv2.findTransformECC(ref_gray, edit_gray, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    aligned = cv2.warpAffine(edited, warp, (reference.shape[1], reference.shape[0]),
                            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
    return aligned, float(correlation), warp


def feather_mask(width, height, feather):
    if feather <= 0:
        return np.ones((height, width, 1), dtype=np.float32)
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    return np.clip(np.minimum(y[:, None], x[None, :]) / feather, 0, 1)[..., None]


def blend_target(result, aligned, crop_box, target_box, feather):
    """Keep fractional overlap until final export, matching the original formula."""
    x, y, _, _ = crop_box
    tx, ty, tw, th = target_box
    alpha = feather_mask(tw, th, min(feather, tw // 4, th // 4))
    old = result[ty:ty + th, tx:tx + tw]
    edited = aligned[ty-y:ty-y+th, tx-x:tx-x+tw].astype(np.float32)
    result[ty:ty + th, tx:tx + tw] = old * (1 - alpha) + edited * alpha


def compose(data, args, base_image):
    width, height = base_image.size
    scale = width / data["source_size"][0]
    base_pixels = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
    reports = []
    with tempfile.TemporaryDirectory(prefix="region-composite-", dir=args.output.parent) as temporary:
        result = np.memmap(Path(temporary) / "canvas.f32", mode="w+", dtype=np.float32, shape=base_pixels.shape)
        output = np.memmap(Path(temporary) / "canvas.u8", mode="w+", dtype=np.uint8, shape=base_pixels.shape)
        try:
            for row in range(0, height, 128):
                result[row:row+128] = base_pixels[row:row+128]
            for item in data["regions"]:
                sx, sy, sw, sh = item["crop_box"]
                x, y = round(sx * scale), round(sy * scale)
                right, bottom = round((sx + sw) * scale), round((sy + sh) * scale)
                cw, ch = right-x, bottom-y
                with Image.open(local_file(args.manifest.parent, item["output"])) as generated:
                    edited = np.asarray(generated.convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS))
                aligned, correlation, warp = register(base_pixels[y:bottom, x:right], edited)
                drift = float(np.max(np.abs(np.linalg.svd(warp[:, :2], compute_uv=False) - 1)))
                translation = max(abs(float(warp[0, 2])) / cw, abs(float(warp[1, 2])) / ch)
                if correlation < args.min_correlation or drift > args.max_scale_error or translation > args.max_translation_ratio:
                    raise ValueError(f"Region {item['name']} registration rejected")
                tx, ty, tw, th = item["target_box"]
                target = [round(tx * scale), round(ty * scale),
                          round((tx + tw) * scale) - round(tx * scale),
                          round((ty + th) * scale) - round(ty * scale)]
                blend_target(result, aligned, [x, y, cw, ch], target, args.feather)
                reports.append({"name": item["name"], "subject_ids": item.get("subject_ids", []),
                                "source_crop_box": item["crop_box"], "source_target_box": item["target_box"],
                                "composite_box": [x, y, cw, ch], "composite_target_box": target,
                                "output_sha256": item["output_sha256"], "correlation": correlation,
                                "scale_error": drift, "translation_ratio": translation, "warp": warp.tolist()})
            for row in range(0, height, 128):
                output[row:row+128] = np.clip(result[row:row+128], 0, 255).astype(np.uint8)
            image = preserve_mode(Image.fromarray(np.asarray(output)), base_image)
            options = {"compress_level": 2}
            if base_image.info.get("icc_profile"):
                options["icc_profile"] = base_image.info["icc_profile"]
            image.save(args.output, **options)
        finally:
            result._mmap.close()
            output._mmap.close()
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--base-report", type=Path)
    parser.add_argument("--feather", type=int, default=32)
    parser.add_argument("--min-correlation", type=float, default=0.45)
    parser.add_argument("--max-translation-ratio", type=float, default=0.04)
    parser.add_argument("--max-scale-error", type=float, default=0.05)
    args = parser.parse_args()
    if args.feather < 0 or any(not math.isfinite(v) for v in (
            args.min_correlation, args.max_translation_ratio, args.max_scale_error)):
        parser.error("Invalid registration/feather settings")
    try:
        new_output(args.output)
        report_path = args.output.with_suffix(args.output.suffix + ".composite-report.json")
        new_output(report_path)
        data = json.loads(args.manifest.read_text())
        source = Path(data["source"])
        base = args.base or source
        require_apple_silicon([source, base, args.output.parent])
        base_hash = sha256(base)
        with Image.open(base) as image:
            if image.mode not in {"RGB", "RGBA", "L", "LA"}:
                raise ValueError("Unsupported base mode; use an explicit color-managed conversion")
            verify_repairs(args.manifest, source, target_size=image.size)
            if args.base:
                if not args.approval or not args.base_report:
                    raise ValueError("Enlarged base requires --approval and --base-report")
                approval = load_approval(args.approval, source)
                validate_resolution(approval, image.size)
                bound = approval.get("repair_manifest") or {}
                if bound.get("sha256") != sha256(args.manifest) or Path(bound["path"]).resolve() != args.manifest.resolve():
                    raise ValueError("Approval does not bind this repair manifest")
                verify_neural_report(json.loads(args.base_report.read_text()), args.approval, approval, base_hash)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            reports = compose(data, args, image)
        report = {"operation": "registered_composite", "source": str(source.resolve()),
                  "source_sha256": data["source_sha256"], "base": str(base.resolve()), "base_sha256": base_hash,
                  "base_report_sha256": sha256(args.base_report) if args.base_report else None,
                  "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
                  "output": str(args.output.resolve()), "output_sha256": sha256(args.output),
                  "feather": args.feather, "regions": reports}
        atomic_json(report_path, report)
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        parser.error(str(error))
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
