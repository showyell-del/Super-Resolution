#!/usr/bin/env python3
"""Apple-Silicon Real-ESRGAN runner using PyTorch MPS with no CPU fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from PIL import Image

from apple_preflight import require_apple_silicon
from realesrgan_mps import RealESRGANMPS

EXPECTED_WEIGHTS_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    skill_root = Path(__file__).resolve().parent.parent
    default_runtime = skill_root / "runtime"
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-pad", type=int, default=24)
    parser.add_argument("--runtime", type=Path, default=default_runtime)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()

    weights = args.weights or args.runtime / "weights" / "RealESRGAN_x4plus.pth"
    require_apple_silicon(
        [args.source, args.approval, args.output.parent, args.runtime, weights],
        check_mps=True,
    )
    if not args.source.is_file():
        parser.error(f"Missing source image: {args.source}")
    if not args.approval.is_file():
        parser.error(f"Missing semantic-master approval: {args.approval}")
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    if approval.get("approved") is not True:
        parser.error("Semantic master is not approved")
    if Path(approval.get("master", "")).resolve() != args.source.resolve():
        parser.error("Approval belongs to a different semantic master")
    source_hash = sha256(args.source)
    if approval.get("master_sha256") != source_hash:
        parser.error("Semantic master changed after native-pixel approval")
    if not weights.is_file():
        parser.error(f"Missing verified Real-ESRGAN model: {weights}")
    weights_hash = sha256(weights)
    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        parser.error(
            f"Unexpected Real-ESRGAN model hash {weights_hash}; expected {EXPECTED_WEIGHTS_SHA256}"
        )
    if args.tile < 64 or args.tile_pad < 0:
        parser.error("Tile must be at least 64 pixels and tile padding must be non-negative")

    import cv2
    import numpy as np
    import torch

    if not torch.backends.mps.is_available():
        parser.error("Apple MPS is unavailable; CPU fallback is forbidden")
    source_original = Image.open(args.source)
    source_width, source_height = source_original.size
    scale_x = args.target_width / source_width
    scale_y = args.target_height / source_height
    if abs(scale_x - scale_y) > 1e-8:
        parser.error("Target dimensions must preserve the source aspect ratio exactly")
    if scale_x <= 1:
        parser.error("Target dimensions must be larger than the source")

    input_rgb = np.asarray(source_original.convert("RGB"), dtype=np.uint8)
    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    upsampler = RealESRGANMPS(
        model_path=weights,
        tile=args.tile,
        tile_pad=args.tile_pad,
    )
    started = time.time()
    output_bgr = upsampler.enhance(input_bgr, outscale=scale_x)
    if (output_bgr.shape[1], output_bgr.shape[0]) != (args.target_width, args.target_height):
        parser.error(
            f"Model produced {(output_bgr.shape[1], output_bgr.shape[0])}, expected "
            f"{(args.target_width, args.target_height)}"
        )
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_rgb)
    if "A" in source_original.mode:
        alpha = source_original.getchannel("A").resize(
            (args.target_width, args.target_height),
            Image.Resampling.LANCZOS,
        )
        output_image.putalpha(alpha)
    if source_original.mode in {"L", "LA"}:
        output_image = output_image.convert("L" if source_original.mode == "L" else "LA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_args = {"compress_level": 2}
    if source_original.info.get("icc_profile"):
        save_args["icc_profile"] = source_original.info["icc_profile"]
    output_image.save(args.output, **save_args)
    report = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "approval": str(args.approval.resolve()),
        "approval_sha256": sha256(args.approval),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "weights": str(weights.resolve()),
        "weights_sha256": weights_hash,
        "runtime_implementation": "scripts/realesrgan_mps.py",
        "device": "mps",
        "mps_fallback": False,
        "torch": torch.__version__,
        "model_native_scale": 4,
        "delivery_scale": scale_x,
        "tile": args.tile,
        "tile_pad": args.tile_pad,
        "target": [args.target_width, args.target_height],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".mps-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
