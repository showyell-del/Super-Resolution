#!/usr/bin/env python3
"""Copy the verified Real-ESRGAN runtime weight to a storage-safe volume."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

from apple_preflight import require_apple_silicon
from pipeline_contract import atomic_json


EXPECTED_WEIGHTS_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def copy_verified(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected:
            raise ValueError("Copied model hash mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    skill_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path, help="Runtime directory on the chosen work volume")
    parser.add_argument(
        "--source-weights",
        type=Path,
        default=skill_root / "runtime" / "weights" / "RealESRGAN_x4plus.pth",
    )
    args = parser.parse_args()

    require_apple_silicon([args.destination.parent])
    if not args.source_weights.is_file():
        parser.error(f"Missing source model: {args.source_weights}")
    source_hash = sha256(args.source_weights)
    if source_hash != EXPECTED_WEIGHTS_SHA256:
        parser.error(f"Unexpected source model hash: {source_hash}")

    destination = args.destination / "weights" / "RealESRGAN_x4plus.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    reused = destination.is_file() and sha256(destination) == EXPECTED_WEIGHTS_SHA256
    if not reused:
        copy_verified(args.source_weights, destination, EXPECTED_WEIGHTS_SHA256)

    report = {
        "runtime": str(args.destination.resolve()),
        "weights": str(destination.resolve()),
        "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "reused_verified_weights": reused,
    }
    report_path = args.destination / "runtime-stage-report.json"
    atomic_json(report_path, report)
    print(args.destination)
    print(report_path)


if __name__ == "__main__":
    main()
