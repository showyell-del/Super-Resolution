#!/usr/bin/env python3
"""Hard preflight for Apple-Silicon semantic reconstruction jobs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


MIN_FREE_GIB = 50
GIB = 1024 ** 3


class PreflightError(RuntimeError):
    pass


def cpu_brand() -> str:
    try:
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError("Unable to read the Apple CPU identity") from exc


def existing_anchor(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise PreflightError(f"Cannot resolve a storage volume for {path}")
    return candidate


def volume_report(paths: list[Path], minimum_gib: int = MIN_FREE_GIB) -> list[dict]:
    reports: list[dict] = []
    seen: set[int] = set()
    for requested in paths:
        anchor = existing_anchor(requested)
        device = os.stat(anchor).st_dev
        if device in seen:
            continue
        seen.add(device)
        usage = shutil.disk_usage(anchor)
        item = {
            "path": str(requested),
            "anchor": str(anchor),
            "free_bytes": usage.free,
            "free_gib": round(usage.free / GIB, 2),
            "required_gib_strictly_greater_than": minimum_gib,
        }
        reports.append(item)
        if usage.free <= minimum_gib * GIB:
            raise PreflightError(
                f"Storage preflight failed for {anchor}: {item['free_gib']} GiB free; "
                f"the workflow requires strictly more than {minimum_gib} GiB"
            )
    return reports


def require_apple_silicon(paths: list[Path], check_mps: bool = False) -> dict:
    if platform.system() != "Darwin":
        raise PreflightError("Super-Resolution requires macOS")
    if platform.machine() != "arm64":
        raise PreflightError("Super-Resolution requires an Apple-Silicon arm64 Mac")
    brand = cpu_brand()
    if "Apple M" not in brand:
        raise PreflightError(f"An Apple M-series processor is required; detected {brand!r}")

    report = {
        "system": platform.system(),
        "architecture": platform.machine(),
        "cpu": brand,
        "volumes": volume_report(paths),
    }

    if check_mps:
        try:
            import torch
        except ImportError as exc:
            raise PreflightError("PyTorch is required for Apple MPS inference") from exc
        if not torch.backends.mps.is_built():
            raise PreflightError("Installed PyTorch was not built with Apple MPS support")
        if not torch.backends.mps.is_available():
            raise PreflightError("Apple MPS is not available on this Mac")
        report["torch"] = torch.__version__
        report["accelerator"] = "mps"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check-mps", action="store_true")
    args = parser.parse_args()
    try:
        report = require_apple_silicon(args.paths, check_mps=args.check_mps)
    except PreflightError as exc:
        parser.exit(2, f"PRECHECK_BLOCKED: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
