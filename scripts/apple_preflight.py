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


def volume_report(paths: list[Path]) -> list[dict]:
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
        }
        reports.append(item)
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
        # macOS swap shares the data volume even when all image files are external.
        "volumes": volume_report([Path("/private/var/vm"), *paths]),
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
    parser.add_argument("--runtime", type=Path, help="Real-ESRGAN runtime directory")
    parser.add_argument(
        "--include-codex-imagegen",
        action="store_true",
        help="also check CODEX_HOME/generated_images",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    paths = list(args.paths)
    if args.runtime:
        paths.extend([args.runtime, args.runtime / "weights"])
    if args.include_codex_imagegen:
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        paths.append(codex_home / "generated_images")
    try:
        report = require_apple_silicon(paths, check_mps=args.check_mps)
    except PreflightError as exc:
        parser.exit(2, f"PRECHECK_BLOCKED: {exc}\n")
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
