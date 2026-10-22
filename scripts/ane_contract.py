"""ANE conversion identity, crop evidence and owned scratch lifecycle."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil

from pipeline_contract import sha256, box, load_approval

DIT_SHA256 = "bdcaa81e4c675b6074de643e27daafe73eb8125d4d0264d1bf30a004e9644b71"
VAE_SHA256 = "1dd2c67f3f29734d48709513ed181d4f351966d76ba95f77a5cff8e1daf547fa"
PACKAGE_FILES = ("Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin")
PACKAGE_COMPONENTS = {"dit-segment-0-ane.mlpackage": ("dit0", DIT_SHA256),
                      "dit-segment-1-ane.mlpackage": ("dit1", DIT_SHA256),
                      "qwen-vae-decoder-128-ane.mlpackage": ("decoder", VAE_SHA256)}


def conversion_evidence(packages: Path):
    evidence = {}
    for name, (component, weight_hash) in PACKAGE_COMPONENTS.items():
        path = packages / f"{component}-conversion.json"
        report = json.loads(path.read_text())
        actual = {relative: sha256(packages / name / relative) for relative in PACKAGE_FILES}
        if (report.get("component") != component or report.get("weight_sha256") != weight_hash
                or report.get("package_files") != actual):
            raise ValueError(f"ANE conversion provenance does not match pinned package {name}")
        evidence[name] = {"report_sha256": sha256(path), "weight_sha256": weight_hash, "files": actual}
    return evidence


@contextmanager
def scratch_workspace(path: Path):
    """Delete only the new task-owned directory, including on failure."""
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def validate_ane_trial(path: Path, source: Path, approval_path: Path, plan_path: Path,
                       seed: int, align: str, conversion):
    from PIL import Image
    from finalize_delivery import validate_native_review
    from pipeline_contract import PEOPLE_CHECKS

    review = json.loads(path.read_text())
    image = Path(review["image"])
    with Image.open(image) as pixels:
        validate_native_review(review, sha256(image), pixels.size)
        size = list(pixels.size)
    report = json.loads(image.with_name(f"{image.stem}-runtime-report.json").read_text())
    expected = {"source_sha256": sha256(source), "approval_sha256": sha256(approval_path),
                "output_sha256": sha256(image), "output_size": size, "run_kind": "diagnostic",
                "seed": seed, "align": align, "package_plan_sha256": sha256(plan_path),
                "conversion_sha256": conversion}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("ANE trial source/model/parameters/output evidence changed")
    approval = load_approval(approval_path, source)
    with Image.open(source) as original:
        trial_box = box(report.get("trial_box"), original.size)
    if size != [trial_box[2] * 4, trial_box[3] * 4]:
        raise ValueError("ANE crop must be native 4x")
    if approval["review"].get("subject_report_required"):
        checks = review.get("people_review", {}).get("checks", {})
        if set(checks) != PEOPLE_CHECKS or any(v != "pass" for v in checks.values()):
            raise ValueError("ANE photographic trial requires passing people checks")
    return review
