"""Runtime contracts tested without loading or converting neural models."""

import ast
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class RuntimeTests(unittest.TestCase):
    def test_photo_12k_cli_rejected_before_models_or_scratch(self):
        from PIL import Image
        from pipeline_contract import sha256
        with tempfile.TemporaryDirectory(dir=os.environ.get("SR_TEST_ROOT", "/Volumes/work/Ai/Super-Resolution-tests")) as raw:
            directory = Path(raw)
            source, approval = directory / "source.png", directory / "approval.json"
            Image.new("RGB", (3072, 8)).save(source)
            digest = sha256(source)
            approval.write_text(json.dumps({"approval_version": 4, "approved": True,
                "master": str(source), "master_sha256": digest,
                "review": {"image_sha256": digest, "visual_contract": {"target_look": "camera_photo"}}}))
            output, scratch = directory / "output.png", directory / "scratch"
            entries = [
                ["apple_mps_upscale.py", str(source), str(output), "--approval", str(approval), "--preset", "12k"],
                ["apple_mps_upscale.py", str(source), str(output), "--approval", str(approval),
                 "--target-width", "12288", "--target-height", "32"],
                ["vosr2_ane_hybrid.py", "--source", str(source), "--output", str(output),
                 "--approval", str(approval), "--scratch", str(scratch),
                 "--trial-review", "unused.json", "--verified-package-plan", "unused.json"]]
            for entry in entries:
                with self.subTest(entry=entry[0], options=entry[1:]):
                    result = subprocess.run([sys.executable, str(ROOT / "scripts" / entry[0])] + entry[1:],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("6144", result.stderr)
                    self.assertFalse(output.exists())
                    self.assertFalse(scratch.exists())

    def test_style_resolution_limits(self):
        from pipeline_contract import validate_resolution
        for look in ("camera_photo", "render", "illustration", "graphic", "mixed"):
            approval = {"review": {"visual_contract": {"target_look": look}}}
            validate_resolution(approval, (6144, 4096))
            validate_resolution(approval, (4096, 6144))
            if look == "camera_photo":
                for size in ((6145, 4096), (4096, 6145), (12288, 8192)):
                    with self.assertRaisesRegex(ValueError, "6144"):
                        validate_resolution(approval, size)
            else:
                validate_resolution(approval, (12288, 8192))
                validate_resolution(approval, (8192, 12288))
                with self.assertRaisesRegex(ValueError, "12288"):
                    validate_resolution(approval, (12289, 8192))

    def test_unknown_style_does_not_gain_12k_permission(self):
        from pipeline_contract import validate_resolution
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_resolution({"review": {"visual_contract": {"target_look": "unknown"}}},
                                (6144, 4096))

    def test_photographic_final_gate_rejects_oversize_pixels(self):
        from PIL import Image
        from finalize_delivery import validate_final_requirements
        with tempfile.TemporaryDirectory(dir=os.environ.get("SR_TEST_ROOT", "/Volumes/work/Ai/Super-Resolution-tests")) as raw:
            source, final = Path(raw) / "source.png", Path(raw) / "final.png"
            Image.new("RGB", (64, 64)).save(source)
            Image.new("RGB", (6145, 64)).save(final)
            approval = {"master": str(source), "review": {"visual_contract": {"target_look": "camera_photo"}}}
            with self.assertRaisesRegex(ValueError, "6144"):
                validate_final_requirements({}, approval, final, None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("SR_TEST_ROOT", "/Volumes/work/Ai/Super-Resolution-tests"))
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_decoder_conversion_does_not_load_model(self):
        tree = ast.parse((ROOT / "scripts/convert_vosr2_ane.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "convert_decoder")
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "convert"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(k.arg == "skip_model_load" and isinstance(k.value, ast.Constant)
                            and k.value.value is True for k in calls[0].keywords))

    def test_staging_partial_copy_cleanup(self):
        module = importlib.import_module("stage_runtime")
        source = self.directory / "weight.pth"
        source.write_bytes(b"fixture")
        target = self.directory / "weights/model.pth"
        def fail_copy(src, dest):
            Path(dest).write_bytes(b"partial")
            raise OSError("disk full fixture")
        with patch.object(shutil, "copyfile", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "disk full"):
                module.copy_verified(source, target, "fixture-hash")
        self.assertFalse(target.exists())
        self.assertFalse(list(target.parent.glob("*.tmp")))

    def test_overlap_blending_retains_exact_fractional_values(self):
        from composite_regions import blend_target, feather_mask
        rng = np.random.default_rng(4)
        base = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        expected = base.astype(np.float32)
        actual = np.memmap(self.directory / "canvas.f32", mode="w+", shape=base.shape, dtype=np.float32)
        actual[:] = expected
        for target in ([4, 4, 40, 40], [20, 12, 40, 40]):
            edit = rng.integers(0, 256, base.shape, dtype=np.uint8)
            alpha = np.zeros((64, 64, 1), dtype=np.float32)
            x, y, w, h = target
            alpha[y:y+h, x:x+w] = feather_mask(w, h, 8)
            expected = expected * (1-alpha) + edit.astype(np.float32) * alpha
            blend_target(actual, edit, [0, 0, 64, 64], target, 8)
        np.testing.assert_array_equal(np.clip(actual, 0, 255).astype(np.uint8),
                                      np.clip(expected, 0, 255).astype(np.uint8))
        actual._mmap.close()

    def test_ane_scratch_cleanup_on_failure(self):
        from ane_contract import scratch_workspace
        scratch = self.directory / "new-scratch"
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with scratch_workspace(scratch):
                (scratch / "tile-001.npz").write_bytes(b"generated fixture")
                raise RuntimeError("fixture")
        self.assertFalse(scratch.exists())

    def test_ane_scratch_never_removes_existing_directory(self):
        from ane_contract import scratch_workspace
        scratch = self.directory / "existing"
        scratch.mkdir()
        precious = scratch / "original"
        precious.write_text("keep")
        with self.assertRaises(FileExistsError):
            with scratch_workspace(scratch):
                pass
        self.assertEqual(precious.read_text(), "keep")

    def test_ane_missing_conversion_evidence_is_rejected(self):
        from ane_contract import conversion_evidence
        with self.assertRaises((FileNotFoundError, ValueError)):
            conversion_evidence(self.directory)

    def test_conversion_bytes_bound_and_stale_package_rejected(self):
        from ane_contract import conversion_evidence, PACKAGE_COMPONENTS, PACKAGE_FILES
        from pipeline_contract import sha256
        for name, (component, weight_hash) in PACKAGE_COMPONENTS.items():
            hashes = {}
            for relative in PACKAGE_FILES:
                path = self.directory / name / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + relative).encode())
                hashes[relative] = sha256(path)
            (self.directory / f"{component}-conversion.json").write_text(json.dumps({
                "component": component, "weight_sha256": weight_hash, "package_files": hashes}))
        self.assertEqual(len(conversion_evidence(self.directory)), 3)
        (self.directory / next(iter(PACKAGE_COMPONENTS)) / PACKAGE_FILES[-1]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "provenance"):
            conversion_evidence(self.directory)

    def test_partial_artifact_copy_preserves_previous_revision(self):
        from pipeline_contract import immutable_artifact
        source = self.directory / "generated.png"
        source.write_bytes(b"accepted old bytes")
        previous = immutable_artifact(self.directory, "region", source)
        source.write_bytes(b"new bytes")
        def fail_copy(src, dest):
            Path(dest).write_bytes(b"partial")
            raise OSError("disk full fixture")
        with patch.object(shutil, "copyfile", side_effect=fail_copy):
            with self.assertRaises(OSError):
                immutable_artifact(self.directory, "region", source)
        self.assertEqual(previous.read_bytes(), b"accepted old bytes")
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_parallel_manifest_updates_are_not_lost(self):
        import subprocess
        manifest = self.directory / "manifest.json"
        manifest.write_text(json.dumps({"value": 0}))
        code = ("import sys,time; from pathlib import Path; from pipeline_contract import locked_manifest; "
                "p=Path(sys.argv[1]);\nwith locked_manifest(p) as data:\n"
                " data['value']+=1\n time.sleep(0.04)\n")
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "scripts"), "PYTHONDONTWRITEBYTECODE": "1"}
        processes = [subprocess.Popen([sys.executable, "-c", code, str(manifest)], env=environment) for _ in range(6)]
        self.assertTrue(all(process.wait() == 0 for process in processes))
        self.assertEqual(json.loads(manifest.read_text())["value"], 6)

    def test_modes_preserve_alpha_and_gray(self):
        from PIL import Image
        from pipeline_contract import preserve_mode
        for mode in ("RGB", "RGBA", "L", "LA"):
            source = Image.new(mode, (16, 16))
            if "A" in mode:
                source.putalpha(83)
            result = preserve_mode(Image.new("RGB", (64, 64), "#654321"), source)
            self.assertEqual(result.mode, mode)
            if "A" in mode:
                self.assertEqual(result.getchannel("A").getextrema(), (83, 83))


if __name__ == "__main__":
    unittest.main()
