#!/usr/bin/env python3
"""Convert pinned VOSR2 DiT halves or Qwen VAE decoder to fixed-shape Core ML."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import coremltools as ct
import torch
from safetensors import safe_open

from apple_preflight import require_apple_silicon


DIT_SHA256 = "bdcaa81e4c675b6074de643e27daafe73eb8125d4d0264d1bf30a004e9644b71"
VAE_SHA256 = "1dd2c67f3f29734d48709513ed181d4f351966d76ba95f77a5cff8e1daf547fa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_dit(runtime: Path, number: int):
    from models.lightningdit import LightningDiT

    class Segment(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = LightningDiT(
                input_size=64, patch_size=2, in_channels=32, out_channels=16,
                hidden_size=1536, depth=18, num_heads=24, mlp_ratio=4,
                z_dims=1024, encdim_ratio=3, auxiliary_time_cond=False,
                use_qknorm=True, use_swiglu=True, use_rope=True,
                use_rmsnorm=True, wo_shift=False, num_fused_layers=1,
            )

        def forward(self, tokens, time_current, time_next, features):
            model = self.dit
            x = model.x_embedder(tokens) if number == 0 else tokens
            c = model.t_embedder(time_current)
            c0 = model.t_block(c)
            z = model.mlp_ca(model.layer_norm(features))
            for block in model.blocks:
                x = block(x, c0, z, model.feat_rope)
            return x if number == 0 else model.unpatchify(model.final_layer(x, c))

    checkpoint = runtime / "preset/ckpts/VOSR2/checkpoints/ema_model.safetensors"
    if sha256(checkpoint) != DIT_SHA256:
        raise ValueError(f"Unverified VOSR2 checkpoint: {checkpoint}")
    segment = Segment().eval()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
        with torch.no_grad():
            for name, destination in segment.dit.state_dict().items():
                source_name = name
                if number == 1 and name.startswith("blocks."):
                    pieces = name.split(".", 2)
                    source_name = f"blocks.{int(pieces[1]) + 18}.{pieces[2]}"
                destination.copy_(archive.get_tensor(source_name))
    torch.manual_seed(42)
    tokens = torch.randn(1, 32, 64, 64) if number == 0 else torch.randn(1, 1024, 1536)
    times = (torch.ones(1), torch.zeros(1))
    features = torch.randn(1, 1024, 1024)
    inputs = (tokens, *times, features)
    with torch.inference_mode():
        sample = segment(*inputs)
        if not torch.isfinite(sample).all():
            raise ValueError("DiT segment produced non-finite output before conversion")
        traced = torch.jit.trace(segment, inputs, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=value.shape) for name, value in zip(
            ("tokens", "time_current", "time_next", "features"), inputs)],
        outputs=[ct.TensorType(name="hidden" if number == 0 else "velocity")],
        convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True,
    )
    return converted, list(sample.shape), DIT_SHA256


def convert_decoder(runtime: Path):
    from models.qwenimage_vae2d import AutoencoderKLQwenImage2D

    class Decoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder

        def forward(self, latent):
            return self.decoder(self.post_quant_conv(latent)).clamp(-1, 1)

    weights = runtime / "preset/ckpts/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors"
    if sha256(weights) != VAE_SHA256:
        raise ValueError(f"Unverified Qwen VAE checkpoint: {weights}")
    vae = AutoencoderKLQwenImage2D.from_pretrained(
        runtime / "preset/ckpts/Qwen-Image-vae-2d")
    model = Decoder(vae).eval()
    latent = torch.zeros(1, 16, 128, 128)
    with torch.inference_mode():
        sample = model(latent)
        if not torch.isfinite(sample).all():
            raise ValueError("VAE decoder produced non-finite output before conversion")
        traced = torch.jit.trace(model, latent, check_trace=False)
    converted = ct.convert(
        traced, inputs=[ct.TensorType(name="latent", shape=latent.shape)],
        outputs=[ct.TensorType(name="pixels")], convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    return converted, list(sample.shape), VAE_SHA256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", choices=("dit0", "dit1", "decoder"), required=True)
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require_apple_silicon([runtime, output_dir], check_mps=True)
    if int(platform.mac_ver()[0].split('.')[0]) < 15:
        raise RuntimeError('VOSR2 ANE packages require macOS 15 or later')
    if not runtime.is_dir():
        raise FileNotFoundError(runtime)
    output_dir.mkdir(parents=True, exist_ok=True)
    package = output_dir / {
        "dit0": "dit-segment-0-ane.mlpackage",
        "dit1": "dit-segment-1-ane.mlpackage",
        "decoder": "qwen-vae-decoder-128-ane.mlpackage",
    }[args.component]
    if package.exists():
        raise FileExistsError(package)
    sys.path.insert(0, str(runtime))
    torch.set_num_threads(4)
    start = time.monotonic()
    if args.component == "decoder":
        converted, shape, weight_hash = convert_decoder(runtime)
    else:
        converted, shape, weight_hash = convert_dit(runtime, int(args.component[-1]))
    converted.save(str(package))
    report = {"component": args.component, "package": str(package),
              "sample_output_shape": shape, "weight_sha256": weight_hash,
              "seconds": time.monotonic() - start,
              "status": "converted_not_yet_validated_on_target_device"}
    (output_dir / f"{args.component}-conversion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
