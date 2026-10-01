"""Experimental 4x VOSR2 route: MPS encode/DINO, ANE DiT/decode.

The two Core ML DiT segments must be the verified pretrained 0–17 and 18–35
packages. This executor never runs the PyTorch DiT or a CPU model fallback.
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

RUNTIME = Path(os.environ['VOSR2_RUNTIME']).expanduser().resolve()
PACKAGES = Path(os.environ['VOSR2_ANE_PACKAGES']).expanduser().resolve()
PACKAGE_FILES = ('Manifest.json', 'Data/com.apple.CoreML/model.mlmodel',
                 'Data/com.apple.CoreML/weights/weight.bin')
PACKAGE_STAGES = (('dit-segment-0-ane.mlpackage', 'dit_0'),
                  ('dit-segment-1-ane.mlpackage', 'dit_1'),
                  ('qwen-vae-decoder-128-ane.mlpackage', 'vae_decoder'))
import sys
sys.path.insert(0, str(RUNTIME))
import tiled_vae
from inference_vosr_onestep import get_venc_features, load_dinov2, wavelet_color_fix
from models.qwenimage_vae2d import AutoencoderKLQwenImage2D


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reject_stacked_vosr2(source):
    report_path = source.with_name(f'{source.stem}-runtime-report.json')
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    if (report.get('output_sha256') == digest(source)
            and report.get('route', '').startswith('MPS VAE encode + MPS DINOv2')):
        raise ValueError('Source is already a VOSR2 output; repeated semantic 4x is not approved')


def machine_name():
    return subprocess.check_output(
        ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()


def verify_package_plan(plan):
    for key, actual in (('machine', machine_name()),
                        ('macos_version', platform.mac_ver()[0]),
                        ('coremltools_version', ct.__version__)):
        if plan.get(key) != actual:
            raise ValueError(f'ANE plan {key} is {plan.get(key)!r}, expected {actual!r}')
    if set(plan['packages']) != {name for name, _ in PACKAGE_STAGES}:
        raise ValueError('ANE plan package set does not match this executor')
    for name, _ in PACKAGE_STAGES:
        evidence = plan['packages'][name]
        if set(evidence['files']) != set(PACKAGE_FILES):
            raise ValueError(f'ANE plan file set is incomplete: {name}')
        for relative in PACKAGE_FILES:
            path = PACKAGES / name / relative
            if digest(path) != evidence['files'][relative]:
                raise ValueError(f'Core ML package changed: {path}')
        counts = evidence['preferred_devices']
        if counts.get('MLNeuralEngineComputeDevice', 0) <= counts.get('MLCPUComputeDevice', 0):
            raise RuntimeError(f'ANE does not own most recognized operations: {name}: {counts}')


def load_ane_model(path, verified_plan=None):
    start = time.monotonic()
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_seconds = time.monotonic() - start
    plan_seconds = 0.0
    if verified_plan is not None:
        counts = verified_plan['packages'][path.name]['preferred_devices']
    else:
        start = time.monotonic()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
        counts = {}
        for op in plan.model_structure.program.functions['main'].block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            key = type(usage.preferred_compute_device).__name__
            counts[key] = counts.get(key, 0) + 1
        plan_seconds = time.monotonic() - start
    if counts.get('MLNeuralEngineComputeDevice', 0) <= counts.get('MLCPUComputeDevice', 0):
        raise RuntimeError(f'ANE does not own most recognized operations: {counts}')
    return model, load_seconds, plan_seconds, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--align', choices=('nofix', 'wavelet'), default='wavelet')
    parser.add_argument('--verified-package-plan', type=Path)
    parser.add_argument('--save-package-plan', type=Path)
    args = parser.parse_args()
    if args.verified_package_plan and args.save_package_plan:
        parser.error('Use either a verified plan or save a new plan, not both')
    if args.save_package_plan and args.save_package_plan.exists():
        raise FileExistsError(args.save_package_plan)
    if not RUNTIME.is_dir() or not PACKAGES.is_dir():
        raise FileNotFoundError('VOSR2_RUNTIME and VOSR2_ANE_PACKAGES must be existing directories')
    if platform.system() != 'Darwin' or int(platform.mac_ver()[0].split('.')[0]) < 15:
        raise RuntimeError('VOSR2 ANE inference requires macOS 15 or later')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS is required for VAE encoding and DINO')
    if args.output.exists() or args.scratch.exists():
        raise FileExistsError('Output and scratch must both be new paths')
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    reject_stacked_vosr2(args.source)
    verified_plan = None
    if args.verified_package_plan is not None:
        verified_plan = json.loads(args.verified_package_plan.read_text())
        verify_package_plan(verified_plan)
    source = Image.open(args.source).convert('RGB')
    width, height = source.size
    if width % 8 or height % 8:
        raise ValueError('Source dimensions must be multiples of 8 for exact 4x canvas')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timings = {}
    report = {'status': 'candidate_not_reviewed', 'source': str(args.source),
              'source_sha256': digest(args.source), 'seed': args.seed,
              'route': 'MPS VAE encode + MPS DINOv2-L + two ANE DiT segments + ANE VAE decode',
              'align': args.align, 'runtime': str(RUNTIME),
              'packages': str(PACKAGES), 'stages': {}}
    if verified_plan is not None:
        report['verified_package_plan_sha256'] = digest(args.verified_package_plan)
    total_start = time.monotonic()

    input_image = source.resize((width * 4, height * 4), Image.BICUBIC)
    expected_size = input_image.size
    lq = transforms.ToTensor()(input_image).unsqueeze(0).to('mps') * 2 - 1
    config = SimpleNamespace(ae_type='qwen', vae_tile_size=1024,
                             vae_tile_overlap=128, tile_overlap=64,
                             posterior_mode=True, dinov2_size=448,
                             enc_type='dinov2l', layer_dinov2b_list=[17])

    start = time.monotonic()
    vae = AutoencoderKLQwenImage2D.from_pretrained(
        RUNTIME / 'preset/ckpts/Qwen-Image-vae-2d').to('mps').eval()
    with torch.inference_mode():
        lq_latent, mean, std = tiled_vae.encode_dispatch(vae, lq, config, 'mps')
    torch.mps.synchronize()
    timings['vae_encode_and_load_seconds'] = time.monotonic() - start
    z_initial = torch.randn_like(lq_latent)
    lq_array = lq_latent.cpu().numpy()
    z_array = z_initial.cpu().numpy()
    mean_array = mean.cpu().numpy()
    std_array = std.cpu().numpy()
    del vae, lq_latent, z_initial, mean, std
    gc.collect()
    torch.mps.empty_cache()

    _, channels, latent_h, latent_w = lq_array.shape
    tile_size, overlap = 64, 8
    h_positions = tiled_vae._make_tile_grid(latent_h, tile_size, overlap)
    w_positions = tiled_vae._make_tile_grid(latent_w, tile_size, overlap)
    locations = [(hi, wi) for hi in h_positions for wi in w_positions]
    report['latent_shape'] = list(lq_array.shape)
    report['dit_grid'] = {'rows': len(h_positions), 'columns': len(w_positions),
                          'tile_count': len(locations), 'latent_tile': tile_size,
                          'latent_overlap': overlap}
    print(f"DINO tiles: {len(locations)} ({len(h_positions)}x{len(w_positions)})", flush=True)

    start = time.monotonic()
    venc = load_dinov2(config, 'mps')
    with torch.inference_mode():
        for index, (hi, wi) in enumerate(locations):
            crop = lq[:, :, hi*8:(hi+tile_size)*8, wi*8:(wi+tile_size)*8]
            features = get_venc_features(venc, crop, config)[0].cpu().numpy()
            token_input = np.concatenate((
                lq_array[:, :, hi:hi+tile_size, wi:wi+tile_size],
                z_array[:, :, hi:hi+tile_size, wi:wi+tile_size]), axis=1)
            np.savez(args.scratch / f'tile-{index:03d}.npz',
                     tokens=token_input, features=features)
            if (index + 1) % 20 == 0 or index + 1 == len(locations):
                print(f'DINO prepared {index + 1}/{len(locations)}', flush=True)
    torch.mps.synchronize()
    timings['dino_load_and_features_seconds'] = time.monotonic() - start
    del venc, lq
    gc.collect()
    torch.mps.empty_cache()

    current_time = np.ones((1,), dtype=np.float32)
    next_time = np.zeros((1,), dtype=np.float32)
    first, load_seconds, plan_seconds, devices = load_ane_model(
        PACKAGES / 'dit-segment-0-ane.mlpackage', verified_plan)
    report['stages']['dit_0'] = {'load_seconds': load_seconds,
                                 'plan_seconds': plan_seconds, 'preferred_devices': devices}
    start = time.monotonic()
    for index in range(len(locations)):
        with np.load(args.scratch / f'tile-{index:03d}.npz') as tile:
            inputs = {'tokens': tile['tokens'], 'features': tile['features'],
                      'time_current': current_time, 'time_next': next_time}
        hidden = first.predict(inputs)['hidden']
        np.save(args.scratch / f'hidden-{index:03d}.npy', hidden)
        if (index + 1) % 20 == 0 or index + 1 == len(locations):
            print(f'ANE DiT 0–17: {index + 1}/{len(locations)}', flush=True)
    timings['dit_0_predict_seconds'] = time.monotonic() - start
    del first
    gc.collect()

    second, load_seconds, plan_seconds, devices = load_ane_model(
        PACKAGES / 'dit-segment-1-ane.mlpackage', verified_plan)
    report['stages']['dit_1'] = {'load_seconds': load_seconds,
                                 'plan_seconds': plan_seconds, 'preferred_devices': devices}
    weight = tiled_vae._gaussian_weights(tile_size, tile_size, channels, 'cpu').numpy()
    velocity_acc = np.zeros_like(z_array)
    weight_acc = np.zeros_like(z_array)
    start = time.monotonic()
    for index, (hi, wi) in enumerate(locations):
        with np.load(args.scratch / f'tile-{index:03d}.npz') as tile:
            inputs = {'tokens': np.load(args.scratch / f'hidden-{index:03d}.npy'),
                      'features': tile['features'],
                      'time_current': current_time, 'time_next': next_time}
        velocity = second.predict(inputs)['velocity']
        velocity_acc[:, :, hi:hi+tile_size, wi:wi+tile_size] += velocity * weight
        weight_acc[:, :, hi:hi+tile_size, wi:wi+tile_size] += weight
        if (index + 1) % 20 == 0 or index + 1 == len(locations):
            print(f'ANE DiT 18–35: {index + 1}/{len(locations)}', flush=True)
    timings['dit_1_predict_seconds'] = time.monotonic() - start
    sr_latent = torch.from_numpy(z_array - velocity_acc / weight_acc)
    del second, velocity_acc, weight_acc
    gc.collect()

    decoder, load_seconds, plan_seconds, devices = load_ane_model(
        PACKAGES / 'qwen-vae-decoder-128-ane.mlpackage', verified_plan)
    report['stages']['vae_decoder'] = {'load_seconds': load_seconds,
                                       'plan_seconds': plan_seconds,
                                       'preferred_devices': devices}
    original_decode = tiled_vae.decode_latent

    def ane_decode_latent(vae, latent, config, mean, std, light_decoder=None):
        if tuple(latent.shape) != (1, 16, 128, 128):
            raise ValueError(f'Unexpected ANE decoder tile: {tuple(latent.shape)}')
        raw = (latent / std + mean).numpy()
        return torch.from_numpy(decoder.predict({'latent': raw})['pixels'])

    tiled_vae.decode_latent = ane_decode_latent
    try:
        start = time.monotonic()
        with torch.inference_mode():
            pixels = tiled_vae.decode_dispatch(
                None, sr_latent, config, torch.from_numpy(mean_array),
                torch.from_numpy(std_array))
        timings['vae_decode_seconds'] = time.monotonic() - start
    finally:
        tiled_vae.decode_latent = original_decode
    del decoder, sr_latent
    gc.collect()

    result = transforms.ToPILImage()(pixels[0] * 0.5 + 0.5)
    if result.size != expected_size:
        raise ValueError(f'Wrong output size: {result.size} != {expected_size}')
    if args.align == 'wavelet':
        result = wavelet_color_fix(result, input_image)
    result.save(args.output)
    report['output'] = str(args.output)
    report['output_sha256'] = digest(args.output)
    report['output_size'] = list(result.size)
    report['output_bytes'] = args.output.stat().st_size
    report['timings'] = timings
    if args.save_package_plan:
        plan = {'machine': machine_name(),
                'macos_version': platform.mac_ver()[0],
                'coremltools_version': ct.__version__,
                'evidence': str(args.output), 'packages': {}}
        for name, stage in PACKAGE_STAGES:
            package = PACKAGES / name
            plan['packages'][name] = {
                'files': {relative: digest(package / relative) for relative in PACKAGE_FILES},
                'preferred_devices': report['stages'][stage]['preferred_devices']}
        args.save_package_plan.parent.mkdir(parents=True, exist_ok=True)
        args.save_package_plan.write_text(json.dumps(plan, indent=2))
        report['saved_package_plan_sha256'] = digest(args.save_package_plan)
    report['total_wall_seconds'] = time.monotonic() - total_start
    (args.output.parent / f'{args.output.stem}-runtime-report.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps({'output': report['output'], 'output_size': report['output_size'],
                      'total_wall_seconds': report['total_wall_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
