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

from PIL import Image
from apple_preflight import require_apple_silicon
from pipeline_contract import (load_approval, new_output, preserve_mode, atomic_json, box,
                               verify_repairs, neural_contract_sha256, validate_resolution)
from ane_contract import PACKAGE_FILES, conversion_evidence, validate_ane_trial, scratch_workspace

RUNTIME = Path(os.environ.get('VOSR2_RUNTIME', '')).expanduser().resolve()
PACKAGES = Path(os.environ.get('VOSR2_ANE_PACKAGES', '')).expanduser().resolve()
PACKAGE_STAGES = (('dit-segment-0-ane.mlpackage', 'dit_0'),
                  ('dit-segment-1-ane.mlpackage', 'dit_1'),
                  ('qwen-vae-decoder-128-ane.mlpackage', 'vae_decoder'))


def initialize_runtime():
    global ct, np, torch, transforms, tiled_vae, get_venc_features, load_dinov2, wavelet_color_fix, AutoencoderKLQwenImage2D
    import coremltools as ct
    import numpy as np
    import torch
    from torchvision import transforms
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


def verify_package_plan(plan, conversion):
    if plan.get('conversion_sha256') != conversion:
        raise ValueError('ANE plan conversion provenance changed')
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
            if conversion[name]['files'][relative] != evidence['files'][relative]:
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
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--trial-box', nargs=4, type=int)
    parser.add_argument('--trial-review', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--align', choices=('nofix', 'wavelet'), default='wavelet')
    parser.add_argument('--verified-package-plan', type=Path)
    parser.add_argument('--save-package-plan', type=Path)
    args = parser.parse_args()
    if args.verified_package_plan and args.save_package_plan:
        parser.error('Use a verified plan or save a new diagnostic plan, not both')
    if args.diagnostic != bool(args.trial_box) or (args.diagnostic and args.trial_review):
        parser.error('Diagnostic execution requires --trial-box and cannot use --trial-review')
    if not args.diagnostic and (not args.verified_package_plan or not args.trial_review):
        parser.error('Full execution requires verified packages and a passing --trial-review')
    if args.save_package_plan and not args.diagnostic:
        parser.error('Create package plans only during diagnostic crops')
    approval = load_approval(args.approval, args.source)
    source = Image.open(args.source)
    if args.trial_box:
        x, y, w, h = box(args.trial_box, source.size)
        source = source.crop((x, y, x+w, y+h))
    try:
        validate_resolution(approval, (source.width * 4, source.height * 4))
    except ValueError as error:
        parser.error(str(error))
    require_apple_silicon([args.source, args.scratch.parent, args.output.parent], check_mps=True)
    new_output(args.output)
    new_output(args.output.with_name(f'{args.output.stem}-runtime-report.json'))
    if args.scratch.exists() or args.scratch.is_symlink():
        raise FileExistsError('Scratch must be a new task-owned path')
    for path in (args.output, args.save_package_plan):
        if path is not None and path.resolve().is_relative_to(args.scratch.resolve()):
            raise ValueError('Output and package evidence must be outside disposable scratch')
    if args.save_package_plan:
        new_output(args.save_package_plan)
    if not os.environ.get('VOSR2_RUNTIME') or not os.environ.get('VOSR2_ANE_PACKAGES'):
        parser.error('Set VOSR2_RUNTIME and VOSR2_ANE_PACKAGES on the work volume')
    if platform.system() != 'Darwin' or int(platform.mac_ver()[0].split('.')[0]) < 15:
        raise RuntimeError('VOSR2 ANE inference requires macOS 15 or later')
    reject_stacked_vosr2(args.source)
    conversion = conversion_evidence(PACKAGES)
    if source.mode not in {'RGB', 'RGBA', 'L', 'LA'}:
        raise ValueError('Unsupported input mode; convert explicitly with color management')
    if source.width % 8 or source.height % 8:
        raise ValueError('Source dimensions must be multiples of 8 for exact 4x canvas')
    if not args.diagnostic and approval.get('repair_manifest'):
        verify_repairs(Path(approval['repair_manifest']['path']), args.source,
                       target_size=(source.width * 4, source.height * 4))
    if not args.diagnostic:
        validate_ane_trial(args.trial_review, args.source, args.approval, args.verified_package_plan,
                           args.seed, args.align, conversion)
    initialize_runtime()
    verified_plan = None
    if args.verified_package_plan:
        verified_plan = json.loads(args.verified_package_plan.read_text())
        verify_package_plan(verified_plan, conversion)
    with scratch_workspace(args.scratch):
        run_pipeline(args, source, verified_plan, conversion)


def run_pipeline(args, source, verified_plan, conversion):
    width, height = source.size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timings = {}
    report = {'status': 'candidate_not_reviewed', 'source': str(args.source),
              'source_sha256': digest(args.source), 'seed': args.seed,
              'approval_sha256': digest(args.approval), 'run_kind': 'diagnostic' if args.diagnostic else 'full',
              'neural_contract_sha256': neural_contract_sha256(load_approval(args.approval, args.source)),
              'trial_box': args.trial_box, 'conversion_sha256': conversion,
              'trial_review_sha256': digest(args.trial_review) if args.trial_review else None,
              'route': 'MPS VAE encode + MPS DINOv2-L + two ANE DiT segments + ANE VAE decode',
              'align': args.align, 'runtime': str(RUNTIME),
              'packages': str(PACKAGES), 'stages': {}}
    if verified_plan is not None:
        report['verified_package_plan_sha256'] = digest(args.verified_package_plan)
        report['package_plan_sha256'] = digest(args.verified_package_plan)
    total_start = time.monotonic()

    input_image = source.convert('RGB').resize((width * 4, height * 4), Image.BICUBIC)
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
        (args.scratch / f'tile-{index:03d}.npz').unlink()
        (args.scratch / f'hidden-{index:03d}.npy').unlink()
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

    def ane_decode_latent(vae, latent, config, mean, std, light_decoder=None, _decoder=decoder):
        if tuple(latent.shape) != (1, 16, 128, 128):
            raise ValueError(f'Unexpected ANE decoder tile: {tuple(latent.shape)}')
        raw = (latent / std + mean).numpy()
        return torch.from_numpy(_decoder.predict({'latent': raw})['pixels'])

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
    del ane_decode_latent, decoder, sr_latent
    gc.collect()

    result = transforms.ToPILImage()(pixels[0] * 0.5 + 0.5)
    if result.size != expected_size:
        raise ValueError(f'Wrong output size: {result.size} != {expected_size}')
    if args.align == 'wavelet':
        result = wavelet_color_fix(result, input_image)
    result = preserve_mode(result, source)
    options = {'compress_level': 2}
    if source.info.get('icc_profile'):
        options['icc_profile'] = source.info['icc_profile']
    result.save(args.output, **options)
    report['output'] = str(args.output)
    report['output_sha256'] = digest(args.output)
    report['output_size'] = list(result.size)
    report['output_bytes'] = args.output.stat().st_size
    report['timings'] = timings
    if args.save_package_plan:
        plan = {'machine': machine_name(),
                'macos_version': platform.mac_ver()[0],
                'coremltools_version': ct.__version__,
                'evidence': str(args.output), 'conversion_sha256': conversion, 'packages': {}}
        for name, stage in PACKAGE_STAGES:
            package = PACKAGES / name
            plan['packages'][name] = {
                'files': {relative: digest(package / relative) for relative in PACKAGE_FILES},
                'preferred_devices': report['stages'][stage]['preferred_devices']}
        args.save_package_plan.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.save_package_plan, plan)
        report['saved_package_plan_sha256'] = digest(args.save_package_plan)
        report['package_plan_sha256'] = digest(args.save_package_plan)
    report['total_wall_seconds'] = time.monotonic() - total_start
    atomic_json(args.output.parent / f'{args.output.stem}-runtime-report.json', report)
    print(json.dumps({'output': report['output'], 'output_size': report['output_size'],
                      'total_wall_seconds': report['total_wall_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
