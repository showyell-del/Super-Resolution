# Third-party components

The minimal MPS inference implementation in `scripts/realesrgan_mps.py` is adapted from the pinned upstream revisions below. The repository does not vendor their unrelated training, CUDA, C++, MATLAB, test, documentation, or example trees. Original license texts are preserved in `third_party/licenses/`.

## BasicSR

- Upstream: <https://github.com/XPixelGroup/BasicSR>
- Revision: `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`
- Adapted component: RRDBNet architecture
- Location: `scripts/realesrgan_mps.py`
- License: Apache-2.0, preserved at `third_party/licenses/BasicSR-APACHE-2.0.txt`

## Real-ESRGAN

- Upstream: <https://github.com/xinntao/Real-ESRGAN>
- Revision: `a4abfb2979a7bbff3f69f58f58ae324608821e27`
- Adapted component: model loading and tiled x4 inference
- Location: `scripts/realesrgan_mps.py`
- License: BSD-3-Clause, preserved at `third_party/licenses/Real-ESRGAN-BSD-3-Clause.txt`

## RealESRGAN_x4plus model

- Official upstream release: <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth>
- Location: `runtime/weights/RealESRGAN_x4plus.pth`
- SHA-256: `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`
- License and attribution follow the Real-ESRGAN project distribution.

## VOSR and DINOv2 MPS patches

- VOSR upstream: <https://github.com/cswry/VOSR>, revision `516f292b99cf23c76fdc33351e86dc4f97711fe8`, Apache-2.0. The repository provides an adaptation patch at `patches/vosr2-mps.patch`; model weights are not bundled.
- DINOv2 upstream: <https://github.com/facebookresearch/dinov2>, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, Apache-2.0. The repository provides a Python 3.9 compatibility patch at `patches/dinov2-python39.patch`; pretrained weights are not bundled.
- Official VOSR2 checkpoint and Qwen 2D VAE source: <https://huggingface.co/CSWRY/VOSR>. Review the upstream model cards and licenses before redistributing weights.

No endorsement by upstream authors is implied.
