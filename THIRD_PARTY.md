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

No endorsement by upstream authors is implied.
