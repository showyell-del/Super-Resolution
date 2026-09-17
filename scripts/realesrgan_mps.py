"""Minimal Real-ESRGAN x4 inference runtime for PyTorch MPS.

RRDBNet is adapted from BasicSR (Apache-2.0):
https://github.com/XPixelGroup/BasicSR

Tiled inference is adapted from Real-ESRGAN (BSD-3-Clause):
https://github.com/xinntao/Real-ESRGAN

See THIRD_PARTY.md and third_party/licenses/ for attribution and license texts.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.rdb1(x)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return output * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *(RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block))
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv_first(x)
        body_feature = self.conv_body(self.body(feature))
        feature = feature + body_feature
        feature = self.lrelu(self.conv_up1(F.interpolate(feature, scale_factor=2, mode="nearest")))
        feature = self.lrelu(self.conv_up2(F.interpolate(feature, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feature)))


class RealESRGANMPS:
    native_scale = 4

    def __init__(self, model_path: Path, tile: int, tile_pad: int) -> None:
        self.device = torch.device("mps")
        self.tile = tile
        self.tile_pad = tile_pad
        self.model = RRDBNet()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        parameters = checkpoint.get("params_ema", checkpoint.get("params"))
        if parameters is None:
            raise ValueError("Real-ESRGAN checkpoint contains neither params_ema nor params")
        self.model.load_state_dict(parameters, strict=True)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def _infer_tiles(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        output = image.new_zeros((1, 3, height * self.native_scale, width * self.native_scale))
        tiles_x = math.ceil(width / self.tile)
        tiles_y = math.ceil(height / self.tile)

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                start_x = tile_x * self.tile
                end_x = min(start_x + self.tile, width)
                start_y = tile_y * self.tile
                end_y = min(start_y + self.tile, height)
                padded_start_x = max(start_x - self.tile_pad, 0)
                padded_end_x = min(end_x + self.tile_pad, width)
                padded_start_y = max(start_y - self.tile_pad, 0)
                padded_end_y = min(end_y + self.tile_pad, height)

                input_tile = image[
                    :, :, padded_start_y:padded_end_y, padded_start_x:padded_end_x
                ]
                output_tile = self.model(input_tile)

                output_start_x = start_x * self.native_scale
                output_end_x = end_x * self.native_scale
                output_start_y = start_y * self.native_scale
                output_end_y = end_y * self.native_scale
                tile_start_x = (start_x - padded_start_x) * self.native_scale
                tile_end_x = tile_start_x + (end_x - start_x) * self.native_scale
                tile_start_y = (start_y - padded_start_y) * self.native_scale
                tile_end_y = tile_start_y + (end_y - start_y) * self.native_scale
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                    :, :, tile_start_y:tile_end_y, tile_start_x:tile_end_x
                ]
                index = tile_y * tiles_x + tile_x + 1
                print(f"\tTile {index}/{tiles_x * tiles_y}")

        return output

    def enhance(self, input_bgr: np.ndarray, outscale: float) -> np.ndarray:
        if input_bgr.dtype != np.uint8 or input_bgr.ndim != 3 or input_bgr.shape[2] != 3:
            raise ValueError("Input must be an 8-bit three-channel BGR image")
        height, width = input_bgr.shape[:2]
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.transpose(input_rgb, (2, 0, 1))).float().div_(255.0)
        tensor = tensor.unsqueeze(0).to(self.device)
        output = self._infer_tiles(tensor)
        output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output_rgb = np.transpose(output, (1, 2, 0))
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        output_bgr = (output_bgr * 255.0).round().astype(np.uint8)
        if outscale != self.native_scale:
            output_bgr = cv2.resize(
                output_bgr,
                (round(width * outscale), round(height * outscale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return output_bgr
