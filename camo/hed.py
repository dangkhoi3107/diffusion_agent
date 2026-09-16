"""HED soft-edge detector (ControlNet "Apache2" variant), without controlnet-aux.

Same network and post-processing as ``controlnet_aux.HEDdetector`` with
``safe=False`` and ``scribble=False``; weights: ``lllyasviel/Annotators/ControlNetHED.pth``.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        for _ in range(1, layers):
            self.convs.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.projection = nn.Conv2d(out_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, down_sampling: bool = False):
        h = F.max_pool2d(x, kernel_size=2, stride=2) if down_sampling else x
        for conv in self.convs:
            h = F.relu(conv(h))
        return h, self.projection(h)


class _ControlNetHED(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.block1 = _DoubleConvBlock(3, 64, 2)
        self.block2 = _DoubleConvBlock(64, 128, 2)
        self.block3 = _DoubleConvBlock(128, 256, 3)
        self.block4 = _DoubleConvBlock(256, 512, 3)
        self.block5 = _DoubleConvBlock(512, 512, 3)

    def forward(self, x: torch.Tensor):
        h = x - self.norm
        h, p1 = self.block1(h)
        h, p2 = self.block2(h, down_sampling=True)
        h, p3 = self.block3(h, down_sampling=True)
        h, p4 = self.block4(h, down_sampling=True)
        _, p5 = self.block5(h, down_sampling=True)
        return p1, p2, p3, p4, p5


class HEDDetector:
    def __init__(self, net: _ControlNetHED, device: str):
        self.net = net.float().eval().to(device)
        self.device = device

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str, filename: str = "ControlNetHED.pth") -> "HEDDetector":
        from huggingface_hub import hf_hub_download

        net = _ControlNetHED()
        state = torch.load(hf_hub_download(repo_id, filename), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        return cls(net, device)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB -> HxW uint8 edge probability."""
        h, w = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1)[None].to(self.device)
        maps = [p[0, 0].float().cpu().numpy() for p in self.net(x)]
        maps = [cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in maps]
        edge = 1.0 / (1.0 + np.exp(-np.mean(np.stack(maps, axis=2), axis=2).astype(np.float64)))
        return (edge * 255.0).clip(0, 255).astype(np.uint8)
