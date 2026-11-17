"""Mô hình phân loại nhị phân (ResNet-18) + tăng cường GPU + suy luận + AdaBN. Cần torch/torchvision.

Tách từ scripts/train_cls.py của Giai đoạn 0 (logic giữ nguyên). Đầu vào là ảnh uint8 (N, H, W) hoặc batch float [0, 1] (B, 1, H, W).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Net:
    def __init__(self, dev: torch.device, pretrained: bool = True):
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.resnet18(weights=w)
        self.model.fc = nn.Linear(512, 1)
        self.model.to(dev)
        self.dev = dev
        self.mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    @property
    def amp(self) -> dict:
        return dict(device_type=self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda")

    def to_input(self, x01: torch.Tensor) -> torch.Tensor:          # (B, 1, H, W) in [0, 1]
        return (x01.expand(-1, 3, -1, -1) - self.mean) / self.std

    def features(self, x: torch.Tensor) -> torch.Tensor:            # 512-d post-GAP (what LMMD aligns)
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return torch.flatten(m.avgpool(x), 1)

    def augment(self, x: torch.Tensor) -> torch.Tensor:             # cùng một phép cho mọi nhánh
        dev, B = self.dev, x.shape[0]
        flip = torch.rand(B, device=dev) < 0.5
        x = torch.where(flip.view(B, 1, 1, 1), x.flip(-1), x)
        ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(8)
        sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.1
        tx, ty = [(torch.rand(B, device=dev) * 2 - 1) * 0.1 for _ in range(2)]
        th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                          torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
        c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.2
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1
        x = (x - 0.5) * c + 0.5 + b
        x = x + torch.randn_like(x) * (torch.rand(B, 1, 1, 1, device=dev) * 0.02)
        return x.clamp(0, 1)

    def tensor(self, x_uint8: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x_uint8).to(self.dev).float().div(255).unsqueeze(1)

    @torch.no_grad()
    def predict(self, x_uint8: np.ndarray) -> np.ndarray:
        self.model.eval()
        s = []
        for i in range(0, len(x_uint8), 64):
            with torch.autocast(**self.amp):
                s.append(torch.sigmoid(self.model(self.to_input(self.tensor(x_uint8[i:i + 64]))).float()).squeeze(1).cpu().numpy())
        return np.concatenate(s)

    def state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def adapt_bn(self, pool: np.ndarray, n: int = 3000, seed: int = 4242) -> None:
        """AdaBN: thống kê BatchNorm tích luỹ trên pool đích không nhãn; trọng số giữ nguyên."""
        bns = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm2d)]
        for m in bns:
            m.reset_running_stats(); m.momentum = None
        sub = np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False)
        self.model.train()
        for i in range(0, len(sub), 64):
            with torch.autocast(**self.amp):
                self.model(self.to_input(self.tensor(pool[np.sort(sub[i:i + 64])])))
        for m in bns:
            m.momentum = 0.1
        self.model.eval()
