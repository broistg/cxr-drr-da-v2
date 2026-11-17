"""U-Net with a ResNet-18 encoder for E16 (torch; imported only by torch scripts). Input (B, 3, H, W) normalised like the classifier
(ImageNet mean/std), output logits (B, 1, H, W). No extra dependency: the decoder is plain torch.

NOT run in the sandbox where it was written (no torch): the first run on the user's machine is the test. Paste any traceback."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNetR18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)            # 64 ch, 1/2
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # 64/4, 128/8, 256/16, 512/32
        self.d4, self.d3 = conv_block(512 + 256, 256), conv_block(256 + 128, 128)
        self.d2, self.d1 = conv_block(128 + 64, 64), conv_block(64 + 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up(a, ref):
        return F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0)); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        y = self.d4(torch.cat([self._up(s4, s3), s3], 1))
        y = self.d3(torch.cat([self._up(y, s2), s2], 1))
        y = self.d2(torch.cat([self._up(y, s1), s1], 1))
        y = self.d1(torch.cat([self._up(y, s0), s0], 1))
        return self.head(F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False))


def to_input(x01: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) in [0, 1] -> (B, 3, H, W) ImageNet-normalised."""
    mean = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01.expand(-1, 3, -1, -1) - mean) / std


def augment_pair(x: torch.Tensor, m: torch.Tensor) -> tuple:
    """Same augmentation as scripts/train_cls.py (flip, +-8 deg, +-10% scale, +-10% shift, contrast/brightness, noise); the geometry is
    applied to the image and the mask together, nearest-like for the mask via the same bilinear warp then re-thresholding at 0.5."""
    dev, B = x.device, x.shape[0]
    z = torch.cat([x, m], 1)
    flip = torch.rand(B, device=dev) < 0.5
    z = torch.where(flip.view(B, 1, 1, 1), z.flip(-1), z)
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(8)
    sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.1
    tx, ty = [(torch.rand(B, device=dev) * 2 - 1) * 0.1 for _ in range(2)]
    th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                      torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
    z = F.grid_sample(z, F.affine_grid(th, z.shape, align_corners=False), padding_mode="zeros", align_corners=False)
    x, m = z[:, :1], (z[:, 1:] > 0.5).float()
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1
    x = (x - 0.5) * c + 0.5 + b
    x = x + torch.randn_like(x) * (torch.rand(B, 1, 1, 1, device=dev) * 0.02)
    return x.clamp(0, 1), m


def soft_dice_loss(logits: torch.Tensor, m: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits.float())
    inter = (p * m).sum()
    return 1 - (2 * inter + eps) / (p.sum() + m.sum() + eps)


def seg_loss(logits: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), m) + soft_dice_loss(logits, m)


@torch.no_grad()
def predict_probs(model: nn.Module, x_uint8, dev, batch: int = 32):
    """(N, H, W) uint8 images -> (N, H, W) float32 probabilities."""
    import numpy as np
    model.eval()
    out = []
    for i in range(0, len(x_uint8), batch):
        xb = torch.from_numpy(x_uint8[i:i + batch]).to(dev).float().div(255).unsqueeze(1)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(to_input(xb)).float()).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + x_uint8.shape[1:], np.float32)
