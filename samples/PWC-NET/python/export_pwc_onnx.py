#!/usr/bin/env python3
"""Export the official PyTorch PWC-Net checkpoint without the CUDA correlation op.

The original repository uses a legacy torch.utils.ffi CUDA correlation extension.
This exporter injects a fixed-displacement, ONNX-exportable implementation of the
same cost volume and patches the old CUDA-only warp helper for CPU export.
"""
import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


class PureCorrelation(torch.nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4,
                 stride1=1, stride2=1, corr_multiply=1):
        super().__init__()
        if kernel_size != 1 or stride1 != 1 or stride2 != 1:
            raise ValueError("exporter only implements the PWC-Net md=4 configuration")
        self.md = int(max_displacement)
        self.pad = int(pad_size)
        self.mult = float(corr_multiply)

    def forward(self, a, b):
        # PWC-Net's CUDA correlation computes a channel-wise mean product for
        # every displacement in the fixed (2*md+1)^2 search window.
        h = a.shape[2]
        w = a.shape[3]
        padded = F.pad(b, (self.pad, self.pad, self.pad, self.pad))
        values = []
        for dy in range(-self.md, self.md + 1):
            for dx in range(-self.md, self.md + 1):
                y0 = self.pad + dy
                x0 = self.pad + dx
                shifted = padded[:, :, y0:y0 + h, x0:x0 + w]
                values.append((a * shifted).mean(dim=1, keepdim=True) * self.mult)
        return torch.cat(values, dim=1)


def pure_warp(self, x, flo):
    b, _, h, w = x.size()
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device),
        indexing="ij")
    grid = torch.stack((xx, yy), dim=0).to(dtype=x.dtype)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    vgrid = grid + flo
    vgrid_x = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
    norm_grid = torch.stack((vgrid_x, vgrid_y), dim=-1)
    out = F.grid_sample(x, norm_grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(torch.ones_like(x[:, :1]), norm_grid,
                         mode="bilinear", padding_mode="zeros",
                         align_corners=False)
    mask = (mask >= 0.9999).to(dtype=x.dtype)
    return out * mask


def install_fake_correlation():
    # Make models.PWCNet importable without the obsolete torch.utils.ffi module.
    pkg = types.ModuleType("correlation_package")
    modules = types.ModuleType("correlation_package.modules")
    corr = types.ModuleType("correlation_package.modules.corr")
    corr.Correlation = PureCorrelation
    corr.Correlation1d = PureCorrelation
    sys.modules[pkg.__name__] = pkg
    sys.modules[modules.__name__] = modules
    sys.modules[corr.__name__] = corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    args = ap.parse_args()
    if args.height % 64 or args.width % 64:
        raise SystemExit("height and width must be multiples of 64")

    install_fake_correlation()
    source = Path(__file__).resolve().parent / "source" / "PyTorch"
    sys.path.insert(0, str(source))
    import models
    from models.PWCNet import PWCDCNet
    PWCDCNet.warp = pure_warp

    net = PWCDCNet().eval().cpu()
    data = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(data["state_dict"] if "state_dict" in data else data)
    sample = torch.zeros(1, 6, args.height, args.width, dtype=torch.float32)
    with torch.no_grad():
        output = net(sample)
    print("output_shape", tuple(output.shape))
    torch.onnx.export(
        net, sample, args.output,
        input_names=["frames"], output_names=["flow"],
        opset_version=16, do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )
    print("wrote", args.output)


if __name__ == "__main__":
    main()
