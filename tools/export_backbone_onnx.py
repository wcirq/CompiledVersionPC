import argparse
from pathlib import Path
from typing import Optional
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.backbone import FeatureBackbone


class BackboneExportWrapper(nn.Module):
    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        self.backbone = FeatureBackbone(weights_path=weights_path).eval()

    def forward(self, x):
        feat2, feat3 = self.backbone(x)
        return feat2, feat3


def parse_args():
    parser = argparse.ArgumentParser(description="Export ResNet50 feature backbone to ONNX for BM1684X.")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--weights_path", type=str, default=None, help="Optional local resnet50-11ad3fa6.pth path")
    parser.add_argument("--height", type=int, default=640, help="Static input height")
    parser.add_argument("--width", type=int, default=640, help="Static input width")
    parser.add_argument("--batch_size", type=int, default=1, help="Static input batch size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = BackboneExportWrapper(weights_path=args.weights_path).eval()
    dummy = torch.randn(args.batch_size, 3, args.height, args.width, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["feat2", "feat3"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported backbone ONNX to: {output_path}")


if __name__ == "__main__":
    main()
