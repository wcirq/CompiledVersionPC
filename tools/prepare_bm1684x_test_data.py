import argparse
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.backbone import FeatureBackbone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare BM1684X test inputs and reference outputs for TPU-MLIR."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="bm1684x_build",
        help="Workspace directory for generated npz files",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Optional local resnet50-11ad3fa6.pth path",
    )
    parser.add_argument("--height", type=int, default=640, help="Backbone input height")
    parser.add_argument("--width", type=int, default=640, help="Backbone input width")
    parser.add_argument("--batch_size", type=int, default=1, help="Backbone input batch size")
    parser.add_argument("--query_rows", type=int, default=1600, help="Vector GEMM query rows")
    parser.add_argument("--database_rows", type=int, default=2048, help="Vector GEMM database rows")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Vector GEMM embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    backbone_input = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)
    np.savez(work_dir / "backbone_ref_input.npz", input=backbone_input)

    backbone = FeatureBackbone(weights_path=args.weights_path).eval()
    with torch.no_grad():
        feat2, feat3 = backbone(torch.from_numpy(backbone_input).float())
    np.savez(
        work_dir / "backbone_top_outputs.npz",
        feat2=feat2.cpu().numpy().astype(np.float32),
        feat3=feat3.cpu().numpy().astype(np.float32),
    )

    queries = np.random.randn(args.query_rows, args.embed_dim).astype(np.float32)
    database = np.random.randn(args.database_rows, args.embed_dim).astype(np.float32)
    similarity = queries @ database.T
    np.savez(work_dir / "vector_gemm_inputs.npz", queries=queries, database=database)
    np.savez(work_dir / "vector_gemm_top_outputs.npz", similarity=similarity.astype(np.float32))

    print(f"Prepared BM1684X test data under: {work_dir}")
    print(f"  - {work_dir / 'backbone_ref_input.npz'}")
    print(f"  - {work_dir / 'backbone_top_outputs.npz'}")
    print(f"  - {work_dir / 'vector_gemm_inputs.npz'}")
    print(f"  - {work_dir / 'vector_gemm_top_outputs.npz'}")


if __name__ == "__main__":
    main()
