import argparse
from pathlib import Path

import torch
import torch.nn as nn


class VectorGemm(nn.Module):
    def forward(self, queries, database):
        return torch.matmul(queries, database.transpose(0, 1))


def parse_args():
    parser = argparse.ArgumentParser(description="Export vector GEMM ONNX for BM1684X KNN backend.")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--query_rows", type=int, default=1600, help="Static query row count")
    parser.add_argument("--database_rows", type=int, default=2048, help="Static database row count")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = VectorGemm().eval()
    queries = torch.randn(args.query_rows, args.embed_dim, dtype=torch.float32)
    database = torch.randn(args.database_rows, args.embed_dim, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (queries, database),
            str(output_path),
            input_names=["queries", "database"],
            output_names=["similarity"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported vector GEMM ONNX to: {output_path}")


if __name__ == "__main__":
    main()
