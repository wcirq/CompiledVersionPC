#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-$ROOT_DIR/bm1684x_build}"
IMAGE="${TPUC_DEV_IMAGE:-sophgo/tpuc_dev:latest}"
CONTAINER_NAME="${TPUC_DEV_CONTAINER:-my_sophgo}"
CONTAINER_WORK_DIR="${TPUC_DEV_CONTAINER_WORKDIR:-/workspace/patchcore_bm1684x_build}"

mkdir -p "$WORK_DIR"

cat <<EOF
Compile workspace: $WORK_DIR
Docker image: $IMAGE
Preferred container: $CONTAINER_NAME

Expected inputs:
  $WORK_DIR/backbone.onnx
  $WORK_DIR/backbone_ref_input.npz
  $WORK_DIR/backbone_top_outputs.npz
  $WORK_DIR/vector_gemm.onnx
  $WORK_DIR/vector_gemm_inputs.npz
  $WORK_DIR/vector_gemm_top_outputs.npz

Outputs:
  $WORK_DIR/backbone_1x3x640x640_bm1684x_f16.bmodel
  $WORK_DIR/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel
EOF

run_compile_commands() {
  cat <<'EOF'
set -euo pipefail

model_transform.py \
  --model_name backbone_resnet50 \
  --model_def backbone.onnx \
  --input_shapes [[1,3,640,640]] \
  --test_input backbone_ref_input.npz \
  --test_result backbone_top_outputs.npz \
  --mlir backbone.mlir

model_deploy.py \
  --mlir backbone.mlir \
  --quantize F16 \
  --processor bm1684x \
  --test_input backbone_resnet50_in_f32.npz \
  --test_reference backbone_top_outputs.npz \
  --model backbone_1x3x640x640_bm1684x_f16.bmodel

model_transform.py \
  --model_name vector_gemm \
  --model_def vector_gemm.onnx \
  --input_shapes [[1600,1024],[2048,1024]] \
  --test_input vector_gemm_inputs.npz \
  --test_result vector_gemm_top_outputs.npz \
  --mlir vector_gemm.mlir

model_deploy.py \
  --mlir vector_gemm.mlir \
  --quantize F16 \
  --processor bm1684x \
  --test_input vector_gemm_in_f32.npz \
  --test_reference vector_gemm_top_outputs.npz \
  --model vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel
EOF
}

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Using existing container: $CONTAINER_NAME"
  docker start "$CONTAINER_NAME" >/dev/null
  docker exec "$CONTAINER_NAME" bash -lc "rm -rf '$CONTAINER_WORK_DIR' && mkdir -p '$CONTAINER_WORK_DIR'"
  docker cp "$WORK_DIR/." "$CONTAINER_NAME:$CONTAINER_WORK_DIR/"
  docker exec "$CONTAINER_NAME" bash -lc "cd '$CONTAINER_WORK_DIR' && $(run_compile_commands)"
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
  docker cp "$CONTAINER_NAME:$CONTAINER_WORK_DIR/." "$WORK_DIR/"
else
  echo "Container $CONTAINER_NAME not found, fallback to docker run with image $IMAGE"
  docker run --rm \
    -v "$WORK_DIR:/workspace" \
    -w /workspace \
    "$IMAGE" \
    bash -lc "$(run_compile_commands)"
fi
