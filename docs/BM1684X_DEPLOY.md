# BM1684X Deployment Guide

这份文档按已经实际跑通的流程整理，覆盖：

- `wcirq-pc` 上训练和编译
- `my_sophgo` 容器中编译 `bmodel`
- BM1684X 设备/容器内部署与验证

如果你只想复制运行命令，直接看：

- [BM1684X_RUN_COMMANDS.md](/Users/yangyinqi/Documents/个人/简创空间/2-列车/CompiledVersionPC_20260419/docs/BM1684X_RUN_COMMANDS.md)

## 1. 最终需要的文件

真正部署到板端推理时，只需要这 3 个模型文件：

- `memory_model.pt`
- `backbone_1x3x640x640_bm1684x_f16.bmodel`
- `vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel`

其余这些文件都只是导出、编译、校验中间产物，板端推理不需要：

- `.mlir`
- `_origin.mlir`
- `_final.mlir`
- `.npz`
- `.json`
- `.profile`
- `.prototxt`
- `_weight.npz`
- `.ref_files.json`

除了模型文件，还必须带完整代码目录，尤其是 `engine/` 目录里的 Python 文件。实际部署时如果 `engine/` 是空目录，`main.py` 会报 `ImportError`。

## 2. `memory_model.pt` 的含义

`memory_model.pt` 不是单独的 ResNet 权重，也不是“只有向量”的纯缓存文件。它是这个项目的部署产物，内部包含：

- `memory_bank`
- `project_matrix`
- 推荐阈值和热力图统计
- 运行配置

板端推理必须有这个文件。

## 3. 在 `wcirq-pc` 上训练 `memory_model.pt`

在服务器的 PyTorch 环境中训练，不在 BM1684X 板子上训练。

```bash
cd ~/yyq/patchcore

export TORCH_HOME=~/yyq/patchcore/.torch
mkdir -p "$TORCH_HOME/hub/checkpoints"
cp /home/wcirq/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth \
  "$TORCH_HOME/hub/checkpoints/resnet50-11ad3fa6.pth"

python3 main.py train \
  --train_dir templates \
  --save_model memory_model.pt \
  --input_size 640 640 \
  --crop_size 640 640 \
  --stride 512 512 \
  --target_embed_dimension 1024
```

这里使用的 `resnet50-11ad3fa6.pth` 是 TorchVision `ResNet50_Weights.IMAGENET1K_V2` 对应的权重。

## 4. 导出 ONNX

### 4.1 导出 backbone

```bash
cd ~/yyq/patchcore

python3 tools/export_backbone_onnx.py \
  --weights_path /home/wcirq/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth \
  --output bm1684x_build/backbone.onnx \
  --batch_size 1 \
  --height 640 \
  --width 640
```

### 4.2 导出 vector GEMM

```bash
python3 tools/export_vector_gemm_onnx.py \
  --output bm1684x_build/vector_gemm.onnx \
  --query_rows 1600 \
  --database_rows 2048 \
  --embed_dim 1024
```

### 4.3 生成 TPU-MLIR 校验输入和参考输出

不要再手工只生成随机输入。`model_transform.py --test_result` 和 `model_deploy.py --test_reference` 都需要参考输出。

```bash
python3 tools/prepare_bm1684x_test_data.py \
  --work_dir bm1684x_build \
  --weights_path /home/wcirq/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth \
  --batch_size 1 \
  --height 640 \
  --width 640 \
  --query_rows 1600 \
  --database_rows 2048 \
  --embed_dim 1024
```

这个脚本会同时生成：

- `bm1684x_build/backbone_ref_input.npz`
- `bm1684x_build/backbone_top_outputs.npz`
- `bm1684x_build/vector_gemm_inputs.npz`
- `bm1684x_build/vector_gemm_top_outputs.npz`

## 5. 编译 `bmodel`

### 5.1 使用已有 `my_sophgo` 容器

你当前已经有容器：

```bash
docker ps -a
```

里面对应的容器名是：

```text
my_sophgo
```

直接执行：

```bash
cd ~/yyq/patchcore
TPUC_DEV_CONTAINER=my_sophgo bash tools/compile_bm1684x_bmodels.sh bm1684x_build
```

预期输出至少包含：

- `bm1684x_build/backbone_1x3x640x640_bm1684x_f16.bmodel`
- `bm1684x_build/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel`

### 5.2 编译时终端看起来“卡住”

这是正常现象。你可能会看到类似日志：

```text
Run TimeStepCombinePass
Run GroupDataMoveOverlapPass
GmemAllocator use OpSizeOrderAssign
in cmodel, enable profile.
```

这通常表示 TPU-MLIR 正在：

- 编译 `mlir -> bmodel`
- 跑 cmodel 校验
- 生成 profile

不一定是卡死。只要最后生成了两个 `.bmodel` 文件，就说明编译成功。

## 6. 板端或推理容器环境

板子只做推理，建议环境里安装：

- `python3`
- `numpy`
- `opencv-python`
- `pillow`
- `tqdm`
- `torch` CPU 版
- `sophon-sail`

如果还要保留 PyTorch backbone 回退路径，额外安装：

- `torchvision`

你实际使用的容器启动方式如下：

```bash
docker run -it --privileged \
  --name patchcore_bm \
  -v /data:/data \
  -v /opt/sophon:/opt/sophon:ro \
  -e LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:/opt/sophon/sophon-ffmpeg-latest/lib:/opt/sophon/sophon-opencv-latest/lib \
  -w /data/patchcore_deploy \
  bm_py310:sail310 bash
```

## 7. 板端部署目录建议

推荐目录结构：

```text
/data/patchcore_deploy/
├── main.py
├── engine/
├── models/
│   ├── memory_model.pt
│   ├── backbone_1x3x640x640_bm1684x_f16.bmodel
│   └── vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel
├── test_imgs/
└── templates/
```

其中 `engine/` 目录至少要包含这些文件：

- `engine/__init__.py`
- `engine/runtime.py`
- `engine/backbone.py`
- `engine/backbone_bm.py`
- `engine/indexing.py`
- `engine/indexing_bm.py`
- `engine/augment.py`
- `engine/utils.py`

## 8. 验证顺序

建议按下面顺序验证，不要一上来直接跑全 TPU：

1. CPU 基线
2. BM KNN
3. BM backbone + BM KNN
4. BM 全链路阈值标定
5. 带阈值批量测试

### 8.1 CPU 基线

```bash
python3 main.py detect \
  --device cpu \
  --model_path ./models/memory_model.pt \
  --input ./test_imgs/1.jpg \
  --output ./output
```

### 8.2 BM KNN

```bash
python3 main.py detect \
  --device cpu \
  --model_path ./models/memory_model.pt \
  --input ./test_imgs/1.jpg \
  --output ./output_bm_knn \
  --knn_backend bm \
  --bm_bmodel_path ./models/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel \
  --bm_device_id 0 \
  --bm_db_chunk_size 2048
```

### 8.3 BM 全链路

```bash
python3 main.py detect \
  --device cpu \
  --model_path ./models/memory_model.pt \
  --input ./test_imgs/1.jpg \
  --output ./output_bm_full \
  --input_size 640 640 \
  --crop_size 640 640 \
  --stride 512 512 \
  --detect_batch_size 1 \
  --backbone_backend bm \
  --backbone_bmodel_path ./models/backbone_1x3x640x640_bm1684x_f16.bmodel \
  --knn_backend bm \
  --bm_bmodel_path ./models/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel \
  --bm_device_id 0 \
  --bm_db_chunk_size 2048
```

你这次实际跑出来的结果是：

- CPU backbone + BM KNN：`26.962603`
- BM backbone + BM KNN：`23.691921`

这是可接受的漂移，原因通常是 BM backbone 使用 `F16`，而 PyTorch CPU 路径一般是 `FP32`。

### 8.4 BM 全链路阈值标定

```bash
python3 main.py calibrate_threshold \
  --device cpu \
  --model_path ./models/memory_model.pt \
  --input ./templates \
  --input_size 640 640 \
  --crop_size 640 640 \
  --stride 512 512 \
  --detect_batch_size 1 \
  --backbone_backend bm \
  --backbone_bmodel_path ./models/backbone_1x3x640x640_bm1684x_f16.bmodel \
  --knn_backend bm \
  --bm_bmodel_path ./models/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel \
  --bm_device_id 0 \
  --bm_db_chunk_size 2048 \
  --quantile 0.99
```

你这次实际标定得到的推荐阈值是：

```text
22.225254
```

### 8.5 批量跑 `test_imgs`

```bash
python3 main.py detect_batch \
  --device cpu \
  --model_path ./models/memory_model.pt \
  --input ./test_imgs \
  --output ./output_batch_bm \
  --threshold 22.225254 \
  --input_size 640 640 \
  --crop_size 640 640 \
  --stride 512 512 \
  --detect_batch_size 1 \
  --backbone_backend bm \
  --backbone_bmodel_path ./models/backbone_1x3x640x640_bm1684x_f16.bmodel \
  --knn_backend bm \
  --bm_bmodel_path ./models/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel \
  --bm_device_id 0 \
  --bm_db_chunk_size 2048
```

结果在：

- `./output_batch_bm/detection_results.json`
- `./output_batch_bm/*_overlay.jpg`

## 9. 常见现象和说明

### 9.1 `open usercpu.so, init user_cpu_init`

这通常不是错误，是 `libsophon` 初始化 user CPU layer 的日志。只要命令正常结束并输出分数，就不用单独处理。

### 9.2 `Premature end of JPEG file`

这表示某张 JPEG 尾部不完整，但 OpenCV 仍然把图读出来了。你的标定流程虽然跑完了，但正式部署前最好把这类损坏图替换掉。

### 9.3 板端推理只需要 3 个模型文件

再次强调，板端推理不需要把整个 `bm1684x_build` 目录原样带过去，只需要：

- `memory_model.pt`
- `backbone_1x3x640x640_bm1684x_f16.bmodel`
- `vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel`

## 10. 参考

- [TorchVision ResNet50 权重定义](https://docs.pytorch.org/vision/2.0/_modules/torchvision/models/resnet.html)
- [TPU-MLIR User Interface](https://doc.sophgo.com/sdk-docs/v23.05.01/docs_latest_release/docs/tpu-mlir/developer_manual_en/html/03_user_interface.html)
