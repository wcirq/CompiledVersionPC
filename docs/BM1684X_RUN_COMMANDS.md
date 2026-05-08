# BM1684X Run Commands

这份文档只整理板端/容器内实际运行用到的命令。

适用前提：

- 当前工作目录：`/data/patchcore_deploy`
- 模型目录：`./models`
- 测试图片目录：`./test_imgs`
- 当前编译好的 TPU 模型：
  - `./models/backbone_1x3x640x640_bm1684x_f16.bmodel`
  - `./models/vector_gemm_q1600_n2048_d1024_bm1684x_f16.bmodel`

## 1. CPU 基线

```bash
python3 main.py detect \
  --device cpu \
  --model_path ./models/memory_model.pt \
  --input ./test_imgs/1.jpg \
  --output ./output
```

## 2. BM KNN 测试

主干仍走 PyTorch/CPU，只把向量相似度检索切到 BM1684X。

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

## 3. BM 全链路测试

主干特征提取和向量检索都切到 BM1684X。

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

## 4. BM 全链路阈值标定

使用正常样本目录 `./templates` 标定阈值。

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

本次标定得到的推荐阈值是：

```text
22.225254
```

如果后续重新训练或重新标定，请替换成新的阈值。

## 5. 用当前阈值批量跑 `test_imgs`

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

## 6. 查看批量结果

直接查看结果 JSON：

```bash
cat ./output_batch_bm/detection_results.json
```

按图片打印分数和结论：

```bash
python3 - <<'PY'
import json
with open("./output_batch_bm/detection_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
for path, item in data.items():
    print(f"{path} | score={item.get('score')} | is_anomaly={item.get('is_anomaly')} | error={item.get('error')}")
PY
```
