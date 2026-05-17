# 列车顶视觉检测平台

`store/` 是一个“列车顶部异物检测”封装库，目标是把训练、检测、模型版本管理、样本向量库管理、HTTP 服务和管理后台整合成一套可直接部署、也可直接 `import` 调用的工具。

它目前包含两条能力线：

- `store_core`
  训练和使用列车顶部异物检测模型
- `store_infer`
  调用包内置的固定权重检测模型，例如火焰/烟雾检测

## 目录结构

核心目录如下：

- `src/store_core/`
  异物检测核心库
- `src/store_service/`
  FastAPI 服务层
- `src/store_web/`
  Web 管理后台
- `src/store_infer/`
  内置固定权重模型推理模块
- `store_data/`
  示例模型仓库目录

## 整体架构

```text
                    +-----------------------------+
                    |        用户调用入口         |
                    |-----------------------------|
                    | 1. import store_core        |
                    | 2. import store_infer       |
                    | 3. Web / HTTP               |
                    +--------------+--------------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
     +-----------------------------+   +-----------------------------+
     |   TrainRoofAnomalyStore     |   |      InferenceRegistry      |
     |   store_core.platform       |   |       store_infer           |
     +--------------+--------------+   +--------------+--------------+
                    |                                 |
                    v                                 v
     +-----------------------------+   +-----------------------------+
     |      ModelStoreManager      |   |      Ultralytics YOLO       |
     |  模型、版本、样本、阈值管理  |   |      固定权重推理封装       |
     +------+-------------+--------+   +-----------------------------+
            |             |
            v             v
   +----------------+   +----------------------+
   | TrainRoof      |   | VisionMemoryEngine   |
   | Segmenter      |   | 异物检测核心引擎      |
   | 顶部分割        |   | 特征提取/记忆库/KNN   |
   +----------------+   +----------------------+
```

## 核心概念

### 异物检测模型训练

这里的“训练”不是重新训练一个大模型，而是：

1. 自动分割图片中的列车顶部区域
2. 裁剪并清洗顶部区域样本
3. 提取特征并构建正常样本记忆库
4. 自动校准推荐阈值 `threshold`

训练结果本质上是：

- 一套正常样本向量库
- 一个默认阈值
- 一组可追溯的模型版本与样本文件

### 异物检测

异物检测阶段会：

1. 对待检图自动分割列车顶部
2. 把顶部区域送入异常检测引擎
3. 输出热力图、全局分数、异常区域轮廓和外接框

### 内置检测模型

`store_infer` 用于管理固定权重模型，不走 `store_data` 的模型仓库机制。

当前已接入：

- `fire_smoke`

后续新增同类模型时，通常只需要：

1. 放入权重文件
2. 在 `src/store_infer/builtin_models.py` 增加一条配置

## 安装与打包

推荐交付方式：

1. 在源码环境里构建 `.whl`
2. 把 `.whl` 发到目标环境
3. 在目标环境基于 wheel 安装和运行

### 构建 wheel

在 `store/` 目录执行：

```bash
pip install -U pip build
python -m build
```

生成文件位于：

```text
dist/
├── train_roof_anomaly_store-0.1.0-py3-none-any.whl
└── train_roof_anomaly_store-0.1.0.tar.gz
```

### 基于 wheel 安装

```bash
pip install train_roof_anomaly_store-0.1.0-py3-none-any.whl
```

或：

```bash
pip install /path/to/train_roof_anomaly_store-0.1.0-py3-none-any.whl
```

### 开发环境安装

如果当前是源码开发环境，可以直接：

```bash
pip install -e .
```

如果需要 `faiss-cpu`：

```bash
pip install -e .[faiss]
```

### BM 设备说明

- `sophon.sail` 不随 `pip` 自动安装
- 需要在 BM 环境里单独安装运行时
- 检测后端会自动按 `bm -> cuda -> cpu` 顺序选择
- 分割模型、ResNet 特征提取和向量检索都遵循同一套自动选择策略

## Python 调用

### 1. 异物检测模型管理与检测

```python
from store_core import TrainRoofAnomalyStore
from store_core.schemas import RuntimeOptions

store = TrainRoofAnomalyStore(
    root_dir="./store_data",
    autostart_service=False,
)

print(store.list_models())

options = RuntimeOptions(
    crop_size=(640, 640),
    stride=(512, 512),
    threshold_quantile=0.99,
)

train_result = store.train_model(
    model_name="木板车顶模型",
    image_dir="templates",
    runtime_options=options,
)

detect_result = store.detect_image(
    model_id=train_result["model_id"],
    image_path="test_imgs/2.jpg",
    include_heatmap_base64=True,
)

print(detect_result)
```

### 2. 内置检测模型

```python
from store_infer import list_models, run_inference

print(list_models())

result = run_inference(
    model_name="fire_smoke",
    image_path="test.jpg",
    conf_threshold=0.25,
    iou_threshold=0.45,
    imgsz=640,
    max_det=100,
    include_visualization_base64=True,
)

print(result)
```

返回结构示例：

```python
{
    "model_name": "fire_smoke",
    "backend": "ultralytics",
    "task_type": "detect",
    "image_width": 1920,
    "image_height": 1080,
    "count": 1,
    "detections": [
        {
            "class_id": 0,
            "class_name": "fire",
            "confidence": 0.93,
            "box": [100, 120, 220, 300],
        }
    ],
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    "imgsz": 640,
    "max_det": 100,
    "visualization_base64": "...",
}
```

## Web 服务启动

安装后可直接执行：

```bash
store_web
```

也支持：

```bash
python -m store_web
```

### 常见启动示例

默认启动：

```bash
store_web
```

指定端口和数据目录：

```bash
store_web --host 0.0.0.0 --port 55666 --root-dir ./store_data
```

指定顶部分割参数：

```bash
store_web \
  --host 0.0.0.0 \
  --port 55555 \
  --root-dir ./store_data \
  --yolo-conf-threshold 0.7 \
  --yolo-device cuda:0
```

查看帮助：

```bash
store_web --help
```

### 主要启动参数

- `--host`
  服务监听地址，默认 `0.0.0.0`
- `--port`
  服务端口，默认 `55555`
- `--root-dir`
  模型仓库目录，默认 `./store_data`
- `--yolo-weight-path`
  自定义顶部分割权重路径，不传则使用库内置权重
- `--yolo-conf-threshold`
  顶部分割置信度阈值，默认 `0.8`
- `--yolo-device`
  顶部分割所用设备，例如 `cpu`、`cuda:0`

默认访问地址：

```text
http://127.0.0.1:55555
```

## HTTP 接口

下面的示例默认服务地址为：

```text
http://127.0.0.1:55555
```

### 1. 健康检查

参数说明：

- 无额外参数

```bash
curl http://127.0.0.1:55555/api/health
```

返回结果说明：

- `status`
  固定为 `ok`，表示服务可用

### 2. 获取业务模型列表

参数说明：

- 无额外参数

```bash
curl http://127.0.0.1:55555/api/models
```

返回结果说明：

- `items`
  模型列表
- `items[].model_id`
  模型 ID
- `items[].model_name`
  模型名称
- `items[].current_version_id`
  当前版本 ID
- `items[].versions`
  版本列表
- `items[].storage_status`
  当前模型样本文件是否完整、是否可查看

### 3. 获取单个业务模型详情

参数说明：

- `model_xxxxx`
  路径中的 `model_id`，替换为实际模型 ID

```bash
curl http://127.0.0.1:55555/api/models/model_xxxxx
```

返回结果说明：

- `model_id`
  模型 ID
- `model_name`
  模型名称
- `current_version_id`
  当前版本 ID
- `versions`
  模型全部版本信息
- `versions[].threshold`
  该版本默认阈值
- `versions[].runtime_options`
  训练和检测相关运行参数
- `storage_status`
  样本文件、派生文件是否完整

### 4. 训练异物检测模型

参数说明：

- `model_name`
  业务模型名称
- `image_dir`
  训练图片目录
- `save_root_dir`
  可选，模型保存目录；不传则保存到服务当前 `root_dir`
- `calibrate_dir`
  可选，独立校准集目录
- `runtime_options.crop_size`
  训练裁剪尺寸，格式为 `[w, h]`
- `runtime_options.stride`
  滑窗步长，格式为 `[w, h]`
- `runtime_options.threshold_quantile`
  自动阈值分位数

```bash
curl -X POST http://127.0.0.1:55555/api/train \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "木板车顶模型",
    "image_dir": "templates",
    "save_root_dir": null,
    "calibrate_dir": null,
    "runtime_options": {
      "crop_size": [640, 640],
      "stride": [512, 512],
      "threshold_quantile": 0.99
    }
  }'
```

返回结果说明：

- `model_id`
  新建模型 ID
- `model_name`
  模型名称
- `current_version_id`
  当前训练产出的版本 ID
- `threshold`
  自动校准得到的默认阈值
- `sample_count`
  成功纳入训练的样本数
- `failed_images`
  训练失败图片列表
- `progress_events`
  训练过程中的阶段事件

### 5. 异物检测

参数说明：

- `model_id`
  必填，业务模型 ID
- `image_file`
  上传待检图片文件，和 `image_path` 二选一
- `image_path`
  服务器本地图片路径，和 `image_file` 二选一
- `threshold`
  可选，临时覆盖模型默认阈值
- `include_heatmap_base64`
  是否返回热力图 base64
- `heatmap_include_background`
  热力图是否叠加原图背景
- `heatmap_zero_below_threshold`
  是否将阈值以下热力值直接置零

上传图片文件：

```bash
curl -X POST http://127.0.0.1:55555/api/detect \
  -F model_id=model_xxxxx \
  -F image_file=@test_imgs/2.jpg \
  -F include_heatmap_base64=true \
  -F heatmap_include_background=true \
  -F heatmap_zero_below_threshold=true
```

传本地图片路径：

```bash
curl -X POST http://127.0.0.1:55555/api/detect \
  -F model_id=model_xxxxx \
  -F image_path=/path/to/test.jpg \
  -F threshold=0.82
```

返回结果说明：

- `model_id`
  使用的模型 ID
- `model_name`
  模型名称
- `threshold`
  本次实际使用的阈值
- `score`
  当前图像总体异常分数
- `is_anomaly`
  是否判定为异常
- `roof_contours`
  识别出的列车顶部轮廓列表
- `anomaly_regions`
  异常区域列表
- `anomaly_regions[].contour`
  异常区域轮廓
- `anomaly_regions[].box`
  异常区域外接框 `[x1, y1, x2, y2]`
- `anomaly_regions[].score`
  异常区域得分
- `heatmap_base64`
  可选，热力图 base64；仅在 `include_heatmap_base64=true` 时返回
- `message`
  无顶部轮廓等特殊情况下的提示信息

### 6. 提取列车顶部轮廓

参数说明：

- `image_file`
  上传图片文件，和 `image_path` 二选一
- `image_path`
  服务器本地图片路径，和 `image_file` 二选一

```bash
curl -X POST http://127.0.0.1:55555/api/extract-contours \
  -F image_file=@test_imgs/2.jpg
```

或：

```bash
curl -X POST http://127.0.0.1:55555/api/extract-contours \
  -F image_path=/path/to/test.jpg
```

返回结果说明：

- `items`
  识别出的顶部区域列表
- `items[].contour`
  顶部轮廓点列表
- `items[].bbox`
  顶部外接框 `[x1, y1, x2, y2]`
- `items[].confidence`
  识别置信度

### 7. 更新模型默认阈值

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `threshold`
  新的默认阈值

```bash
curl -X PATCH http://127.0.0.1:55555/api/models/model_xxxxx/threshold \
  -H 'Content-Type: application/json' \
  -d '{"threshold": 0.82}'
```

返回结果说明：

- `model_id`
  模型 ID
- `threshold`
  更新后的默认阈值
- `message`
  更新结果说明

### 8. 删除模型

参数说明：

- `model_xxxxx`
  路径中的 `model_id`

```bash
curl -X DELETE http://127.0.0.1:55555/api/models/model_xxxxx
```

返回结果说明：

- `model_id`
  已删除模型 ID
- `message`
  删除结果说明

### 9. 精简模型文件

参数说明：

- `model_xxxxx`
  路径中的 `model_id`

```bash
curl -X POST http://127.0.0.1:55555/api/models/model_xxxxx/prune-assets
```

返回结果说明：

- `model_id`
  模型 ID
- `message`
  精简结果说明
- `removed_paths`
  被删除的文件或目录列表

### 10. 获取样本列表

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `page`
  页码，从 `1` 开始
- `page_size`
  每页样本数，建议不超过 `200`

```bash
curl "http://127.0.0.1:55555/api/models/model_xxxxx/samples?page=1&page_size=20"
```

返回结果说明：

- `items`
  当前页样本列表
- `items[].sample_id`
  样本 ID
- `items[].source_image_name`
  来源图片名
- `items[].bbox`
  样本框
- `items[].contour`
  样本轮廓
- `page`
  当前页码
- `page_size`
  每页数量
- `total`
  样本总数

### 11. 获取样本详情

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `sample_xxxxx`
  路径中的 `sample_id`

```bash
curl http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx
```

返回结果说明：

- `sample`
  样本基础信息
- `sample.sample_id`
  样本 ID
- `sample.contour`
  当前轮廓
- `sample.note`
  备注
- `file_status`
  当前样本原图、处理图、子图是否齐全
- `tiles`
  子图列表
- `tiles[].tile_id`
  子图 ID
- `tiles[].enabled`
  子图是否启用

### 12. 获取样本图片

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `sample_xxxxx`
  路径中的 `sample_id`
- `kind`
  图片类型，可选 `processed` 或 `raw`

处理图：

```bash
curl -o sample_processed.jpg \
  "http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx/image?kind=processed"
```

原图：

```bash
curl -o sample_raw.jpg \
  "http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx/image?kind=raw"
```

返回结果说明：

- 直接返回图片文件流，不是 JSON

### 13. 获取样本子图

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `sample_xxxxx`
  路径中的 `sample_id`
- `tile_xxxxx`
  路径中的 `tile_id`

```bash
curl -o tile.jpg \
  "http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx/tiles/tile_xxxxx/image"
```

返回结果说明：

- 直接返回图片文件流，不是 JSON

### 14. 扫描样本库异常

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `threshold`
  可选，临时使用的阈值；不传则使用模型默认阈值

```bash
curl -X POST http://127.0.0.1:55555/api/models/model_xxxxx/scan-samples \
  -H 'Content-Type: application/json' \
  -d '{"threshold": 0.82}'
```

返回结果说明：

- `model_id`
  模型 ID
- `threshold`
  本次扫描使用的阈值
- `items`
  样本扫描结果列表
- `items[].sample_id`
  样本 ID
- `items[].score`
  样本异常分数
- `items[].is_anomaly`
  是否判定为异常

### 15. 删除样本

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `sample_xxxxx`
  路径中的 `sample_id`

```bash
curl -X DELETE http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx
```

返回结果说明：

- `model_id`
  模型 ID
- `sample_id`
  已删除样本 ID
- `message`
  删除结果说明

### 16. 更新样本轮廓与子图启停

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `sample_xxxxx`
  路径中的 `sample_id`
- `contour`
  样本轮廓，二维点列表，例如 `[[x1,y1],[x2,y2],...]`
- `note`
  可选，样本备注
- `enabled_tile_ids`
  当前仍启用的子图 ID 列表

```bash
curl -X PATCH http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx \
  -H 'Content-Type: application/json' \
  -d '{
    "contour": [[100, 100], [500, 100], [500, 280], [100, 280]],
    "note": "手工修正轮廓",
    "enabled_tile_ids": ["tile_001", "tile_002"]
  }'
```

返回结果说明：

- `model_id`
  模型 ID
- `sample_id`
  样本 ID
- `sample`
  更新后的样本信息
- `message`
  更新结果说明

### 17. 仅更新子图启停

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `sample_xxxxx`
  路径中的 `sample_id`
- `enabled_tile_ids`
  需要保留启用状态的子图 ID 列表

```bash
curl -X POST http://127.0.0.1:55555/api/models/model_xxxxx/samples/sample_xxxxx/tiles \
  -H 'Content-Type: application/json' \
  -d '{
    "enabled_tile_ids": ["tile_001", "tile_002"]
  }'
```

返回结果说明：

- `model_id`
  模型 ID
- `sample_id`
  样本 ID
- `enabled_tile_ids`
  当前启用的子图 ID 列表
- `message`
  保存结果说明

### 18. 追加正样本

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `image_file`
  上传图片文件，和 `image_path` 二选一
- `image_path`
  服务器本地图片路径，和 `image_file` 二选一
- `contour_json`
  可选，手工指定轮廓；不传则自动提取列车顶部
- `note`
  可选，样本备注
- `append_max_vectors`
  单次向量追加上限

```bash
curl -X POST http://127.0.0.1:55555/api/models/model_xxxxx/samples \
  -F image_file=@test_imgs/2.jpg \
  -F note="新增正常样本" \
  -F append_max_vectors=20
```

返回结果说明：

- `model_id`
  模型 ID
- `sample_id`
  新增样本 ID
- `message`
  追加结果说明
- `sample`
  新增样本信息

如果要显式传轮廓：

```bash
curl -X POST http://127.0.0.1:55555/api/models/model_xxxxx/samples \
  -F image_file=@test_imgs/2.jpg \
  -F contour_json='[[100,100],[500,100],[500,280],[100,280]]' \
  -F note="指定轮廓追加" \
  -F append_max_vectors=20
```

### 19. 获取导出摘要

参数说明：

- `model_xxxxx`
  路径中的 `model_id`

```bash
curl http://127.0.0.1:55555/api/models/model_xxxxx/export-summary
```

返回结果说明：

- `model_id`
  模型 ID
- `full`
  完整模型包摘要
- `full.file_count`
  完整包文件数
- `full.total_size_bytes`
  完整包总字节数
- `full.items`
  完整包包含的文件列表
- `deploy`
  部署模型包摘要
- `deploy.file_count`
  部署包文件数
- `deploy.total_size_bytes`
  部署包总字节数
- `deploy.items`
  部署包包含的文件列表

### 20. 创建模型导出任务

参数说明：

- `model_xxxxx`
  路径中的 `model_id`
- `deployment_only`
  是否只导出部署所需关键文件；`false` 为完整包，`true` 为部署包

完整模型包：

```bash
curl -X POST "http://127.0.0.1:55555/api/models/model_xxxxx/export?deployment_only=false"
```

部署模型包：

```bash
curl -X POST "http://127.0.0.1:55555/api/models/model_xxxxx/export?deployment_only=true"
```

返回结果说明：

- `task_id`
  导出任务 ID
- `model_id`
  模型 ID
- `deployment_only`
  是否为部署包导出

### 21. 查询导出任务状态

参数说明：

- `export_xxxxx`
  路径中的导出任务 `task_id`

```bash
curl http://127.0.0.1:55555/api/model-export-tasks/export_xxxxx
```

返回结果说明：

- `task_id`
  导出任务 ID
- `model_id`
  模型 ID
- `status`
  任务状态，例如 `pending`、`running`、`ready`、`error`
- `progress`
  当前进度百分比
- `message`
  当前状态说明
- `filename`
  下载文件名
- `error`
  失败时的错误信息

### 22. 下载导出文件

参数说明：

- `export_xxxxx`
  路径中的导出任务 `task_id`

```bash
curl -L -o model_export.zip \
  http://127.0.0.1:55555/api/model-export-tasks/export_xxxxx/download
```

返回结果说明：

- 直接返回 zip 文件流，不是 JSON

### 23. 导入模型压缩包

参数说明：

- `model_file`
  导出的 zip 模型包

```bash
curl -X POST http://127.0.0.1:55555/api/models/import \
  -F model_file=@model_export.zip
```

返回结果说明：

- `model_id`
  导入后的模型 ID
- `model_name`
  导入后的模型名称
- `message`
  导入结果说明

### 24. 获取内置检测模型列表

参数说明：

- 无额外参数

```bash
curl http://127.0.0.1:55555/api/inference/models
```

返回结果说明：

- `items`
  内置检测模型列表
- `items[].name`
  模型名
- `items[].task_type`
  任务类型
- `items[].conf_threshold`
  默认置信度阈值
- `items[].iou_threshold`
  默认 IoU 阈值
- `items[].imgsz`
  默认输入尺寸
- `items[].max_det`
  默认最大检测框数

### 25. 获取单个内置检测模型配置

参数说明：

- `fire_smoke`
  路径中的内置模型名，可替换为其他已注册模型名

```bash
curl http://127.0.0.1:55555/api/inference/models/fire_smoke
```

返回结果说明：

- `name`
  模型名
- `backend`
  推理后端
- `task_type`
  任务类型
- `class_names`
  类别名称列表
- `conf_threshold`
  默认置信度阈值
- `iou_threshold`
  默认 IoU 阈值
- `imgsz`
  默认输入尺寸
- `max_det`
  默认最大检测框数

### 26. 调用内置检测模型

参数说明：

- `fire_smoke`
  路径中的内置模型名
- `image_file`
  上传图片文件，和 `image_path` 二选一
- `image_path`
  服务器本地图片路径，和 `image_file` 二选一
- `conf_threshold`
  可选，置信度阈值
- `iou_threshold`
  可选，NMS 的 IoU 阈值
- `imgsz`
  可选，推理输入尺寸，例如 `640`
- `max_det`
  可选，单张图最多保留的检测框数量
- `device`
  可选，推理设备，例如 `cpu`、`cuda:0`
- `include_visualization_base64`
  是否返回可视化结果图的 base64

上传图片文件：

```bash
curl -X POST http://127.0.0.1:55555/api/inference/fire_smoke \
  -F image_file=@test.jpg \
  -F conf_threshold=0.25 \
  -F iou_threshold=0.45 \
  -F imgsz=640 \
  -F max_det=100 \
  -F include_visualization_base64=true
```

传本地图片路径：

```bash
curl -X POST http://127.0.0.1:55555/api/inference/fire_smoke \
  -F image_path=/path/to/test.jpg \
  -F conf_threshold=0.25 \
  -F iou_threshold=0.45 \
  -F imgsz=640 \
  -F max_det=100
```

如果希望把返回的可视化结果直接保存成图片，可以结合 `jq` 和 `base64`：

```bash
curl -s -X POST http://127.0.0.1:55555/api/inference/fire_smoke \
  -F image_file=@test.jpg \
  -F conf_threshold=0.25 \
  -F iou_threshold=0.45 \
  -F imgsz=640 \
  -F max_det=100 \
  -F include_visualization_base64=true \
| jq -r '.visualization_base64' \
| base64 -d > fire_smoke_result.jpg
```

如果想同时保留完整 JSON 返回和结果图，可以先保存 JSON，再提取图片：

```bash
curl -s -X POST http://127.0.0.1:55555/api/inference/fire_smoke \
  -F image_file=@test.jpg \
  -F include_visualization_base64=true \
  > fire_smoke_result.json

cat fire_smoke_result.json | jq -r '.visualization_base64' | base64 -d > fire_smoke_result.jpg
```

返回结果说明：

- `model_name`
  调用的内置模型名
- `backend`
  推理后端
- `task_type`
  任务类型
- `image_width`
  输入图宽度
- `image_height`
  输入图高度
- `count`
  检测结果数量
- `detections`
  检测结果列表
- `detections[].class_id`
  类别 ID
- `detections[].class_name`
  类别名称
- `detections[].confidence`
  置信度
- `detections[].box`
  外接框 `[x1, y1, x2, y2]`
- `conf_threshold`
  本次实际使用的置信度阈值
- `iou_threshold`
  本次实际使用的 IoU 阈值
- `imgsz`
  本次实际使用的输入尺寸
- `max_det`
  本次实际使用的最大检测框数
- `visualization_base64`
  可选，可视化结果图 base64；仅在 `include_visualization_base64=true` 时返回

## Web 管理后台

默认打开：

```text
http://127.0.0.1:55555/
```

页面中主要有两类能力：

- 异物检测模型管理
  训练、查看、导入导出、样本维护、检测
- 检测测试
  调用内置固定权重模型并查看可视化结果
