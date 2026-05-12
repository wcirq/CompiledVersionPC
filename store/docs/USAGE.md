# Store 库使用说明

## 1. 库的定位

`store/` 是一个“列车顶部异物检测”封装库，目标是把训练、检测、模型版本管理、样本向量库管理、HTTP 服务和管理后台整合成一套可直接部署、也可直接 `import` 调用的工具。

它主要解决两类问题：

- 训练异物检测模型
  输入一批正常样本图，先自动分割列车顶部，再建立正常特征记忆库，并自动校准默认阈值
- 对单张图片做异物检测
  输入待检图片，自动分割列车顶部，输出是否异常、异常区域、热力图和分数

---

## 2. 整体架构

### 2.1 目录结构

`store/` 目录核心内容如下：

- `pyproject.toml`
  打包配置、命令行入口配置
- `src/store_core/`
  核心库
- `src/store_service/`
  FastAPI 服务层
- `src/store_web/`
  Web 管理后台与 `python -m store_web` 模块入口
- `src/store_core/assets/weights/train_roof_yolo11n_best.pt`
  内置 YOLO 列车顶部分割权重
- `docs/USAGE.md`
  当前说明文档

### 2.2 架构图

```text
                    +-----------------------------+
                    |        用户调用入口         |
                    |-----------------------------|
                    | 1. import store_core        |
                    | 2. Web/HTTP                 |
                    | 3. store_web / CLI          |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    |   TrainRoofAnomalyStore     |
                    |   store_core.platform        |
                    +--------------+--------------+
                                   |
                                   v
                    +-----------------------------+
                    |      ModelStoreManager      |
                    |  模型、版本、样本、阈值管理  |
                    +------+-------------+--------+
                           |             |
                           |             |
                           v             v
               +----------------+   +----------------------+
               | TrainRoof      |   | VisionMemoryEngine   |
               | Segmenter      |   | 异物检测核心引擎      |
               | YOLO 顶部分割   |   | 特征提取/记忆库/KNN   |
               +----------------+   +----------------------+
                           |             |
                           +------+------+ 
                                  |
                                  v
                    +-----------------------------+
                    |        store_data/          |
                    |-----------------------------|
                    | registry.json               |
                    | models/<model_id>/          |
                    |   model.json                |
                    |   versions/<version_id>/    |
                    |     version.json            |
                    |     samples.json            |
                    |     memory_model.pt         |
                    |     raw/ processed/ tiles/  |
                    +-----------------------------+
```

### 2.3 Web 服务架构图

```text
浏览器
  |
  v
store_web/static
  |
  v
store_service.api (FastAPI)
  |
  v
TrainRoofAnomalyStore / ModelStoreManager
  |
  v
YOLO 顶部分割 + 异物检测引擎 + 本地模型仓库
```

---

## 3. 核心概念

### 3.1 训练的作用

这里的“训练”不是重新训练一个大模型，而是：

1. 用 YOLO 自动分割出图片中的列车顶部区域
2. 把顶部区域裁剪、清洗成训练样本
3. 提取特征并构建正常样本记忆库
4. 根据训练集或校准集自动计算一个推荐阈值 `threshold`

训练结果本质上是：

- 一套正常样本向量库
- 一个默认阈值
- 一组可追溯的样本与版本文件

### 3.2 异物检测的作用

“异物检测”阶段的目标是：

1. 对待检图自动分割列车顶部
2. 把顶部区域送入异常检测引擎
3. 计算热力图和全局分数
4. 按阈值输出异常区域轮廓、外接框和异常判定

### 3.3 模型与版本

一个业务模型包含：

- `model_id`
- `model_name`
- `current_version_id`
- 多个版本记录

一个版本包含：

- 当前版本的默认 `threshold`
- 运行参数 `runtime_options`
- 当前版本样本列表
- 当前版本记忆库文件 `memory_model.pt`

---

## 4. 安装与打包

推荐的交付和部署方式是：

1. 在源码环境里构建 `.whl`
2. 把 `.whl` 发给目标环境
3. 在目标环境中基于 `.whl` 安装和使用

这样更接近真实交付方式，也更便于版本管理和离线部署。

### 4.1 构建 wheel

在 `store/` 目录执行：

```bash
pip install -U pip build
python -m build
```

生成文件位于：

```text
store/dist/
├── train_roof_anomaly_store-0.1.0-py3-none-any.whl
└── train_roof_anomaly_store-0.1.0.tar.gz
```

实际交付时，通常使用：

```text
train_roof_anomaly_store-0.1.0-py3-none-any.whl
```

### 4.2 基于 wheel 安装

把 `.whl` 拷贝到目标机器后执行：

```bash
pip install train_roof_anomaly_store-0.1.0-py3-none-any.whl
```

如果 wheel 文件不在当前目录，可以写完整路径：

```bash
pip install /path/to/train_roof_anomaly_store-0.1.0-py3-none-any.whl
```

安装完成后，就可以直接使用：

```bash
store_web --help
```

或者：

```bash
python -m store_web --help
```

### 4.3 基于 wheel 的典型交付流程

源码机上构建：

```bash
cd store
python -m build
```

目标机上安装：

```bash
pip install train_roof_anomaly_store-0.1.0-py3-none-any.whl
```

目标机上启动服务：

```bash
store_web --port 55555 --root-dir ./store_data
```

目标机上 Python 调用：

```python
from store_core import TrainRoofAnomalyStore

store = TrainRoofAnomalyStore(root_dir="./store_data", autostart_service=False)
print(store.list_models())
```

### 4.4 开发环境补充说明

如果你当前就在源码仓库里开发，而不是做交付安装，也可以直接：

```bash
pip install -e .
```

如果需要 `faiss-cpu`：

```bash
pip install -e .[faiss]
```

这一节只是开发补充，不是推荐的交付安装方式。

### 4.5 BM 设备说明

- `sophon.sail` 不随 `pip` 自动安装
- 需要在 BM 环境里自行安装运行时
- `RuntimeOptions` 中把 `knn_backend="bm"` 或 `backbone_backend="bm"` 即可启用 BM 相关逻辑

---

## 5. 使用方式一：Web 服务方式

## 5.1 内置 YOLO 推理模型

`store` 现在支持一组与 anomaly store 并列的“内置推理模型”能力，代码位于 `src/store_infer/`。

这套能力适合放置固定权重的 `YOLOv11` 任务模型，例如：

- 火焰 / 烟雾检测
- 其他后续新增的目标检测模型

当前内置模型通过 `store_infer/builtin_models.py` 注册，第一版已接入：

- `fire_smoke`

它使用包内权重文件：

```text
store/src/store_infer/assets/weights/fire-smoke_model.pt
```

### Python 调用

```python
from store_infer import list_models, run_inference

print(list_models())

result = run_inference(
    model_name="fire_smoke",
    image_path="test.jpg",
    conf_threshold=0.25,
    iou_threshold=0.45,
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
    "max_det": 100,
}
```

### HTTP 调用

获取当前内置推理模型列表：

```bash
curl http://127.0.0.1:55555/api/inference/models
```

对指定内置模型做推理：

```bash
curl -X POST http://127.0.0.1:55555/api/inference/fire_smoke \
  -F image_file=@test.jpg \
  -F conf_threshold=0.25 \
  -F iou_threshold=0.45 \
  -F max_det=100
```

也可以传本地路径：

```bash
curl -X POST http://127.0.0.1:55555/api/inference/fire_smoke \
  -F image_path=/path/to/test.jpg
```

适合场景：

- 需要浏览器管理页面
- 需要 HTTP 接口
- 需要给其他服务、前端或运维同学调用

### 5.1 直接启动命令

安装后可直接执行：

```bash
store_web
```

也支持：

```bash
python -m store_web
```

### 5.2 常见启动示例

默认启动：

```bash
store_web
```

指定端口和数据目录：

```bash
store_web --host 0.0.0.0 --port 55666 --root-dir ./store_data
```

指定 YOLO 参数：

```bash
store_web \
  --host 0.0.0.0 \
  --port 55555 \
  --root-dir ./store_data \
  --yolo-conf-threshold 0.7 \
  --yolo-device cuda:0
```

查看中文帮助：

```bash
store_web --help
```

### 5.3 模块启动参数说明

- `--host`
  服务监听地址，默认 `0.0.0.0`
- `--port`
  服务端口，默认 `55555`
- `--root-dir`
  模型仓库目录，默认 `./store_data`
- `--yolo-weight-path`
  自定义 YOLO 权重路径，不传则使用库内置权重
- `--yolo-conf-threshold`
  YOLO 顶部分割置信度阈值，默认 `0.8`
- `--yolo-device`
  YOLO 使用的设备，例如 `cpu`、`cuda:0`

### 5.4 启动后的访问方式

默认服务地址：

```text
http://127.0.0.1:55555
```

内容包括：

- `/`
  Web 管理后台
- `/api/models`
  模型列表接口
- `/api/train`
  训练接口
- `/api/detect`
  检测接口

### 5.5 常用 HTTP 接口

#### 5.5.1 训练接口

```text
POST /api/train
```

主要请求体字段：

- `model_name`
  业务模型名
- `image_dir`
  训练图片目录
- `save_root_dir`
  可选，模型保存目录
- `calibrate_dir`
  可选，单独校准集目录
- `runtime_options`
  训练运行参数

#### 5.5.2 异物检测接口

```text
POST /api/detect
```

表单字段：

- `model_id`
  模型 ID
- `image_file`
  上传图片文件
- `image_path`
  可选，服务器本地图片路径
- `threshold`
  可选，临时覆盖默认 threshold
- `include_heatmap_base64`
  是否返回热力图
- `heatmap_include_background`
  热力图是否叠加背景
- `heatmap_zero_below_threshold`
  是否把阈值以下热力值置零

#### 5.5.3 提取列车顶部轮廓接口

```text
POST /api/extract-contours
```

表单字段：

- `image_file`
  上传图片文件
- `image_path`
  可选，服务器本地图片路径

返回内容包括：

- `items`
  YOLO 识别出的顶部轮廓和框
- `preview_base64`
  预览图

#### 5.5.4 追加正样本接口

```text
POST /api/models/{model_id}/samples
```

表单字段：

- `image_file`
  上传图片文件
- `image_path`
  可选，服务器本地图片路径
- `contour_json`
  可选，轮廓 JSON；不传时会先自动做 YOLO 顶部提取
- `note`
  可选，备注
- `append_max_vectors`
  单次追加向量上限

说明：

- 这个接口用于把“正常样本”追加进当前模型的向量库
- 如果不传 `contour_json`，服务端会先自动提取列车顶部轮廓
- 如果传了 `contour_json`，服务端按你指定的轮廓追加

返回结果通常包含：

- `model_id`
- `version_id`
- `added_count`
- `items`
- `threshold`
- `added_vector_count`
- `append_max_vectors`

#### 5.5.5 修改默认 threshold 接口

```text
PATCH /api/models/{model_id}/threshold
```

JSON 请求体：

```json
{
  "threshold": 1.2
}
```

作用：

- 修改当前版本默认 threshold
- 后续 Web 检测和 Python 检测在不显式传参时都使用新值

### 5.6 Web 方式下的典型流程

1. 启动 `store_web`
2. 浏览器打开管理页面
3. 训练业务模型
4. 调整默认 threshold
5. 上传测试图做检测
6. 查看样本、编辑轮廓、追加正样本

---

## 6. 使用方式二：Python import 方式

适合场景：

- 你要把它嵌入现有 Python 项目
- 你只需要训练/检测能力，不一定要 Web 页面
- 你希望把流程和自己的业务逻辑串起来

### 6.1 初始化

```python
from store_core import TrainRoofAnomalyStore

store = TrainRoofAnomalyStore(
    root_dir="./store_data",
    autostart_service=False,
)
```

参数说明：

- `root_dir`
  模型仓库目录
- `autostart_service`
  是否自动在后台启动 Web/HTTP 服务
- `service_host`
  自动启动后台服务时的监听地址
- `service_port`
  自动启动后台服务时的起始端口，默认从 `55555` 开始
- `yolo_weight_path`
  自定义 YOLO 权重
- `yolo_conf_threshold`
  YOLO 顶部分割置信度阈值
- `yolo_device`
  YOLO 推理设备

### 6.2 如果要顺带启动后台服务

```python
store = TrainRoofAnomalyStore(
    root_dir="./store_data",
    autostart_service=True,
    service_port=55555,
)

print(store.service_info)
```

说明：

- `autostart_service=True` 会后台拉起 HTTP 服务
- 如果主程序不希望结束，可以再调用：

```python
store.serve_forever()
```

这里的 `serve_forever()` 只负责阻塞主线程，防止 Python 进程结束；后台服务本身还是由 `autostart_service=True` 启动。

---

## 7. 训练模型详解

### 7.1 最小示例

```python
from store_core import TrainRoofAnomalyStore
from store_core.schemas import RuntimeOptions

store = TrainRoofAnomalyStore(root_dir="./store_data", autostart_service=False)

def on_progress(event: dict):
    print(event)

options = RuntimeOptions(
    device="cuda",
    knn_backend="faiss",
    crop_size=(640, 640),
    stride=(512, 512),
    threshold_quantile=0.99,
)

result = store.train_model(
    model_name="木板车顶模型",
    image_dir="/data/train_images",
    runtime_options=options,
    calibrate_dir=None,
    progress_callback=on_progress,
)

print(result["model_id"])
```

### 7.2 训练过程做了什么

训练的大致流程如下：

```text
训练图片
  -> YOLO 分割列车顶部
  -> 生成 raw / processed 样本
  -> 切 tile
  -> 提取 embedding
  -> 建立 memory bank
  -> 用训练集或校准集校准 threshold
  -> 保存版本文件
```

### 7.2.1 推荐的训练与部署方式

建议把训练和部署分开：

1. 在高性能机器上训练
   例如 GPU 服务器、显存和存储更充足的训练机
2. 训练完成后导出模型
   导出当前模型对应的完整模型目录
3. 在部署环境中导入模型
   部署环境只负责推理、检测、Web 管理和样本维护

推荐这样做的原因：

- 训练阶段通常更吃 GPU、内存和磁盘 IO
- 部署环境通常更关注稳定运行，不适合承担训练负载
- 导出/导入模型后，可以把训练机和部署机彻底解耦
- 更适合正式交付、版本归档和多环境迁移

推荐流程：

```text
高性能训练机
  -> 训练模型
  -> 导出模型 zip
  -> 传输到部署机
  -> 部署机导入模型
  -> 开始异物检测服务
```

### 7.3 `train_model()` 参数说明

- `model_name`
  业务模型名，当前要求唯一
- `image_dir`
  训练图片目录
- `runtime_options`
  运行参数对象，控制特征提取、切图、阈值校准等
- `save_root_dir`
  可选，指定模型保存根目录；一般与初始化时的 `root_dir` 保持一致
- `calibrate_dir`
  可选，单独的阈值校准图片目录
- `progress_callback`
  可选，接收训练进度事件

### 7.4 训练输出结果说明

返回结果通常包含：

- `model_id`
- `model_name`
- `current_version_id`
- `versions`
- 当前版本阈值
- 失败图片数量与列表

### 7.5 常用训练参数说明

下面是最常用、最值得先理解的参数。

#### 设备与后端

- `device`
  异常检测主流程设备，例如 `cpu`、`cuda`
- `backbone_backend`
  主干特征提取后端，默认 `torch`
- `knn_backend`
  KNN 检索后端，常用 `auto`、`faiss`、`bm`

#### 切图与推理尺寸

- `input_size`
  主干网络输入尺寸
- `crop_size`
  滑窗裁剪尺寸，越大单块感受野越大
- `stride`
  滑窗步长，越小重叠越多，精度可能更高但速度更慢
- `infer_long_side`
  推理前是否按长边缩放；`0` 表示不强制缩放

#### 批量与性能

- `batch_size`
  训练阶段批大小
- `detect_batch_size`
  检测阶段滑窗批大小
- `use_amp`
  是否开启混合精度

#### 记忆库规模

- `memory_ratio`
  记忆库压缩比例，越小内存占用越低
- `target_embed_dimension`
  特征维度投影目标大小
- `max_embeddings`
  最大 embedding 数量限制

#### 阈值与热力图

- `threshold_quantile`
  用于从分数分布中计算推荐阈值的分位数
- `heatmap_std_scale`
  热力图显示范围的标准差放大倍数
- `heatmap_quantile`
  热力图显示范围上界的分位数
- `max_heatmap_samples`
  阈值校准时热力图采样上限
- `heatmap_zero_below_threshold`
  是否把阈值以下热力值直接置零，默认 `True`

#### 训练增强与在线压缩

- `train_crop_scale_range`
  训练裁剪缩放范围
- `train_crop_round_multiple`
  裁剪尺寸对齐倍数
- `train_min_crop_size`
  最小训练裁剪尺寸
- `stream_to_disk`
  是否把中间 embedding 流式落盘
- `online_compress_ratio`
  在线压缩比例
- `online_novelty_threshold`
  在线新颖性筛选阈值

---

## 8. 异物检测详解

### 8.1 最小示例

```python
result = store.detect_image(
    model_id="model_xxxxx",
    image_path="/data/test.jpg",
    include_heatmap_base64=True,
)

print(result)
```

直接传 `numpy.ndarray` 也支持：

```python
import cv2

image_bgr = cv2.imread("/data/test.jpg")

result = store.detect_image(
    model_id="model_xxxxx",
    image_bgr=image_bgr,
    include_heatmap_base64=True,
)
```

说明：

- `image_bgr` 适用于 Python `import` 方式
- 数组默认按 OpenCV 风格的 `BGR` 三通道图像解释
- 如果传入二维灰度图，内部会自动转成三通道 `BGR`

### 8.2 检测流程做了什么

```text
待检图片
  -> YOLO 分割列车顶部
  -> 对每个顶部区域做异常检测
  -> 合并热力图
  -> 按 threshold 提取异常区域
  -> 输出结果
```

### 8.3 `detect_image()` 参数说明

- `model_id`
  要使用的业务模型 ID
- `image_path`
  图片路径；与 `image_bytes` 二选一
- `image_bytes`
  图片二进制内容；适合接口或内存调用
- `image_bgr`
  `numpy.ndarray` 格式的图像数据；适合 Python 内部直接传 OpenCV 图像
- `include_heatmap_base64`
  是否返回 base64 编码热力图
- `threshold`
  可选，临时覆盖模型默认阈值
- `heatmap_include_background`
  热力图是否叠加原图背景
- `heatmap_zero_below_threshold`
  是否把阈值以下热力值直接清零；如果不传，则使用模型运行参数中的默认值

说明：

- `image_path`、`image_bytes`、`image_bgr` 三者传一个即可
- 如果同时传多个，优先顺序为：`image_bgr` -> `image_bytes` -> `image_path`

### 8.4 检测返回字段说明

- `model_id`
  检测使用的模型 ID
- `model_name`
  模型名
- `version_id`
  使用的模型版本 ID
- `threshold`
  本次检测实际使用的阈值
- `is_anomaly`
  是否判断为异常
- `score`
  全局分数
- `roof_contours`
  YOLO 提取的列车顶部轮廓
- `anomaly_regions`
  异常区域轮廓、框和分数
- `heatmap_include_background`
  热力图是否叠加背景
- `heatmap_zero_below_threshold`
  本次热力图是否启用阈值以下置零
- `heatmap_base64`
  可选，热力图 base64

### 8.5 检测结果如何理解

- `roof_contours`
  表示模型在哪些区域认为是列车顶部
- `anomaly_regions`
  表示在顶部区域内检测到的异常热点区域
- `score`
  是整张图或多个顶部区域汇总后的全局异常程度
- `threshold`
  是把 `score` 和热力图解释成“正常/异常”的默认分界线

---

## 9. 模型和样本管理

### 9.1 列出模型

```python
print(store.list_models())
```

### 9.2 查看模型详情

```python
print(store.get_model("model_xxxxx"))
```

### 9.3 修改当前版本默认 threshold

```python
print(store.update_model_threshold("model_xxxxx", 1.2))
```

作用：

- 更新当前版本默认 threshold
- 后续不显式传 `threshold` 时，将使用新值

### 9.4 导出模型

Web / HTTP 方式：

- 选择当前模型
- 点击“导出当前模型”
- 服务端会把该模型完整目录打包成 zip 下载

HTTP 接口：

```text
GET /api/models/{model_id}/export
```

用途：

- 训练完成后的模型交付
- 模型备份
- 迁移到另一台机器或另一个环境

### 9.5 导入模型

Web / HTTP 方式：

- 点击“导入模型”
- 选择之前导出的模型 zip
- 导入成功后会自动加入当前模型列表

HTTP 接口：

```text
POST /api/models/import
```

表单字段：

- `model_file`
  导出的模型 zip 文件

说明：

- 导入时会校验 `model_id` 和 `model_name`
- 如果目标环境里已经存在同名 `model_id` 或 `model_name`，会直接拒绝导入
- 推荐把训练好的模型从高性能训练机导出，再导入部署环境使用

### 9.6 分页查看样本

```python
print(store.list_samples("model_xxxxx", page=1, page_size=20))
```

### 9.7 查看单个样本详情

```python
print(store.get_sample_detail("model_xxxxx", "sample_xxxxx"))
```

### 9.8 扫描向量库中的异常样本

```python
print(store.scan_vector_bank("model_xxxxx"))
```

### 9.9 删除样本

```python
store.delete_sample("model_xxxxx", "sample_xxxxx")
```

### 9.10 修改样本轮廓

```python
store.update_sample_contour(
    model_id="model_xxxxx",
    sample_id="sample_xxxxx",
    contour=[[10, 10], [300, 10], [300, 120], [10, 120]],
    note="轮廓修正",
)
```

### 9.11 追加正样本

自动使用 YOLO 提取轮廓：

```python
result = store.add_positive_sample(
    model_id="model_xxxxx",
    image_path="/data/new.jpg",
)
```

手动指定轮廓：

```python
result = store.add_positive_sample(
    model_id="model_xxxxx",
    image_path="/data/new.jpg",
    contour=[[10, 10], [300, 10], [300, 120], [10, 120]],
    note="人工指定轮廓",
    append_max_vectors=20,
)
```

说明：

- 如果不传 `contour`，会先用 YOLO 自动提取列车顶部轮廓
- 追加正样本会把新样本 embedding 追加到当前版本记忆库
- `append_max_vectors` 控制单次新增向量数量上限

---

## 10. 常见文件说明

训练后，在 `root_dir` 下通常能看到：

```text
store_data/
├── registry.json
├── models/
│   └── model_xxxxx/
│       ├── model.json
│       └── versions/
│           └── v_xxxxx/
│               ├── version.json
│               ├── samples.json
│               ├── memory_model.pt
│               ├── raw/
│               ├── processed/
│               ├── tiles/
│               └── calibrate_processed/
└── tmp/
```

含义如下：

- `registry.json`
  全局模型索引
- `model.json`
  单个模型元信息
- `version.json`
  单个版本元信息，包括默认 threshold 和运行参数
- `samples.json`
  当前版本样本记录
- `memory_model.pt`
  当前版本异常检测引擎保存文件
- `raw/`
  原始样本图
- `processed/`
  顶部裁剪处理后的样本图
- `tiles/`
  子图和子图 embedding
- `calibrate_processed/`
  单独校准集预处理结果

---

## 11. 推荐使用建议

### 11.1 什么时候用 Web 服务

建议在这些场景优先用 `store_web`：

- 需要给运维或业务同学使用
- 需要可视化查看样本、热力图、轮廓和向量子图
- 需要通过 HTTP 给外部系统调用

### 11.2 什么时候用 import

建议在这些场景优先用 Python `import`：

- 你要和自己的推理流程、调度系统集成
- 你要做批量离线训练或批量检测
- 你要把训练和检测封装进已有服务

### 11.3 实际部署建议

- 调试阶段
  先直接 `store_web` 跑起来看效果
- 业务集成阶段
  优先用 `import` 调用核心接口
- 交付阶段
  用 `store_web` 或 `python -m store_web` 作为运维入口

---

## 12. 常见问题

### 12.1 为什么训练出来后还有 threshold？

因为这个库的核心不是分类 softmax，而是异常分数。必须有一个阈值，才能把连续分数解释成“正常/异常”。

### 12.2 为什么训练时要先做列车顶部分割？

因为业务只关心列车顶部区域。先分割顶部可以降低背景干扰，提高异常检测稳定性。

### 12.3 为什么追加正样本后不一定重训整个模型？

因为这里的核心模型是记忆库式异常检测。很多情况下，只需要给当前版本追加正常样本 embedding，就能更新识别边界。

### 12.4 `serve_forever()` 是做什么的？

它不是前台启动服务，而是“后台服务已经启动后，阻塞主线程，防止 Python 进程结束”。

### 12.5 `store_web --help` 和 `python -m store_web --help` 有什么区别？

没有本质区别，都是同一套启动逻辑和同一套中文参数说明。前者更适合安装后直接运行，后者更适合开发调试。
