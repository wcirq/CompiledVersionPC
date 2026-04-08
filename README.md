# CompiledVersion 使用说明

这个目录是从原始工程里拆出来的“可编译版本”。

目标是：

- 顶层 `main.py` 保持为 Python 示例入口，便于调试和调用
- 核心算法集中放在 `engine/` 目录，方便后续编译为二进制扩展
- 运行时既可以直接用源码，也可以替换为编译后的二进制模块

## 目录结构

```text
./
├── main.py
├── build_engine.py
├── setup_binary.py
├── BUILD.md
└── engine/
    ├── __init__.py
    ├── augment.py
    ├── backbone.py
    ├── indexing.py
    ├── runtime.py
    └── utils.py
```

各文件职责如下：

- `main.py`
  顶层示例入口，负责命令行参数解析、模式分发、调用 `engine`
- `engine/runtime.py`
  主检测引擎
- `engine/backbone.py`
  特征提取网络
- `engine/indexing.py`
  最近邻索引
- `engine/augment.py`
  训练增强逻辑
- `engine/utils.py`
  工具函数、ROI 选择、数据读写、参数解析
- `build_engine.py`
  一键编译脚本
- `setup_binary.py`
  `setuptools + Cython` 编译配置

## 环境准备

建议先准备 Python 环境，并安装以下依赖：

```bash
pip install cython setuptools torch torchvision opencv-python numpy scikit-learn tqdm pillow
```

如果你要使用 `faiss` 后端，需要额外安装对应版本的 `faiss`。

如果你的环境里没有 C/C++ 编译工具链，编译二进制扩展时会失败。

常见要求：

- Linux：`gcc/g++`、Python 开发头文件
- Windows：Visual Studio C++ Build Tools

## 直接用源码运行

先进入当前目录：

```bash
cd "$(dirname "$0")"
```

然后直接运行：

```bash
python main.py -h
```

当前支持的模式：

- `train`
- `detect`
- `detect_batch`
- `calibrate_threshold`
- `append_positive`

## 常见用法

### 1. 训练正常样本库

```bash
python main.py train \
  --train_dir /path/to/train_images \
  --save_model memory_model.pt \
  --input_size 240 240 \
  --crop_size 160 160 \
  --batch_size 32
```

说明：

- `--train_dir` 是正常样本目录
- `--save_model` 是训练后保存的模型文件
- `--crop_size` 是滑窗尺寸
- `--input_size` 是送入骨干网络的尺寸

### 2. 单张图片检测

```bash
python main.py detect \
  --model_path memory_model.pt \
  --input /path/to/test.jpg \
  --output ./output \
  --threshold 1.0
```

如果需要手动框选 ROI：

```bash
python main.py detect \
  --model_path memory_model.pt \
  --input /path/to/test.jpg \
  --output ./output \
  --threshold 1.0 \
  --select_roi
```

### 3. 批量检测

```bash
python main.py detect_batch \
  --model_path memory_model.pt \
  --input /path/to/test_dir \
  --output ./output_batch \
  --threshold 1.0
```

输出目录里会包含：

- 可视化结果图
- `detection_results.json`

### 4. 用正常验证集自动标定阈值

```bash
python main.py calibrate_threshold \
  --model_path memory_model.pt \
  --input /path/to/val_normal_images \
  --quantile 0.99
```

标定完成后会把推荐阈值写回模型文件。

### 5. 追加新的正常样本

```bash
python main.py append_positive \
  --model_path memory_model.pt \
  --input /path/to/new_normal_images
```

如果要框 ROI 后再追加：

```bash
python main.py append_positive \
  --model_path memory_model.pt \
  --input /path/to/new_normal_images \
  --append_select_roi
```

## 如何编译

这个工程的设计是：

- `main.py` 不编译
- 只编译 `engine/` 里的核心模块

这样做的好处是：

- 入口还保留源码，便于改参数和调试
- 核心算法可以转成二进制扩展
- 发布时更灵活

### 1. 安装编译依赖

```bash
pip install cython setuptools
```

### 2. 执行一键编译

在当前目录下运行：

```bash
python build_engine.py --clean
```

这个命令会：

- 清理旧的 `build/` 和 `dist_binary/`
- 编译 `engine/` 下的 Python 模块
- 在原地生成二进制扩展
- 把结果收集到 `dist_binary/engine/`
- 默认删除编译生成的 `.c` 中间文件

### 3. 编译输出位置

编译完成后，二进制文件会放到：

```text
./dist_binary/engine/
```

常见结果：

- Linux：`*.so`
- Windows：`*.pyd`

同时会保留：

- `dist_binary/engine/__init__.py`

### 4. 保留中间生成的 C 文件

如果你想保留 `Cython` 生成的 `.c` 文件：

```bash
python build_engine.py --clean --keep-generated-c
```

## 编译后怎么用

有两种常见方式。

### 方式一：开发阶段

继续直接使用源码目录运行：

```bash
python main.py detect --model_path memory_model.pt --input /path/to/test.jpg
```

这种方式适合你本地调试。

### 方式二：发布阶段

把这些文件一起带走：

- `main.py`
- `dist_binary/engine/__init__.py`
- `dist_binary/engine/*.so` 或 `dist_binary/engine/*.pyd`

然后把运行环境里的导入路径指向 `dist_binary`。

如果你后续希望“优先加载二进制，找不到再回退源码”，可以再改一版 `main.py` 的导入逻辑。

## 重要说明

### 1. 当前模型文件

默认模型保存名是：

```text
memory_model.pt
```

你也可以通过这些参数改掉：

- `--save_model`
- `--model_path`

### 2. 默认骨干网络

当前只支持：

```text
resnet50
```

### 3. 运行依赖不会因为编译而消失

即使把 `engine/` 编译成二进制扩展，运行时仍然需要这些依赖已经安装：

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `tqdm`
- `Pillow`

### 4. 二进制扩展和 Python 版本有关

编译出来的 `.so` 或 `.pyd` 通常和下面这些条件绑定：

- Python 主版本和次版本
- 操作系统
- CPU 架构
- 编译器环境

也就是说：

- 在 Linux 上编出来的文件，通常不能直接拿到 Windows 用
- 在 Python 3.10 上编出来的文件，通常不能直接给 Python 3.12 用

## 快速命令汇总

查看帮助：

```bash
python main.py -h
```

训练：

```bash
python main.py train --train_dir /path/to/train_images --save_model memory_model.pt
```

单图检测：

```bash
python main.py detect --model_path memory_model.pt --input /path/to/test.jpg --output ./output
```

批量检测：

```bash
python main.py detect_batch --model_path memory_model.pt --input /path/to/test_dir --output ./output_batch
```

阈值标定：

```bash
python main.py calibrate_threshold --model_path memory_model.pt --input /path/to/val_normal_images
```

编译：

```bash
python build_engine.py --clean
```
