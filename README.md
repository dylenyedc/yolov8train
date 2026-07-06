# YOLOv8 训练与测试工具

这个目录用于训练 YOLOv8 检测模型，并保存已经训练好的权重。

## 目录说明

- `train.py`：训练脚本。
- `predict_testimage.py`：测试脚本，会读取 `testimage/` 里的所有图片，先清空 `testimage-result/`，再保存识别结果。
- `requirements.txt`：Python 依赖。
- `model_archieve/`：已归档的模型权重和对应元数据。
- `testimage/`：用于快速测试识别效果的图片目录。

## 安装环境

推荐使用 `uv` 创建项目内虚拟环境：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

如果需要使用显卡训练，请确保 `.venv` 里安装的是 CUDA 版 PyTorch。可以用下面的命令检查：

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 训练模型

直接训练默认数据集：

```bash
.venv/bin/python train.py
```

通过环境变量切换数据集：

```bash
YOLO_DATASET_PATH=/path/to/dataset .venv/bin/python train.py
```

`YOLO_DATASET_PATH` 可以指向数据集目录，也可以直接指向 `data.yaml`：

```bash
YOLO_DATASET_PATH=/path/to/dataset/data.yaml .venv/bin/python train.py
```

如果要混合多个数据集，用冒号分隔：

```bash
YOLO_DATASET_PATH=/path/to/dataset_a:/path/to/dataset_b .venv/bin/python train.py
```

训练完成后，脚本会把 `runs/detect/train/weights/best.pt` 复制到 `model_archieve/`。权重文件名包含基础模型、训练时间和数据集名称，例如：

```text
yolov8s_20260706_205349_white_cylinder_detection.pt
```

同时会保存一个同目录的 JSON 元数据文件，里面记录训练使用的数据集、类别名、Roboflow 版本信息、训练参数和对应权重路径。

## 测试图片

把要测试的图片放进 `testimage/`，然后运行：

```bash
.venv/bin/python predict_testimage.py
```

脚本会：

1. 清空 `testimage-result/`
2. 读取 `testimage/` 里的所有图片
3. 使用默认权重进行识别
4. 把标注后的图片保存到 `testimage-result/`

默认权重路径是：

```text
runs/detect/train/weights/best.pt
```

如果这个文件不存在，脚本会退回使用 `yolov8n.pt`。
