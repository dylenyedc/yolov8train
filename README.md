# YOLOv8 训练与测试工具

这个目录用于训练 YOLOv8 检测模型，并保存已经训练好的权重。

## 目录说明

- `train.py`：训练脚本。
- `interactive.py`：交互式训练/验证脚本。
- `webui.py`：网页训练/验证控制台。
- `requirements.txt`：Python 依赖。
- `model_archieve/`：已归档的模型权重和对应元数据。

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

`YOLO_DATASET_PATH` 可以写 `datasets/` 下面的数据集文件夹名、数据集目录路径，也可以直接指向 `data.yaml`：

```bash
YOLO_DATASET_PATH=bucket .venv/bin/python train.py
YOLO_DATASET_PATH=datasets/bucket .venv/bin/python train.py
YOLO_DATASET_PATH=/path/to/dataset/data.yaml .venv/bin/python train.py
```

如果要混合多个数据集，用冒号或逗号分隔：

```bash
YOLO_DATASET_PATH=barrel:bucket:white_cylinder_detection .venv/bin/python train.py
YOLO_DATASET_PATH=/path/to/dataset_a:/path/to/dataset_b .venv/bin/python train.py
YOLO_DATASET_PATH=barrel,bucket .venv/bin/python train.py
```

如果要做交叉验证，也可以分别指定训练集来源和验证集来源。两边都支持多选：

```bash
YOLO_TRAIN_DATASET_PATH=barrel YOLO_VALID_DATASET_PATH=bucket .venv/bin/python train.py
YOLO_TRAIN_DATASET_PATH=barrel:bucket YOLO_VALID_DATASET_PATH=white_cylinder_detection,bucket .venv/bin/python train.py
```

兼容写法 `YOLO_VAL_DATASET_PATH` 也可以使用，但推荐用 `YOLO_VALID_DATASET_PATH`。

可以使用金样本子集 `golden`。它不是 `train/valid/test`，而是每个数据集目录下的额外子目录：

```text
datasets/<dataset>/golden/images
datasets/<dataset>/golden/labels
```

如果某个数据集有 `golden/images`，交互界面会显示 `dataset/golden` 选项。命令行中可以写成 `dataset@golden`，它可以被加入训练集、验证集或测试集：

```bash
YOLO_TRAIN_DATASET_PATH=bucket@golden YOLO_VALID_DATASET_PATH=bucket .venv/bin/python train.py
YOLO_TRAIN_DATASET_PATH=barrel:bucket@golden YOLO_VALID_DATASET_PATH=white_cylinder_detection .venv/bin/python train.py
```

训练完成后，脚本会把 `runs/detect/train/weights/best.pt` 复制到 `model_archieve/`。权重文件名包含基础模型、训练时间和数据集名称，例如：

```text
yolov8s_20260706_205349_white_cylinder_detection.pt
```

同时会保存一个同目录的 JSON 元数据文件，里面记录训练使用的数据集、类别名、Roboflow 版本信息、训练参数和对应权重路径。

混合数据集训练时，权重名会包含所有数据集来源，例如：

```text
yolov8n_20260707_021500_mixed_barrel_bucket_white_cylinder_detection.pt
```

对应 JSON 里会有 `is_mixed_dataset`、`dataset_sources` 和 `datasets`，用于查看本次训练具体混合了哪些数据集。

## 交互式训练和验证

也可以使用交互式脚本：

```bash
.venv/bin/python interactive.py
```

交互流程：

1. 询问是否开始训练。
2. 选择基础模型，例如 `yolov8n`、`yolov8s`。
3. 选择一个或多个训练集来源。
4. 选择一个或多个验证集来源。
5. 数据集列表里会显示每个数据集目录的最后更新时间；如果存在 `golden/images`，也会显示 `dataset/golden` 金样本子集。
6. 数据集列表里用 ↑/↓ 移动，Enter 勾选，移动到最下面的 `Next` 后按 Enter 继续。
7. 如果选择训练，脚本会开始训练并自动归档权重。
8. 训练结束后会询问是否验证。
9. 验证权重来源可以选择 `archieve` 或 `lastrun`。

`archieve` 会列出 `model_archieve/` 里的归档权重，并显示基础模型、数据集来源和创建时间等信息。

`lastrun` 会提供最近一次训练的：

```text
runs/detect/train/weights/last.pt
runs/detect/train/weights/best.pt
```

交互式验证会让你选择测试集来源，然后使用这些数据集里的 `test/images` 跑 YOLO 验证。测试集来源也支持多选和 `dataset/golden`。

命令行也可以生成测试集配置：

```bash
YOLO_TEST_DATASET_PATH=bucket@golden .venv/bin/python - <<'PY'
import train
print(train.testing_data_config())
PY
```

## WebUI

也可以启动网页控制台：

```bash
.venv/bin/python webui.py
```

默认地址：

```text
http://127.0.0.1:7860
```

页面功能：

- 选择基础模型。
- 多选训练集来源。
- 多选验证集来源。
- 多选测试集来源。
- 选择 `model_archieve/` 里的归档权重或最近一次训练的 `last.pt` / `best.pt`。
- 右侧显示后台任务日志和退出码，效果类似终端状态。
- 展示 `runs/` 里的训练、验证、测试结果，包括曲线图、混淆矩阵、batch 图、`results.csv` 预览和文件链接。
