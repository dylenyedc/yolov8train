# YOLOv8 训练与测试工具

这个目录用于训练 YOLOv8 检测模型，并保存已经训练好的权重。
把数据集扔到datasets/目录下，等待魔法发生

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

训练集支持按来源抽样或翻倍，倍率范围是 `0.1x` 到 `10x`。命令行用 `YOLO_TRAIN_SAMPLE_MULTIPLIERS` 传 JSON，键名是训练来源名；普通数据集就是目录名，金样本会写成 `dataset_golden`：

```bash
YOLO_TRAIN_DATASET_PATH=bucket:golden@golden \
YOLO_VALID_DATASET_PATH=bucket \
YOLO_TRAIN_SAMPLE_MULTIPLIERS='{"bucket":0.5,"golden_golden":2.0}' \
.venv/bin/python train.py
```

小于 `1x` 会抽样训练图，大于 `1x` 会在训练列表里重复图片；验证集和测试集不会被倍率影响。

训练支持增强参数配方，可以用 `YOLO_TRAIN_PRESET` 指定：

```bash
YOLO_TRAIN_PRESET=official_default .venv/bin/python train.py
YOLO_TRAIN_PRESET=saturation_heavy .venv/bin/python train.py
YOLO_TRAIN_PRESET=physical_heavy .venv/bin/python train.py
YOLO_TRAIN_PRESET=saturation_physical_heavy .venv/bin/python train.py
```

当前内置配方：

- `official_default`：Ultralytics YOLOv8 默认训练/增强参数。
- `saturation_heavy`：颜色扰动更强，几何扰动保持温和。
- `physical_heavy`：几何/视角扰动更强，颜色扰动较克制。
- `saturation_physical_heavy`：颜色和几何/视角扰动都更强。

交互界面和 WebUI 里选择基础模型时，会先选择 YOLO 家族，再选择尺寸。家族列表里 `yolo8` 和 `yolo26` 会置顶；其中 `yolo8+n` 会解析为 `yolov8n`，`yolo26+n` 会解析为 `yolo26n`。如果项目根目录里有对应的本地权重文件，例如 `yolo26n.pt`，训练会优先使用本地文件。

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

训练完成后，脚本会把 `runs/detect/train/weights/best.pt` 复制到 `model_archieve/`。权重文件名会尽量短：紧凑模型名、月日，以及一组容易记的英文 `形容词+名词`。例如：

```text
8n_0709_vivid_harbor.pt
```

如果想自己指定最后的名字字段，可以在终端或 WebUI 里填写自定义批注，也可以用环境变量：

```bash
YOLO_MODEL_NOTE=field_test_a .venv/bin/python train.py
```

生成的权重名会类似：

```text
8n_0709_field_test_a.pt
```

完整信息不再塞进 `.pt` 文件名，而是保存到同目录 JSON 元数据文件里。JSON 会记录训练使用的数据集、类别名、Roboflow 版本信息、增强配方、训练参数、训练集倍率、占比最大的训练来源和对应权重路径。

## 交互式训练和批量测试

也可以使用交互式脚本：

```bash
.venv/bin/python interactive.py
```

交互流程：

1. 主菜单包含 `训练`、`测试`、`模型管理`、`退出` 四个入口。
2. 训练入口会依次选择 YOLO 家族、模型尺寸、增强参数配方、训练集来源、验证集来源，然后开始训练并自动归档权重。
3. 测试入口会依次选择权重来源、一个或多个模型权重、测试集来源，然后开始批量测试。
4. 训练或测试结束后，会返回主菜单。
5. 模型管理入口可以把模型移入停用归档区，或从停用归档区恢复。
6. 每级菜单都有 `返回` 选项；列表里用 ↑/↓ 移动，Enter 勾选或确认。
7. 数据集列表里会显示每个数据集目录的最后更新时间；如果存在 `golden/images`，也会显示 `dataset/golden` 金样本子集。
8. 训练集来源列表里可以用 `+` / `-` 调整当前数据集倍率，用 `d` 查看当前数据集详情；模型列表里也可以用 `d` 查看元数据详情。

`archieve` 会列出 `model_archieve/` 里的归档权重，并显示基础模型、数据集来源和创建时间等信息。

`lastrun` 会提供最近一次训练的：

```text
runs/detect/train/weights/last.pt
runs/detect/train/weights/best.pt
```

交互式测试会让你选择测试集来源，然后使用这些数据集里的 `test/images` 跑 YOLO 测试。测试集来源也支持多选和 `dataset/golden`。

测试集来源同样支持按来源抽样或翻倍，倍率范围和训练集一致。终端界面里在测试集列表按 `+` / `-` 调整；WebUI 里每个测试集来源旁边有倍率下拉。命令行可以这样写：

```bash
YOLO_TEST_DATASET_PATH=bucket:golden@golden \
YOLO_TEST_SAMPLE_MULTIPLIERS='{"bucket":0.5,"golden_golden":2.0}' \
.venv/bin/python - <<'PY'
from pathlib import Path
import train
train.run_batch_testing([Path("model_archieve/your_model.pt")])
PY
```

批量测试会为每个模型单独跑一次 `val(split="test")`，指标和曲线会保存在：

```text
runs/detect/batch_test_<时间>_<测试集名>/<序号>_<模型名>/
```

同时脚本会把多个模型的 PR、F1-Confidence、Precision-Confidence、Recall-Confidence 曲线画到同一张图里，不同模型使用不同颜色。对比曲线保存在：

```text
runs/detect/batch_test_<时间>_<测试集名>/comparison_curves.png
runs/detect/batch_test_<时间>_<测试集名>/comparison_curves.json
```

如果一次选择了多个测试集，脚本还会给每个测试集来源单独生成 watch 预测拼图，避免只看到 `val_batch0/1/2` 里第一个测试集的样例。为了节省存储，watch 不会保留单张预测图，只保存 batch 拼图：

```text
runs/detect/batch_test_<时间>_<测试集名>/watch/<测试集来源>/<模型名>/batch_001.jpg
```

每个测试集来源默认抽取前 24 张图片生成预测可视化，每张拼图默认展示 12 张。

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
http://0.0.0.0:7860
```

在本机浏览器中也可以继续打开：

```text
http://127.0.0.1:7860
```

页面功能：

- 选择 YOLO 家族和模型尺寸。
- 选择增强参数配方。
- 多选训练集来源，并为每个训练来源选择抽样/翻倍倍率。
- 多选验证集来源。
- 多选测试集来源。
- 多选 `model_archieve/` 里的归档权重或最近一次训练的 `last.pt` / `best.pt` 进行批量测试。
- 管理模型：可以把不想继续参与比较的模型移入 `model_archieve_disabled/`，也可以恢复回来。
- 右侧显示后台任务日志和退出码，效果类似终端状态。
- 展示 `runs/` 里的训练、验证、测试结果，包括曲线图、混淆矩阵、batch 图、批量测试对比曲线、`results.csv` 预览和文件链接。
- 急停按钮可以终止当前训练或批量测试进程。

`model_archieve_disabled/` 是停用归档区，不会上传到 GitHub。被移入这个目录的模型不会出现在批量测试权重列表里。
