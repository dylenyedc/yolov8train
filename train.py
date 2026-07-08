import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

from faker import Faker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "white_cylinder_detection"
GENERATED_DIR = ROOT / ".generated"
ARCHIVE_DIR = ROOT / "model_archieve"
DISABLED_ARCHIVE_DIR = ROOT / "model_archieve_disabled"
BASE_MODEL = "yolov8n"
MODEL_FAMILIES = {
    "yolo8": {
        "prefix": "yolov8",
        "sizes": ["n", "s", "m", "l", "x"],
        "description": "YOLOv8",
    },
    "yolo26": {
        "prefix": "yolo26",
        "sizes": ["n", "s", "m", "l", "x"],
        "description": "YOLO26",
    },
    "yolo11": {
        "prefix": "yolo11",
        "sizes": ["n", "s", "m", "l", "x"],
        "description": "YOLO11",
    },
    "yolo10": {
        "prefix": "yolov10",
        "sizes": ["n", "s", "m", "l", "x"],
        "description": "YOLOv10",
    },
}
DEFAULT_MODEL_FAMILY = "yolo8"
DEFAULT_MODEL_SIZE = "n"
DATASET_ENV = "YOLO_DATASET_PATH"
TRAIN_DATASET_ENV = "YOLO_TRAIN_DATASET_PATH"
VAL_DATASET_ENV = "YOLO_VAL_DATASET_PATH"
VALID_DATASET_ENV = "YOLO_VALID_DATASET_PATH"
TEST_DATASET_ENV = "YOLO_TEST_DATASET_PATH"
TRAIN_PRESET_ENV = "YOLO_TRAIN_PRESET"
TRAIN_SAMPLE_ENV = "YOLO_TRAIN_SAMPLE_MULTIPLIERS"
MIXED_DATASET_CLASS_NAME = "busket"
GOLDEN_SPLIT = "golden"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WATCH_IMAGES_PER_SOURCE = 24
TRAIN_SAMPLE_CHOICES = [0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
FAKER = Faker("en_US")
MODEL_ADJECTIVES = [
    "agile", "amber", "brave", "bright", "calm", "clever", "crisp", "eager",
    "fierce", "fresh", "golden", "keen", "lucky", "nimble", "prime", "rapid",
    "royal", "sharp", "silent", "steady", "swift", "vivid", "wise", "zesty",
]
MODEL_NOUNS = [
    "anchor", "beacon", "comet", "copper", "delta", "ember", "harbor", "iris",
    "kernel", "lantern", "matrix", "needle", "orbit", "pixel", "quartz",
    "rocket", "signal", "summit", "vertex", "voyage",
]
MODEL_COLORS = [
    (36, 99, 235),
    (220, 38, 38),
    (5, 150, 105),
    (217, 119, 6),
    (124, 58, 237),
    (8, 145, 178),
    (190, 24, 93),
    (77, 124, 15),
]
AUGMENT_PRESETS = {
    "official_default": {
        "description": "Ultralytics YOLOv8 默认训练/增强参数",
        "args": {
            "epochs": 100,
            "patience": 100,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "perspective": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "close_mosaic": 10,
        },
    },
    "saturation_heavy": {
        "description": "颜色扰动更强，几何扰动保持温和",
        "args": {
            "epochs": 60,
            "patience": 12,
            "hsv_h": 0.03,
            "hsv_s": 0.9,
            "hsv_v": 0.55,
            "degrees": 8.0,
            "translate": 0.1,
            "scale": 0.5,
            "perspective": 0.0005,
            "fliplr": 0.5,
            "mosaic": 0.3,
            "mixup": 0.0,
            "close_mosaic": 10,
        },
    },
    "physical_heavy": {
        "description": "几何/视角扰动更强，颜色扰动较克制",
        "args": {
            "epochs": 60,
            "patience": 12,
            "hsv_h": 0.02,
            "hsv_s": 0.5,
            "hsv_v": 0.35,
            "degrees": 15.0,
            "translate": 0.15,
            "scale": 0.65,
            "perspective": 0.001,
            "fliplr": 0.5,
            "mosaic": 0.3,
            "mixup": 0.0,
            "close_mosaic": 10,
        },
    },
    "saturation_physical_heavy": {
        "description": "颜色和几何/视角扰动都更强",
        "args": {
            "epochs": 60,
            "patience": 12,
            "hsv_h": 0.03,
            "hsv_s": 0.9,
            "hsv_v": 0.55,
            "degrees": 15.0,
            "translate": 0.15,
            "scale": 0.65,
            "perspective": 0.001,
            "fliplr": 0.5,
            "mosaic": 0.3,
            "mixup": 0.0,
            "close_mosaic": 10,
        },
    },
}
DEFAULT_AUGMENT_PRESET = "official_default"


def safe_name(value: object) -> str:
    text = str(value or "dataset").strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("._-") or "dataset"


def compact_model_name(base_model: str) -> str:
    name = safe_name(base_model)
    for prefix in ["yolov", "yolo"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def sample_choice_options() -> list[float]:
    return list(TRAIN_SAMPLE_CHOICES)


def clamp_sample_multiplier(value: object) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError("训练集倍率必须大于 0")
    if number < TRAIN_SAMPLE_CHOICES[0] or number > TRAIN_SAMPLE_CHOICES[-1]:
        raise ValueError(f"训练集倍率必须在 {TRAIN_SAMPLE_CHOICES[0]} 到 {TRAIN_SAMPLE_CHOICES[-1]} 之间")
    return number


def format_multiplier(value: float) -> str:
    return f"{value:g}x"


def memorable_model_words() -> str:
    adjective = FAKER.random_element(MODEL_ADJECTIVES)
    noun = FAKER.random_element(MODEL_NOUNS)
    return safe_name(f"{adjective}_{noun}").lower()


def model_family_options() -> list[str]:
    return list(MODEL_FAMILIES)


def model_size_options(family: str) -> list[str]:
    family_name = safe_name(family)
    if family_name not in MODEL_FAMILIES:
        available = ", ".join(model_family_options())
        raise ValueError(f"Unknown model family: {family_name}. Available families: {available}")
    return list(MODEL_FAMILIES[family_name]["sizes"])


def resolve_base_model(family_or_model: str | None = None, size: str | None = None) -> str:
    value = safe_name(family_or_model or DEFAULT_MODEL_FAMILY)
    if size is None and value not in MODEL_FAMILIES:
        return value
    family_name = value
    if family_name not in MODEL_FAMILIES:
        available = ", ".join(model_family_options())
        raise ValueError(f"Unknown model family: {family_name}. Available families: {available}")
    model_size = safe_name(size or DEFAULT_MODEL_SIZE)
    if model_size not in MODEL_FAMILIES[family_name]["sizes"]:
        available = ", ".join(model_size_options(family_name))
        raise ValueError(f"Unknown model size for {family_name}: {model_size}. Available sizes: {available}")
    return f"{MODEL_FAMILIES[family_name]['prefix']}{model_size}"


def model_weight_source(base_model: str) -> str:
    local_weights = ROOT / f"{base_model}.pt"
    if local_weights.exists():
        return str(local_weights)
    return f"{base_model}.pt"


def augment_preset_options() -> list[str]:
    return list(AUGMENT_PRESETS)


def resolve_augment_preset(value: str | None = None) -> tuple[str, dict]:
    preset_name = safe_name(value or os.environ.get(TRAIN_PRESET_ENV) or DEFAULT_AUGMENT_PRESET)
    if preset_name not in AUGMENT_PRESETS:
        available = ", ".join(augment_preset_options())
        raise ValueError(f"Unknown training preset: {preset_name}. Available presets: {available}")
    preset = AUGMENT_PRESETS[preset_name]
    return preset_name, dict(preset["args"])


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def dataset_config_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        dataset_alias = DATASETS_DIR / path
        if dataset_alias.exists():
            path = dataset_alias
        else:
            path = ROOT / path
    if path.is_dir():
        path = path / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config was not found: {path}")
    return path.resolve()


def parse_dataset_sources(value: str, env_name: str) -> list[tuple[Path, str | None]]:
    raw_paths = [
        part.strip()
        for chunk in value.split(os.pathsep)
        for part in chunk.split(",")
        if part.strip()
    ]
    sources: list[tuple[Path, str | None]] = []
    for raw_path in raw_paths:
        if "@" in raw_path:
            dataset_path, split = raw_path.rsplit("@", 1)
            sources.append((dataset_config_path(Path(dataset_path)), safe_name(split)))
        else:
            sources.append((dataset_config_path(Path(raw_path)), None))
    unique_sources = list(dict.fromkeys(sources))
    if not unique_sources:
        raise ValueError(f"{env_name} did not contain any dataset paths")
    return unique_sources


def parse_dataset_configs(value: str, env_name: str) -> list[Path]:
    return list(dict.fromkeys(config_path for config_path, _ in parse_dataset_sources(value, env_name)))


def selected_dataset_configs(env_name: str = DATASET_ENV, default: Path | str | None = None) -> list[Path]:
    value = os.environ.get(env_name)
    if value is None:
        value = str(default or DEFAULT_DATASET)
    return parse_dataset_configs(value, env_name)


def resolve_split(config_path: Path, config: dict, split: str) -> Path | None:
    value = config.get(split)
    if value:
        if isinstance(value, list):
            raise ValueError(f"{config_path} already contains a list for {split}; nested mixing is not supported")
        path = Path(value)
    elif split == GOLDEN_SPLIT:
        path = Path(GOLDEN_SPLIT) / "images"
    else:
        return None
    if path.is_absolute():
        return path
    base = Path(config.get("path") or config_path.parent)
    if not base.is_absolute():
        base = config_path.parent / base
    resolved = (base / path).resolve()
    if resolved.exists():
        return resolved

    # Roboflow exports often use ../train/images even when the dataset is
    # later placed in its own folder. In that layout, ./train/images is right.
    parts = path.parts
    if parts and parts[0] == "..":
        local_path = (config_path.parent / Path(*parts[1:])).resolve()
        if local_path.exists() or not resolved.exists():
            return local_path
    return resolved


def dataset_id(config_path: Path) -> str:
    return safe_name(config_path.parent.name)


def dataset_metadata(config_path: Path) -> dict:
    config = load_yaml(config_path)
    roboflow = config.get("roboflow") or {}
    dataset_root = Path(config.get("path") or config_path.parent)
    if not dataset_root.is_absolute():
        dataset_root = (config_path.parent / dataset_root).resolve()
    return {
        "dataset_name": dataset_id(config_path),
        "dataset_root": str(dataset_root),
        "config_path": str(config_path),
        "class_names": config.get("names"),
        "roboflow": roboflow,
        "version": roboflow.get("version") or config.get("version"),
    }


def mixed_dataset_config(config_paths: list[Path]) -> Path:
    names = dataset_names(config_paths)
    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []

    for config_path in config_paths:
        config = load_yaml(config_path)
        train_path = resolve_split(config_path, config, "train")
        val_path = resolve_split(config_path, config, "val")
        test_path = resolve_split(config_path, config, "test")
        if train_path is None or val_path is None:
            raise ValueError(f"{config_path} must define both train and val splits")
        train_paths.append(str(train_path))
        val_paths.append(str(val_path))
        if test_path is not None:
            test_paths.append(str(test_path))

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / f"mixed_{safe_name('_'.join(names))}.yaml"
    output = {
        "train": train_paths,
        "val": val_paths,
        "nc": 1,
        "names": [MIXED_DATASET_CLASS_NAME],
    }
    if test_paths:
        output["test"] = test_paths
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(output, file, sort_keys=False, allow_unicode=True)
    return output_path


def dataset_names(config_paths: list[Path]) -> list[str]:
    return [dataset_id(path) for path in config_paths]


def source_name(source: tuple[Path, str | None], default_split: str) -> str:
    config_path, split = source
    split_name = split or default_split
    if split_name in {"train", "val", "valid", "test"}:
        return dataset_id(config_path)
    return safe_name(f"{dataset_id(config_path)}_{split_name}")


def source_names(sources: list[tuple[Path, str | None]], default_split: str) -> list[str]:
    return [source_name(source, default_split) for source in sources]


def combined_dataset_name(config_paths: list[Path], prefix: str = "mixed") -> str:
    names = dataset_names(config_paths)
    if len(names) == 1:
        return names[0]
    return safe_name(prefix + "_" + "_".join(names))


def joined_dataset_name(config_paths: list[Path]) -> str:
    return safe_name("_".join(dataset_names(config_paths)))


def split_paths(config_paths: list[Path], split: str) -> list[str]:
    paths: list[str] = []
    for config_path in config_paths:
        config = load_yaml(config_path)
        split_path = resolve_split(config_path, config, split)
        if split_path is None:
            raise ValueError(f"{config_path} must define {split} split")
        if not split_path.exists():
            raise FileNotFoundError(f"{config_path} defines {split} split, but path was not found: {split_path}")
        paths.append(str(split_path))
    return paths


def source_split_paths(sources: list[tuple[Path, str | None]], default_split: str) -> list[str]:
    paths: list[str] = []
    for config_path, explicit_split in sources:
        split = explicit_split or default_split
        config = load_yaml(config_path)
        split_path = resolve_split(config_path, config, split)
        if split_path is None:
            raise ValueError(f"{config_path} must define {split} split")
        if not split_path.exists():
            if explicit_split == GOLDEN_SPLIT:
                raise FileNotFoundError(f"Golden split was selected but not found: {split_path}")
            raise FileNotFoundError(f"{config_path} defines {split} split, but path was not found: {split_path}")
        paths.append(str(split_path))
    return paths


def source_image_groups(
    sources: list[tuple[Path, str | None]],
    default_split: str,
    limit: int | None = None,
) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    for source in sources:
        image_dir = Path(source_split_paths([source], default_split)[0])
        images = sorted(path.resolve() for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if limit is not None:
            images = images[:limit]
        groups.append((source_name(source, default_split), images))
    return groups


def image_paths_for_source(source: tuple[Path, str | None], default_split: str) -> list[Path]:
    image_dir = Path(source_split_paths([source], default_split)[0])
    return sorted(path.resolve() for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def parse_train_sample_multipliers(value: str | None, sources: list[tuple[Path, str | None]]) -> dict[str, float]:
    keys = {source_name(source, "train") for source in sources}
    multipliers = {key: 1.0 for key in keys}
    if not value:
        return multipliers

    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        payload = {}
        for item in value.split(","):
            if not item.strip():
                continue
            key, raw_multiplier = item.split("=", 1)
            payload[key.strip()] = raw_multiplier.strip()

    if not isinstance(payload, dict):
        raise ValueError(f"{TRAIN_SAMPLE_ENV} must be a JSON object or key=value list")

    for key, raw_multiplier in payload.items():
        safe_key = safe_name(key)
        if safe_key in multipliers:
            multipliers[safe_key] = clamp_sample_multiplier(raw_multiplier)
    return multipliers


def source_sample_multiplier(source: tuple[Path, str | None], multipliers: dict[str, float] | None) -> float:
    if not multipliers:
        return 1.0
    return multipliers.get(source_name(source, "train"), 1.0)


def sampled_images(images: list[Path], multiplier: float, seed_key: str) -> list[Path]:
    if not images:
        return []
    rng = random.Random(seed_key)
    shuffled = images[:]
    rng.shuffle(shuffled)
    if multiplier < 1:
        sample_count = max(1, round(len(images) * multiplier))
        return shuffled[:sample_count]

    full_repeats = int(multiplier)
    fraction = multiplier - full_repeats
    selected = images * full_repeats
    if fraction > 0:
        extra_count = round(len(images) * fraction)
        selected.extend(shuffled[:extra_count])
    return selected


def materialize_train_sources(
    train_sources: list[tuple[Path, str | None]],
    multipliers: dict[str, float] | None = None,
) -> tuple[list[str], list[dict]]:
    paths: list[str] = []
    sample_info: list[dict] = []
    sampled_lines: list[str] = []
    uses_sampling = False

    for source in train_sources:
        label = source_name(source, "train")
        multiplier = source_sample_multiplier(source, multipliers)
        images = image_paths_for_source(source, "train")
        effective_images = sampled_images(images, multiplier, f"{label}:{multiplier}")
        if multiplier != 1.0:
            uses_sampling = True
        sample_info.append({
            "source": label,
            "multiplier": multiplier,
            "original_image_count": len(images),
            "effective_image_count": len(effective_images),
        })
        sampled_lines.extend(str(path) for path in effective_images)
        if multiplier == 1.0:
            paths.extend(source_split_paths([source], "train"))

    if not uses_sampling:
        return paths, sample_info

    output_name = safe_name(
        "sampled_train_"
        + "_".join(f"{item['source']}_{format_multiplier(item['multiplier'])}" for item in sample_info)
    )
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / f"{output_name}.txt"
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(sampled_lines))
        file.write("\n")
    return [str(output_path)], sample_info


def dominant_train_source(sample_info: list[dict], fallback: str = "dataset") -> str:
    if not sample_info:
        return safe_name(fallback)
    dominant = max(sample_info, key=lambda item: item.get("effective_image_count", 0))
    return safe_name(dominant.get("source") or fallback)


def optional_split_paths(config_paths: list[Path], split: str) -> list[str]:
    paths: list[str] = []
    for config_path in config_paths:
        config = load_yaml(config_path)
        split_path = resolve_split(config_path, config, split)
        if split_path is not None and split_path.exists():
            paths.append(str(split_path))
    return paths


def cross_validation_dataset_config(
    train_sources: list[tuple[Path, str | None]],
    val_sources: list[tuple[Path, str | None]],
    train_multipliers: dict[str, float] | None = None,
) -> tuple[Path, list[dict]]:
    train_name = safe_name("_".join(source_names(train_sources, "train")))
    val_name = safe_name("_".join(source_names(val_sources, "val")))
    output_name = safe_name(f"cross_{train_name}_valid_{val_name}")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / f"{output_name}.yaml"
    train_paths, sample_info = materialize_train_sources(train_sources, train_multipliers)
    output = {
        "train": train_paths,
        "val": source_split_paths(val_sources, "val"),
        "nc": 1,
        "names": [MIXED_DATASET_CLASS_NAME],
    }
    val_paths = list(dict.fromkeys(config_path for config_path, _ in val_sources))
    test_paths = optional_split_paths(val_paths, "test")
    if test_paths:
        output["test"] = test_paths
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(output, file, sort_keys=False, allow_unicode=True)
    return output_path, sample_info


def test_dataset_config(test_sources: list[tuple[Path, str | None]]) -> tuple[Path, str, list[tuple[Path, str | None]]]:
    test_name = safe_name("_".join(source_names(test_sources, "test")))
    output_name = safe_name(f"test_{test_name}")
    test_image_paths = source_split_paths(test_sources, "test")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / f"{output_name}.yaml"
    output = {
        "train": test_image_paths,
        "val": test_image_paths,
        "test": test_image_paths,
        "nc": 1,
        "names": [MIXED_DATASET_CLASS_NAME],
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(output, file, sort_keys=False, allow_unicode=True)
    return output_path, output_name, test_sources


def testing_data_config() -> tuple[Path, str, list[tuple[Path, str | None]]]:
    value = os.environ.get(TEST_DATASET_ENV, str(DEFAULT_DATASET))
    test_sources = parse_dataset_sources(value, TEST_DATASET_ENV)
    return test_dataset_config(test_sources)


def mean_curve(y_values: object, x_values: np.ndarray) -> np.ndarray:
    values = np.asarray(y_values, dtype=float)
    if values.ndim == 1:
        return values
    if values.shape[0] == len(x_values):
        return values.mean(axis=1)
    if values.shape[-1] == len(x_values):
        return values.mean(axis=0)
    return values.reshape(-1, values.shape[-1]).mean(axis=0)


def metric_curve_payload(metrics, weight_path: Path) -> dict:
    curves = {}
    for x_values, y_values, x_label, y_label in metrics.curves_results:
        x_array = np.asarray(x_values, dtype=float)
        y_array = mean_curve(y_values, x_array)
        key = safe_name(f"{y_label}_vs_{x_label}")
        curves[key] = {
            "x": x_array.tolist(),
            "y": y_array.tolist(),
            "x_label": x_label,
            "y_label": y_label,
        }
    return {
        "model": weight_path.name,
        "weight_path": str(weight_path),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "curves": curves,
    }


def write_batch_metric_curves(results: list[dict], output_dir: Path) -> None:
    if not results:
        raise ValueError("没有可绘制的测试结果")

    preferred_order = [
        "Precision_vs_Recall",
        "F1_vs_Confidence",
        "Precision_vs_Confidence",
        "Recall_vs_Confidence",
    ]
    available = [
        key
        for key in preferred_order
        if any(key in item["curves"] for item in results)
    ]
    if not available:
        raise ValueError("测试结果里没有可绘制的曲线数据")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
    axes_flat = axes.flatten()
    for axis in axes_flat:
        axis.set_visible(False)

    for curve_index, curve_key in enumerate(available[:4]):
        axis = axes_flat[curve_index]
        axis.set_visible(True)
        for model_index, item in enumerate(results):
            curve = item["curves"].get(curve_key)
            if curve is None:
                continue
            color = tuple(component / 255 for component in MODEL_COLORS[model_index % len(MODEL_COLORS)])
            axis.plot(curve["x"], curve["y"], linewidth=2, color=color, label=f"{model_index + 1}: {item['model']}")
            axis.set_xlabel(curve["x_label"])
            axis.set_ylabel(curve["y_label"])
        axis.set_title(curve_key.replace("_", " "))
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, 1.02)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_dir / "comparison_curves.png")
    plt.close(fig)

    with (output_dir / "comparison_curves.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def write_watch_predictions(
    weight_paths: list[Path],
    test_sources: list[tuple[Path, str | None]],
    output_dir: Path,
    max_images_per_source: int = WATCH_IMAGES_PER_SOURCE,
) -> None:
    groups = source_image_groups(test_sources, "test", limit=max_images_per_source)
    watch_root = output_dir / "watch"
    watch_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source_label, image_paths in groups:
        if not image_paths:
            print(f"Watch skip empty test source: {source_label}")
            continue
        print(f"Writing watch predictions for {source_label}: {len(image_paths)} images")
        source_dir = watch_root / safe_name(source_label)
        for index, weight_path in enumerate(weight_paths, start=1):
            model_name = f"{index:02d}_{safe_name(weight_path.stem)}"
            YOLO(str(weight_path)).predict(
                source=[str(path) for path in image_paths],
                device=0,
                project=str(source_dir),
                name=model_name,
                exist_ok=True,
                save=True,
                verbose=False,
            )
            manifest.append({
                "source": source_label,
                "model": weight_path.name,
                "output_dir": str(source_dir / model_name),
                "image_count": len(image_paths),
            })
    with (watch_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def run_batch_testing(weight_paths: list[Path]) -> Path:
    if not weight_paths:
        raise ValueError("至少需要选择一个权重")
    resolved_weights = [Path(path).expanduser().resolve() for path in weight_paths]
    missing_weights = [path for path in resolved_weights if not path.exists()]
    if missing_weights:
        raise FileNotFoundError(f"权重不存在: {missing_weights[0]}")

    test_config, test_name, test_sources = testing_data_config()
    now = datetime.now()
    output_dir = ROOT / "runs" / "detect" / f"batch_test_{now:%Y%m%d_%H%M%S}_{test_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test config: {test_config}")
    print(f"Test name: {test_name}")
    print(f"Batch output: {output_dir}")

    batch_results: list[dict] = []
    for index, weight_path in enumerate(resolved_weights, start=1):
        run_name = f"{index:02d}_{safe_name(weight_path.stem)}"
        print(f"Validate [{index}/{len(resolved_weights)}]: {weight_path}")
        metrics = YOLO(str(weight_path)).val(
            data=str(test_config),
            split="test",
            device=0,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
        )
        batch_results.append(metric_curve_payload(metrics, weight_path))

    print("Writing comparison curves...")
    write_batch_metric_curves(batch_results, output_dir)
    print("Writing per-source watch predictions...")
    write_watch_predictions(resolved_weights, test_sources, output_dir)
    print(f"Batch testing finished: {output_dir}")
    return output_dir


def training_data_config() -> tuple[Path, str, list[Path], list[Path], list[dict]]:
    train_value = os.environ.get(TRAIN_DATASET_ENV)
    val_value = os.environ.get(VALID_DATASET_ENV) or os.environ.get(VAL_DATASET_ENV)
    if train_value is not None or val_value is not None:
        train_sources = parse_dataset_sources(train_value or str(DEFAULT_DATASET), TRAIN_DATASET_ENV)
        val_sources = parse_dataset_sources(val_value or train_value or str(DEFAULT_DATASET), VAL_DATASET_ENV)
        train_multipliers = parse_train_sample_multipliers(os.environ.get(TRAIN_SAMPLE_ENV), train_sources)
        train_paths = list(dict.fromkeys(config_path for config_path, _ in train_sources))
        val_paths = list(dict.fromkeys(config_path for config_path, _ in val_sources))
        dataset_name = safe_name(
            f"train_{'_'.join(source_names(train_sources, 'train'))}"
            f"_valid_{'_'.join(source_names(val_sources, 'val'))}"
        )
        data_config, sample_info = cross_validation_dataset_config(train_sources, val_sources, train_multipliers)
        return data_config, dataset_name, train_paths, val_paths, sample_info

    config_paths = selected_dataset_configs()
    train_sources = [(path, None) for path in config_paths]
    sample_info = [
        {
            "source": dataset_id(path),
            "multiplier": 1.0,
            "original_image_count": len(image_paths_for_source((path, None), "train")),
            "effective_image_count": len(image_paths_for_source((path, None), "train")),
        }
        for path in config_paths
    ]
    if len(config_paths) == 1:
        return config_paths[0], dataset_id(config_paths[0]), config_paths, config_paths, sample_info
    return mixed_dataset_config(config_paths), combined_dataset_name(config_paths), config_paths, config_paths, sample_info


def archive_best_weights(
    dataset_name: str,
    data_config: Path,
    train_config_paths: list[Path],
    val_config_paths: list[Path],
    base_model: str = BASE_MODEL,
    augment_preset_name: str = DEFAULT_AUGMENT_PRESET,
    train_args: dict | None = None,
    train_sample_info: list[dict] | None = None,
) -> Path:
    best_weights = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished, but best.pt was not found: {best_weights}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    dominant_source = dominant_train_source(train_sample_info or [], dataset_name)
    for _ in range(20):
        archive_name = f"{compact_model_name(base_model)}_{dominant_source}_{memorable_model_words()}.pt"
        archive_path = ARCHIVE_DIR / archive_name
        if not archive_path.exists():
            break
    else:
        archive_name = f"{compact_model_name(base_model)}_{dominant_source}_{now:%Y%m%d_%H%M%S}.pt"
        archive_path = ARCHIVE_DIR / archive_name
    shutil.copy2(best_weights, archive_path)

    metadata = {
        "created_at": now.isoformat(timespec="microseconds"),
        "base_model": base_model,
        "compact_model": compact_model_name(base_model),
        "augment_preset": augment_preset_name,
        "train_args": train_args or {},
        "dominant_train_source": dominant_source,
        "train_sample_multipliers": train_sample_info or [],
        "memorable_name": archive_path.stem,
        "weights": str(archive_path),
        "source_weights": str(best_weights),
        "training_data_config": str(data_config),
        "is_mixed_dataset": len(set(train_config_paths + val_config_paths)) > 1,
        "is_cross_validation": train_config_paths != val_config_paths,
        "dataset_name": dataset_name,
        "dataset_sources": dataset_names(list(dict.fromkeys(train_config_paths + val_config_paths))),
        "train_dataset_sources": dataset_names(train_config_paths),
        "val_dataset_sources": dataset_names(val_config_paths),
        "train_datasets": [dataset_metadata(path) for path in train_config_paths],
        "val_datasets": [dataset_metadata(path) for path in val_config_paths],
    }
    metadata_path = ARCHIVE_DIR / f"{now:%Y%m%d_%H%M%S_%f}.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return archive_path


def run_training(base_model: str = BASE_MODEL, augment_preset: str | None = None) -> Path:
    base_model = resolve_base_model(base_model)
    data_config, dataset_name, train_config_paths, val_config_paths, train_sample_info = training_data_config()
    augment_preset_name, train_args = resolve_augment_preset(augment_preset)
    print(f"Using dataset config: {data_config}")
    print(f"Using base model: {base_model}")
    print(f"Using training preset: {augment_preset_name}")
    print(f"Training sample multipliers: {json.dumps(train_sample_info, ensure_ascii=False, sort_keys=True)}")
    print(f"Training args: {json.dumps(train_args, ensure_ascii=False, sort_keys=True)}")
    model = YOLO(model_weight_source(base_model))
    model.train(
        data=str(data_config),
        imgsz=640,
        batch=16,
        device=0,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=True,
        **train_args,
    )
    archive_path = archive_best_weights(
        dataset_name,
        data_config,
        train_config_paths,
        val_config_paths,
        base_model,
        augment_preset_name,
        train_args,
        train_sample_info,
    )
    print(f"Archived best weights to {archive_path}")
    return archive_path


def main() -> None:
    archive_path = run_training(BASE_MODEL)
    print(f"Finished training with archived weights: {archive_path}")


if __name__ == "__main__":
    main()
