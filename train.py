import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "white_cylinder_detection"
GENERATED_DIR = ROOT / ".generated"
ARCHIVE_DIR = ROOT / "model_archieve"
BASE_MODEL = "yolov8n"
DATASET_ENV = "YOLO_DATASET_PATH"
TRAIN_DATASET_ENV = "YOLO_TRAIN_DATASET_PATH"
VAL_DATASET_ENV = "YOLO_VAL_DATASET_PATH"
VALID_DATASET_ENV = "YOLO_VALID_DATASET_PATH"
TEST_DATASET_ENV = "YOLO_TEST_DATASET_PATH"
MIXED_DATASET_CLASS_NAME = "busket"
GOLDEN_SPLIT = "golden"


def safe_name(value: object) -> str:
    text = str(value or "dataset").strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("._-") or "dataset"


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
) -> Path:
    train_name = safe_name("_".join(source_names(train_sources, "train")))
    val_name = safe_name("_".join(source_names(val_sources, "val")))
    output_name = safe_name(f"cross_{train_name}_valid_{val_name}")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DIR / f"{output_name}.yaml"
    output = {
        "train": source_split_paths(train_sources, "train"),
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
    return output_path


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


def training_data_config() -> tuple[Path, str, list[Path], list[Path]]:
    train_value = os.environ.get(TRAIN_DATASET_ENV)
    val_value = os.environ.get(VALID_DATASET_ENV) or os.environ.get(VAL_DATASET_ENV)
    if train_value is not None or val_value is not None:
        train_sources = parse_dataset_sources(train_value or str(DEFAULT_DATASET), TRAIN_DATASET_ENV)
        val_sources = parse_dataset_sources(val_value or train_value or str(DEFAULT_DATASET), VAL_DATASET_ENV)
        train_paths = list(dict.fromkeys(config_path for config_path, _ in train_sources))
        val_paths = list(dict.fromkeys(config_path for config_path, _ in val_sources))
        dataset_name = safe_name(
            f"train_{'_'.join(source_names(train_sources, 'train'))}"
            f"_valid_{'_'.join(source_names(val_sources, 'val'))}"
        )
        return cross_validation_dataset_config(train_sources, val_sources), dataset_name, train_paths, val_paths

    config_paths = selected_dataset_configs()
    if len(config_paths) == 1:
        return config_paths[0], dataset_id(config_paths[0]), config_paths, config_paths
    return mixed_dataset_config(config_paths), combined_dataset_name(config_paths), config_paths, config_paths


def archive_best_weights(
    dataset_name: str,
    data_config: Path,
    train_config_paths: list[Path],
    val_config_paths: list[Path],
    base_model: str = BASE_MODEL,
) -> Path:
    best_weights = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished, but best.pt was not found: {best_weights}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    archive_name = f"{base_model}_{now:%Y%m%d_%H%M%S}_{dataset_name}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy2(best_weights, archive_path)

    metadata = {
        "created_at": now.isoformat(timespec="microseconds"),
        "base_model": base_model,
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


def run_training(base_model: str = BASE_MODEL) -> Path:
    data_config, dataset_name, train_config_paths, val_config_paths = training_data_config()
    print(f"Using dataset config: {data_config}")
    model = YOLO(f"{base_model}.pt")
    model.train(
        data=str(data_config),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=str(ROOT / "runs" / "detect"),
        name="train",
        exist_ok=True,
    )
    archive_path = archive_best_weights(dataset_name, data_config, train_config_paths, val_config_paths, base_model)
    print(f"Archived best weights to {archive_path}")
    return archive_path


def main() -> None:
    archive_path = run_training(BASE_MODEL)
    print(f"Finished training with archived weights: {archive_path}")


if __name__ == "__main__":
    main()
