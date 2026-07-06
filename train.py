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
BASE_MODEL = "yolov8s"
DATASET_ENV = "YOLO_DATASET_PATH"


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
        path = ROOT / path
    if path.is_dir():
        path = path / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config was not found: {path}")
    return path.resolve()


def selected_dataset_configs() -> list[Path]:
    value = os.environ.get(DATASET_ENV, str(DEFAULT_DATASET))
    raw_paths = [
        part.strip()
        for chunk in value.split(os.pathsep)
        for part in chunk.split(",")
        if part.strip()
    ]
    return [dataset_config_path(Path(path)) for path in raw_paths]


def resolve_split(config_path: Path, config: dict, split: str) -> Path | None:
    value = config.get(split)
    if not value:
        return None
    if isinstance(value, list):
        raise ValueError(f"{config_path} already contains a list for {split}; nested mixing is not supported")
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(config.get("path") or config_path.parent)
    if not base.is_absolute():
        base = config_path.parent / base
    return (base / path).resolve()


def dataset_id(config_path: Path) -> str:
    return safe_name(config_path.parent.name)


def dataset_metadata(config_path: Path) -> dict:
    config = load_yaml(config_path)
    roboflow = config.get("roboflow") or {}
    return {
        "dataset_name": dataset_id(config_path),
        "config_path": str(config_path),
        "class_names": config.get("names"),
        "roboflow": roboflow,
        "version": roboflow.get("version") or config.get("version"),
    }


def mixed_dataset_config(config_paths: list[Path]) -> Path:
    names = [dataset_id(path) for path in config_paths]
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
        "names": ["busket"],
    }
    if test_paths:
        output["test"] = test_paths
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(output, file, sort_keys=False, allow_unicode=True)
    return output_path


def training_data_config() -> tuple[Path, str, list[Path]]:
    config_paths = selected_dataset_configs()
    if len(config_paths) == 1:
        return config_paths[0], dataset_id(config_paths[0]), config_paths
    dataset_name = safe_name("mixed_" + "_".join(dataset_id(path) for path in config_paths))
    return mixed_dataset_config(config_paths), dataset_name, config_paths


def archive_best_weights(dataset_name: str, data_config: Path, config_paths: list[Path]) -> Path:
    best_weights = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished, but best.pt was not found: {best_weights}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    archive_name = f"{BASE_MODEL}_{now:%Y%m%d_%H%M%S}_{dataset_name}.pt"
    archive_path = ARCHIVE_DIR / archive_name
    shutil.copy2(best_weights, archive_path)

    metadata = {
        "created_at": now.isoformat(timespec="microseconds"),
        "base_model": BASE_MODEL,
        "weights": str(archive_path),
        "source_weights": str(best_weights),
        "training_data_config": str(data_config),
        "dataset_name": dataset_name,
        "datasets": [dataset_metadata(path) for path in config_paths],
    }
    metadata_path = ARCHIVE_DIR / f"{now:%Y%m%d_%H%M%S_%f}.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return archive_path


def main() -> None:
    data_config, dataset_name, config_paths = training_data_config()
    print(f"Using dataset config: {data_config}")
    model = YOLO(f"{BASE_MODEL}.pt")
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
    archive_path = archive_best_weights(dataset_name, data_config, config_paths)
    print(f"Archived best weights to {archive_path}")


if __name__ == "__main__":
    main()
