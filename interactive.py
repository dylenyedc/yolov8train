import curses
import json
import os
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

import train


BASE_MODEL_CHOICES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


def draw_menu(stdscr, title: str, options: list[str], cursor: int, selected: set[int] | None = None) -> None:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    stdscr.addstr(0, 0, title)
    stdscr.addstr(1, 0, "↑/↓ 移动，Enter 确认/勾选，q 退出")
    visible_rows = max(1, height - 3)
    start = max(0, min(cursor - visible_rows + 1, max(0, len(options) - visible_rows)))
    for row, index in enumerate(range(start, min(len(options), start + visible_rows)), start=3):
        option = options[index]
        if row >= height:
            break
        marker = ""
        if selected is not None:
            marker = "[x] " if index in selected else "[ ] "
        prefix = "> " if index == cursor else "  "
        line = f"{prefix}{marker}{option}"
        stdscr.addstr(row, 0, line[: max(1, width - 1)])
    stdscr.refresh()


def choose_one(stdscr, title: str, options: list[str]) -> str:
    cursor = 0
    while True:
        draw_menu(stdscr, title, options, cursor)
        key = stdscr.getch()
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == curses.KEY_UP:
            cursor = (cursor - 1) % len(options)
        elif key == curses.KEY_DOWN:
            cursor = (cursor + 1) % len(options)
        elif key in (curses.KEY_ENTER, 10, 13):
            return options[cursor]


def choose_many(stdscr, title: str, options: list[str]) -> list[str]:
    cursor = 0
    selected: set[int] = set()
    display_options = options + ["Next"]
    while True:
        draw_menu(stdscr, title, display_options, cursor, selected)
        key = stdscr.getch()
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == curses.KEY_UP:
            cursor = (cursor - 1) % len(display_options)
        elif key == curses.KEY_DOWN:
            cursor = (cursor + 1) % len(display_options)
        elif key in (curses.KEY_ENTER, 10, 13):
            if cursor == len(display_options) - 1:
                if selected:
                    return [options[index] for index in sorted(selected)]
                continue
            if cursor in selected:
                selected.remove(cursor)
            else:
                selected.add(cursor)


def dataset_last_updated(dataset_dir: Path) -> str:
    newest = max((path.stat().st_mtime for path in dataset_dir.rglob("*") if path.is_file()), default=0)
    if newest <= 0:
        return "unknown"
    return datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")


def dataset_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for config_path in sorted(train.DATASETS_DIR.glob("*/data.yaml")):
        name = config_path.parent.name
        label = f"{name} | updated {dataset_last_updated(config_path.parent)}"
        choices.append((label, name))
        golden_dir = config_path.parent / train.GOLDEN_SPLIT / "images"
        if golden_dir.exists():
            golden_label = f"{name}/{train.GOLDEN_SPLIT} | updated {dataset_last_updated(golden_dir.parent)}"
            choices.append((golden_label, f"{name}@{train.GOLDEN_SPLIT}"))
    return choices


def archive_options() -> list[tuple[str, Path]]:
    options: list[tuple[str, Path]] = []
    metadata_by_weight: dict[str, dict] = {}
    for metadata_path in train.ARCHIVE_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata_by_weight[Path(metadata.get("weights", "")).name] = metadata

    for weight_path in sorted(train.ARCHIVE_DIR.glob("*.pt")):
        metadata = metadata_by_weight.get(weight_path.name, {})
        base_model = metadata.get("base_model", "?")
        sources = ", ".join(metadata.get("dataset_sources") or [metadata.get("dataset_name", "?")])
        created_at = metadata.get("created_at", "?")
        label = f"{weight_path.name} | model={base_model} | datasets={sources} | time={created_at}"
        options.append((label, weight_path))
    return options


def lastrun_options() -> list[tuple[str, Path]]:
    weights_dir = train.ROOT / "runs" / "detect" / "train" / "weights"
    return [
        ("last.pt", weights_dir / "last.pt"),
        ("best.pt", weights_dir / "best.pt"),
    ]


def validate_weights(weights_path: Path, test_config: Path) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"权重不存在: {weights_path}")
    model = YOLO(str(weights_path))
    model.val(
        data=str(test_config),
        split="test",
        device=0,
        project=str(train.ROOT / "runs" / "detect"),
        name="test",
        exist_ok=True,
    )


def choose_dataset_names(stdscr, title: str) -> list[str]:
    choices = dataset_choices()
    labels = [label for label, _ in choices]
    selected_labels = choose_many(stdscr, title, labels)
    name_by_label = dict(choices)
    return [name_by_label[label] for label in selected_labels]


def choose_training_datasets(stdscr) -> tuple[list[str], list[str], Path, str]:
    train_datasets = choose_dataset_names(stdscr, "选择训练集来源")
    val_datasets = choose_dataset_names(stdscr, "选择验证集来源")
    os.environ[train.TRAIN_DATASET_ENV] = os.pathsep.join(train_datasets)
    os.environ[train.VALID_DATASET_ENV] = os.pathsep.join(val_datasets)
    data_config, dataset_name, _, _ = train.training_data_config()
    return train_datasets, val_datasets, data_config, dataset_name


def choose_test_datasets(stdscr) -> tuple[list[str], Path, str]:
    test_datasets = choose_dataset_names(stdscr, "选择测试集来源")
    os.environ[train.TEST_DATASET_ENV] = os.pathsep.join(test_datasets)
    test_config, test_name, _ = train.testing_data_config()
    return test_datasets, test_config, test_name


def run_outside_curses(stdscr, action) -> None:
    curses.def_prog_mode()
    curses.endwin()
    try:
        action()
        input("按 Enter 返回交互菜单...")
    finally:
        curses.reset_prog_mode()
        curses.curs_set(0)
        stdscr.refresh()


def interactive(stdscr) -> None:
    curses.curs_set(0)
    should_train = choose_one(stdscr, "是否开始训练？", ["yes", "no"]) == "yes"

    if should_train:
        base_model = choose_one(stdscr, "选择基础模型", BASE_MODEL_CHOICES)
        train_datasets, val_datasets, data_config, dataset_name = choose_training_datasets(stdscr)

        def train_action() -> None:
            print(f"Base model: {base_model}")
            print(f"Train datasets: {', '.join(train_datasets)}")
            print(f"Valid datasets: {', '.join(val_datasets)}")
            print(f"Dataset name: {dataset_name}")
            print(f"Dataset config: {data_config}")
            train.run_training(base_model)

        run_outside_curses(stdscr, train_action)

    should_validate = choose_one(stdscr, "是否进行验证？", ["yes", "no"]) == "yes"
    if not should_validate:
        return

    source = choose_one(stdscr, "选择验证权重来源", ["archieve", "lastrun"])
    if source == "archieve":
        choices = archive_options()
        if not choices:
            raise FileNotFoundError("model_archieve 中没有可用权重")
        label = choose_one(stdscr, "选择归档权重", [item[0] for item in choices])
        weights_path = dict(choices)[label]
    else:
        choices = lastrun_options()
        label = choose_one(stdscr, "选择最近一次训练权重", [item[0] for item in choices])
        weights_path = dict(choices)[label]

    test_datasets, test_config, test_name = choose_test_datasets(stdscr)

    def validate_action() -> None:
        print(f"Validate weights: {weights_path}")
        print(f"Test datasets: {', '.join(test_datasets)}")
        print(f"Test name: {test_name}")
        print(f"Test config: {test_config}")
        validate_weights(weights_path, test_config)

    run_outside_curses(stdscr, validate_action)


def main() -> None:
    curses.wrapper(interactive)


if __name__ == "__main__":
    main()
