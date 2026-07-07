import curses
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

import train


BACK_OPTION = "返回"
NEXT_OPTION = "下一步"
EXIT_OPTION = "退出"


def ellipsize(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    if width <= 3:
        return "." * width
    left = max(1, (width - 3) // 2)
    right = max(1, width - 3 - left)
    return f"{text[:left]}...{text[-right:]}"


def draw_menu(
    stdscr,
    title: str,
    options: list[str],
    cursor: int,
    selected: set[int] | None = None,
    selectable_count: int | None = None,
) -> None:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    stdscr.addstr(0, 0, title)
    stdscr.addstr(1, 0, "↑/↓ 移动，Enter 确认/勾选，q 退出")
    visible_rows = max(1, height - 3)
    start = max(0, min(cursor - visible_rows + 1, max(0, len(options) - visible_rows)))
    selectable_count = len(options) if selectable_count is None else selectable_count
    for row, index in enumerate(range(start, min(len(options), start + visible_rows)), start=3):
        option = options[index]
        if row >= height:
            break
        marker = ""
        if selected is not None and index < selectable_count:
            marker = "[x] " if index in selected else "[ ] "
        prefix = "> " if index == cursor else "  "
        available_width = max(1, width - len(prefix) - len(marker) - 1)
        line = f"{prefix}{marker}{ellipsize(option, available_width)}"
        stdscr.addstr(row, 0, line[: max(1, width - 1)])
    stdscr.refresh()


def choose_one(stdscr, title: str, options: list[str], allow_back: bool = True) -> str | None:
    cursor = 0
    display_options = options + ([BACK_OPTION] if allow_back else [])
    while True:
        draw_menu(stdscr, title, display_options, cursor)
        key = stdscr.getch()
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == curses.KEY_UP:
            cursor = (cursor - 1) % len(display_options)
        elif key == curses.KEY_DOWN:
            cursor = (cursor + 1) % len(display_options)
        elif key in (curses.KEY_ENTER, 10, 13):
            if allow_back and cursor == len(display_options) - 1:
                return None
            return display_options[cursor]


def choose_many(stdscr, title: str, options: list[str], allow_back: bool = True) -> list[str] | None:
    cursor = 0
    selected: set[int] = set()
    display_options = options + [NEXT_OPTION] + ([BACK_OPTION] if allow_back else [])
    next_index = len(options)
    back_index = len(display_options) - 1 if allow_back else None
    while True:
        draw_menu(stdscr, title, display_options, cursor, selected, selectable_count=len(options))
        key = stdscr.getch()
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == curses.KEY_UP:
            cursor = (cursor - 1) % len(display_options)
        elif key == curses.KEY_DOWN:
            cursor = (cursor + 1) % len(display_options)
        elif key in (curses.KEY_ENTER, 10, 13):
            if cursor == next_index:
                if selected:
                    return [options[index] for index in sorted(selected)]
                continue
            if back_index is not None and cursor == back_index:
                return None
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
        preset = metadata.get("augment_preset", "?")
        sources = ",".join(metadata.get("dataset_sources") or [metadata.get("dataset_name", "?")])
        created_at = metadata.get("created_at", "?")
        label = f"model={base_model} | preset={preset} | datasets={sources} | time={created_at} | file={weight_path.name}"
        options.append((label, weight_path))
    return options


def metadata_for_weight(weight_path: Path, root: Path = train.ARCHIVE_DIR) -> list[Path]:
    matches: list[Path] = []
    for metadata_path in sorted(root.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if Path(metadata.get("weights", "")).name == weight_path.name:
            matches.append(metadata_path)
    return matches


def move_model_files(weight_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir = weight_path.parent
    related = [weight_path] + metadata_for_weight(weight_path, source_dir)
    for source in related:
        if not source.exists():
            continue
        target = target_dir / source.name
        if target.exists():
            raise FileExistsError(f"目标已存在: {target}")
        shutil.move(str(source), str(target))


def active_model_options() -> list[tuple[str, Path]]:
    return archive_options()


def disabled_model_options() -> list[tuple[str, Path]]:
    options: list[tuple[str, Path]] = []
    for weight_path in sorted(train.DISABLED_ARCHIVE_DIR.glob("*.pt")):
        label = f"disabled | file={weight_path.name}"
        options.append((label, weight_path))
    return options


def lastrun_options() -> list[tuple[str, Path]]:
    weights_dir = train.ROOT / "runs" / "detect" / "train" / "weights"
    return [
        ("last.pt", weights_dir / "last.pt"),
        ("best.pt", weights_dir / "best.pt"),
    ]


def augment_preset_choices() -> list[tuple[str, str]]:
    return [
        (f"{name} | {preset['description']}", name)
        for name, preset in train.AUGMENT_PRESETS.items()
    ]


def model_family_choices() -> list[tuple[str, str]]:
    return [
        (f"{family} | {config['description']}", family)
        for family, config in train.MODEL_FAMILIES.items()
    ]


def validate_weights(weight_paths: list[Path]) -> None:
    train.run_batch_testing(weight_paths)


def choose_dataset_names(stdscr, title: str) -> list[str] | None:
    choices = dataset_choices()
    labels = [label for label, _ in choices]
    selected_labels = choose_many(stdscr, title, labels)
    if selected_labels is None:
        return None
    name_by_label = dict(choices)
    return [name_by_label[label] for label in selected_labels]


def choose_training_datasets(stdscr) -> tuple[list[str], list[str], Path, str] | None:
    while True:
        train_datasets = choose_dataset_names(stdscr, "选择训练集来源")
        if train_datasets is None:
            return None
        val_datasets = choose_dataset_names(stdscr, "选择验证集来源")
        if val_datasets is None:
            continue
        os.environ[train.TRAIN_DATASET_ENV] = os.pathsep.join(train_datasets)
        os.environ[train.VALID_DATASET_ENV] = os.pathsep.join(val_datasets)
        data_config, dataset_name, _, _ = train.training_data_config()
        return train_datasets, val_datasets, data_config, dataset_name


def choose_test_datasets(stdscr) -> tuple[list[str], Path, str] | None:
    test_datasets = choose_dataset_names(stdscr, "选择测试集来源")
    if test_datasets is None:
        return None
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


def training_menu(stdscr) -> None:
    while True:
        family_choices = model_family_choices()
        family_label = choose_one(stdscr, "训练 / 选择 YOLO 家族", [item[0] for item in family_choices])
        if family_label is None:
            return
        model_family = dict(family_choices)[family_label]
        model_size = choose_one(stdscr, f"训练 / 选择 {model_family} 尺寸", train.model_size_options(model_family))
        if model_size is None:
            continue
        base_model = train.resolve_base_model(model_family, model_size)
        preset_choices = augment_preset_choices()
        preset_label = choose_one(stdscr, "训练 / 选择增强参数配方", [item[0] for item in preset_choices])
        if preset_label is None:
            continue
        augment_preset = dict(preset_choices)[preset_label]
        selected = choose_training_datasets(stdscr)
        if selected is None:
            continue
        train_datasets, val_datasets, data_config, dataset_name = selected

        def train_action() -> None:
            print(f"Model family: {model_family}")
            print(f"Model size: {model_size}")
            print(f"Base model: {base_model}")
            print(f"Augment preset: {augment_preset}")
            print(f"Train datasets: {', '.join(train_datasets)}")
            print(f"Valid datasets: {', '.join(val_datasets)}")
            print(f"Dataset name: {dataset_name}")
            print(f"Dataset config: {data_config}")
            train.run_training(base_model, augment_preset)

        run_outside_curses(stdscr, train_action)
        return


def weight_paths_menu(stdscr) -> list[Path] | None:
    while True:
        source = choose_one(stdscr, "测试 / 选择权重来源", ["archieve", "lastrun"])
        if source is None:
            return None
        if source == "archieve":
            choices = archive_options()
            if not choices:
                raise FileNotFoundError("model_archieve 中没有可用权重")
            labels = choose_many(stdscr, "测试 / 选择一个或多个归档权重", [item[0] for item in choices])
        else:
            choices = lastrun_options()
            labels = choose_many(stdscr, "测试 / 选择一个或多个最近训练权重", [item[0] for item in choices])
        if labels is None:
            continue
        return [dict(choices)[label] for label in labels]


def testing_menu(stdscr) -> None:
    while True:
        weights_paths = weight_paths_menu(stdscr)
        if weights_paths is None:
            return
        selected = choose_test_datasets(stdscr)
        if selected is None:
            continue
        test_datasets, test_config, test_name = selected

        def validate_action() -> None:
            print("Test weights:")
            for weights_path in weights_paths:
                print(f"- {weights_path}")
            print(f"Test datasets: {', '.join(test_datasets)}")
            print(f"Test name: {test_name}")
            print(f"Test config: {test_config}")
            validate_weights(weights_paths)

        run_outside_curses(stdscr, validate_action)
        return


def model_management_menu(stdscr) -> None:
    while True:
        action = choose_one(stdscr, "模型管理", ["移入停用归档区", "从停用归档区恢复"])
        if action is None:
            return
        if action == "移入停用归档区":
            choices = active_model_options()
            if not choices:
                continue
            labels = choose_many(stdscr, "选择要停用的模型", [item[0] for item in choices])
            if labels is None:
                continue
            paths = [dict(choices)[label] for label in labels]

            def archive_action() -> None:
                for path in paths:
                    print(f"Archive model: {path.name}")
                    move_model_files(path, train.DISABLED_ARCHIVE_DIR)

            run_outside_curses(stdscr, archive_action)
        elif action == "从停用归档区恢复":
            choices = disabled_model_options()
            if not choices:
                continue
            labels = choose_many(stdscr, "选择要恢复的模型", [item[0] for item in choices])
            if labels is None:
                continue
            paths = [dict(choices)[label] for label in labels]

            def restore_action() -> None:
                for path in paths:
                    print(f"Restore model: {path.name}")
                    move_model_files(path, train.ARCHIVE_DIR)

            run_outside_curses(stdscr, restore_action)


def interactive(stdscr) -> None:
    curses.curs_set(0)
    while True:
        action = choose_one(stdscr, "YOLOv8 交互菜单", ["训练", "测试", "模型管理", EXIT_OPTION], allow_back=False)
        if action == "训练":
            training_menu(stdscr)
        elif action == "测试":
            testing_menu(stdscr)
        elif action == "模型管理":
            model_management_menu(stdscr)
        elif action == EXIT_OPTION:
            return


def main() -> None:
    curses.wrapper(interactive)


if __name__ == "__main__":
    main()
