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
    help_text: str | None = None,
) -> None:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    stdscr.addstr(0, 0, title)
    stdscr.addstr(1, 0, help_text or "↑/↓ 移动，Enter 确认/勾选，q 退出")
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


def show_detail(stdscr, title: str, text: str) -> None:
    lines = text.splitlines() or ["无详情"]
    cursor = 0
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, title[: max(1, width - 1)])
        stdscr.addstr(1, 0, "↑/↓ 滚动，Enter/Esc 返回")
        visible_rows = max(1, height - 3)
        cursor = max(0, min(cursor, max(0, len(lines) - visible_rows)))
        for row, index in enumerate(range(cursor, min(len(lines), cursor + visible_rows)), start=3):
            stdscr.addstr(row, 0, lines[index][: max(1, width - 1)])
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            cursor = max(0, cursor - 1)
        elif key == curses.KEY_DOWN:
            cursor = min(max(0, len(lines) - visible_rows), cursor + 1)
        elif key in (curses.KEY_ENTER, 10, 13, 27):
            return


def prompt_text(stdscr, title: str, prompt: str) -> str | None:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    stdscr.addstr(0, 0, title[: max(1, width - 1)])
    stdscr.addstr(1, 0, "直接 Enter 留空；Esc 返回")
    stdscr.addstr(3, 0, prompt[: max(1, width - 1)])
    stdscr.addstr(4, 0, "> ")
    stdscr.refresh()
    curses.echo()
    try:
        text = stdscr.getstr(4, 2, max(1, width - 3)).decode("utf-8", errors="replace").strip()
    except KeyboardInterrupt:
        return None
    finally:
        curses.noecho()
    return text


def choose_one(
    stdscr,
    title: str,
    options: list[str],
    allow_back: bool = True,
    details: dict[str, str] | None = None,
) -> str | None:
    cursor = 0
    display_options = options + ([BACK_OPTION] if allow_back else [])
    while True:
        draw_menu(stdscr, title, display_options, cursor, help_text="↑/↓ 移动，Enter 确认，d 详情，q 退出")
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
        elif key == ord("d") and cursor < len(options) and details:
            show_detail(stdscr, "详情", details.get(display_options[cursor], "无详情"))


def choose_many(
    stdscr,
    title: str,
    options: list[str],
    allow_back: bool = True,
    details: dict[str, str] | None = None,
    adjust=None,
) -> list[str] | None:
    cursor = 0
    selected: set[int] = set()
    display_options = options + [NEXT_OPTION] + ([BACK_OPTION] if allow_back else [])
    next_index = len(options)
    back_index = len(display_options) - 1 if allow_back else None
    while True:
        draw_menu(
            stdscr,
            title,
            display_options,
            cursor,
            selected,
            selectable_count=len(options),
            help_text="↑/↓ 移动，Enter 勾选，+/- 调倍率，d 详情，下一步继续",
        )
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
        elif key == ord("d") and cursor < len(options) and details:
            show_detail(stdscr, "详情", details.get(display_options[cursor], "无详情"))
        elif key in (ord("+"), ord("="), ord("-"), ord("_")) and cursor < len(options) and adjust:
            new_label = adjust(display_options[cursor], 1 if key in (ord("+"), ord("=")) else -1)
            if new_label and new_label != display_options[cursor]:
                options[cursor] = new_label
                display_options[cursor] = new_label


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


def dataset_detail(value: str) -> str:
    config_path, explicit_split = train.parse_dataset_sources(value, "interactive")[0]
    metadata = train.dataset_metadata(config_path)
    lines = [
        f"名称: {value}",
        f"目录: {config_path.parent}",
        f"配置: {config_path}",
        f"版本: {metadata.get('version') or 'unknown'}",
        f"类别: {metadata.get('class_names')}",
    ]
    for split in ["train", "val", "valid", "test", train.GOLDEN_SPLIT]:
        try:
            split_path = train.resolve_split(config_path, train.load_yaml(config_path), split)
        except Exception as error:
            lines.append(f"{split}: error {error}")
            continue
        if split_path and split_path.exists():
            image_count = len([path for path in split_path.rglob("*") if path.suffix.lower() in train.IMAGE_SUFFIXES])
            mark = " <- 当前特殊子集" if explicit_split == split else ""
            lines.append(f"{split}: {split_path} ({image_count} images){mark}")
    roboflow = metadata.get("roboflow") or {}
    if roboflow:
        lines.append("roboflow:")
        lines.extend(f"  {key}: {value}" for key, value in roboflow.items())
    return "\n".join(lines)


def model_detail(weight_path: Path) -> str:
    metadata_paths = metadata_for_weight(weight_path, weight_path.parent)
    if not metadata_paths:
        return f"权重: {weight_path}\n元数据: 未找到"
    try:
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return f"权重: {weight_path}\n元数据读取失败: {error}"
    return json.dumps(metadata, ensure_ascii=False, indent=2)


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
        label = f"{weight_path.name} | datasets={sources} | base={base_model} | time={created_at} | preset={preset}"
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
    details = {label: dataset_detail(value) for label, value in choices}
    selected_labels = choose_many(stdscr, title, labels, details=details)
    if selected_labels is None:
        return None
    name_by_label = dict(choices)
    return [name_by_label[label] for label in selected_labels]


def multiplier_label(label: str, multiplier: float, kind: str = "train") -> str:
    return f"{label} | {kind} x{multiplier:g}"


def choose_train_dataset_names(stdscr, title: str) -> tuple[list[str], dict[str, float]] | None:
    choices = dataset_choices()
    values = [value for _, value in choices]
    base_labels = {value: label for label, value in choices}
    multipliers = {value: 1.0 for value in values}

    def current_labels() -> list[str]:
        return [multiplier_label(base_labels[value], multipliers[value], "train") for value in values]

    labels = current_labels()
    value_by_label = {label: value for label, value in zip(labels, values)}
    details = {label: dataset_detail(value) for label, value in value_by_label.items()}

    def adjust(label: str, direction: int) -> str:
        value = value_by_label[label]
        choices = train.sample_choice_options()
        index = min(range(len(choices)), key=lambda i: abs(choices[i] - multipliers[value]))
        index = max(0, min(len(choices) - 1, index + direction))
        multipliers[value] = choices[index]
        new_label = multiplier_label(base_labels[value], multipliers[value], "train")
        value_by_label[new_label] = value
        details[new_label] = dataset_detail(value)
        return new_label

    selected_labels = choose_many(stdscr, title, labels, details=details, adjust=adjust)
    if selected_labels is None:
        return None
    selected_values = [value_by_label[label] for label in selected_labels]
    selected_multipliers = {
        train.source_name(train.parse_dataset_sources(value, "interactive")[0], "train"): multipliers[value]
        for value in selected_values
    }
    return selected_values, selected_multipliers


def choose_test_dataset_names(stdscr, title: str) -> tuple[list[str], dict[str, float]] | None:
    choices = dataset_choices()
    values = [value for _, value in choices]
    base_labels = {value: label for label, value in choices}
    multipliers = {value: 1.0 for value in values}

    labels = [multiplier_label(base_labels[value], multipliers[value], "test") for value in values]
    value_by_label = {label: value for label, value in zip(labels, values)}
    details = {label: dataset_detail(value) for label, value in value_by_label.items()}

    def adjust(label: str, direction: int) -> str:
        value = value_by_label[label]
        choices = train.sample_choice_options()
        index = min(range(len(choices)), key=lambda i: abs(choices[i] - multipliers[value]))
        index = max(0, min(len(choices) - 1, index + direction))
        multipliers[value] = choices[index]
        new_label = multiplier_label(base_labels[value], multipliers[value], "test")
        value_by_label[new_label] = value
        details[new_label] = dataset_detail(value)
        return new_label

    selected_labels = choose_many(stdscr, title, labels, details=details, adjust=adjust)
    if selected_labels is None:
        return None
    selected_values = [value_by_label[label] for label in selected_labels]
    selected_multipliers = {
        train.source_name(train.parse_dataset_sources(value, "interactive")[0], "test"): multipliers[value]
        for value in selected_values
    }
    return selected_values, selected_multipliers


def choose_training_datasets(stdscr) -> tuple[list[str], list[str], Path, str, dict[str, float]] | None:
    while True:
        train_selection = choose_train_dataset_names(stdscr, "选择训练集来源")
        if train_selection is None:
            return None
        train_datasets, sample_multipliers = train_selection
        val_datasets = choose_dataset_names(stdscr, "选择验证集来源")
        if val_datasets is None:
            continue
        os.environ[train.TRAIN_DATASET_ENV] = os.pathsep.join(train_datasets)
        os.environ[train.VALID_DATASET_ENV] = os.pathsep.join(val_datasets)
        os.environ[train.TRAIN_SAMPLE_ENV] = json.dumps(sample_multipliers)
        data_config, dataset_name, _, _, _ = train.training_data_config()
        return train_datasets, val_datasets, data_config, dataset_name, sample_multipliers


def choose_test_datasets(stdscr) -> tuple[list[str], Path, str, dict[str, float]] | None:
    selected = choose_test_dataset_names(stdscr, "选择测试集来源")
    if selected is None:
        return None
    test_datasets, sample_multipliers = selected
    os.environ[train.TEST_DATASET_ENV] = os.pathsep.join(test_datasets)
    os.environ[train.TEST_SAMPLE_ENV] = json.dumps(sample_multipliers)
    test_config, test_name, _, _ = train.testing_data_config()
    return test_datasets, test_config, test_name, sample_multipliers


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
        model_note = prompt_text(stdscr, "训练 / 自定义批注", "批注会替换文件名里的随机名字字段：")
        if model_note is None:
            continue
        selected = choose_training_datasets(stdscr)
        if selected is None:
            continue
        train_datasets, val_datasets, data_config, dataset_name, sample_multipliers = selected

        def train_action() -> None:
            print(f"Model family: {model_family}")
            print(f"Model size: {model_size}")
            print(f"Base model: {base_model}")
            print(f"Augment preset: {augment_preset}")
            print(f"Model note: {model_note or '(random)'}")
            print(f"Train datasets: {', '.join(train_datasets)}")
            print(f"Train sample multipliers: {json.dumps(sample_multipliers, ensure_ascii=False, sort_keys=True)}")
            print(f"Valid datasets: {', '.join(val_datasets)}")
            print(f"Dataset name: {dataset_name}")
            print(f"Dataset config: {data_config}")
            train.run_training(base_model, augment_preset, model_note)

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
            details = {label: model_detail(path) for label, path in choices}
            labels = choose_many(stdscr, "测试 / 选择一个或多个归档权重", [item[0] for item in choices], details=details)
        else:
            choices = lastrun_options()
            details = {label: f"权重: {path}\n存在: {path.exists()}" for label, path in choices}
            labels = choose_many(stdscr, "测试 / 选择一个或多个最近训练权重", [item[0] for item in choices], details=details)
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
        test_datasets, test_config, test_name, sample_multipliers = selected

        def validate_action() -> None:
            print("Test weights:")
            for weights_path in weights_paths:
                print(f"- {weights_path}")
            print(f"Test datasets: {', '.join(test_datasets)}")
            print(f"Test sample multipliers: {json.dumps(sample_multipliers, ensure_ascii=False, sort_keys=True)}")
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
            details = {label: model_detail(path) for label, path in choices}
            labels = choose_many(stdscr, "选择要停用的模型", [item[0] for item in choices], details=details)
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
            details = {label: model_detail(path) for label, path in choices}
            labels = choose_many(stdscr, "选择要恢复的模型", [item[0] for item in choices], details=details)
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
