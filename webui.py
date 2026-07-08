import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file

import train


APP = Flask(__name__)
MAX_LOG_LINES = 2000

STATE = {
    "running": False,
    "kind": None,
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "logs": [],
}
STATE_LOCK = threading.Lock()
CURRENT_PROCESS: subprocess.Popen | None = None
CURRENT_PROCESS_LOCK = threading.Lock()


PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLOv8 训练控制台</title>
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }
    header { padding: 18px 24px; background: #ffffff; border-bottom: 1px solid #d8dee6; }
    h1 { margin: 0; font-size: 22px; }
    main { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 16px; padding: 16px; }
    section { background: #ffffff; border: 1px solid #d8dee6; border-radius: 8px; padding: 14px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    h3 { margin: 14px 0 8px; font-size: 14px; }
    label { display: block; margin: 6px 0; line-height: 1.35; }
    select, button { font: inherit; }
    select { width: 100%; padding: 8px; border: 1px solid #c7d0da; border-radius: 6px; background: white; }
    select[multiple] { min-height: 180px; }
    button { border: 1px solid #243b53; border-radius: 6px; padding: 8px 12px; color: #fff; background: #243b53; cursor: pointer; }
    button.secondary { color: #243b53; background: #fff; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .choices { max-height: 210px; overflow: auto; border: 1px solid #e1e7ef; border-radius: 6px; padding: 8px; background: #fbfcfd; }
    .hint { color: #627386; font-size: 13px; }
    .runs { display: grid; gap: 12px; }
    .run { border: 1px solid #e1e7ef; border-radius: 8px; padding: 10px; background: #fbfcfd; }
    .model-list { display: grid; gap: 8px; max-height: 360px; overflow: auto; }
    .model-row { border: 1px solid #e1e7ef; border-radius: 6px; padding: 8px; background: #fbfcfd; }
    .model-row strong { display: block; overflow-wrap: anywhere; margin-bottom: 4px; }
    .model-row .meta { color: #627386; font-size: 12px; overflow-wrap: anywhere; margin-bottom: 8px; }
    .run-title { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
    .files { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
    .file { color: #243b53; text-decoration: none; border: 1px solid #c7d0da; border-radius: 6px; padding: 5px 7px; background: #fff; font-size: 12px; }
    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
    .shot { border: 1px solid #d8dee6; border-radius: 6px; background: #fff; padding: 6px; }
    .shot img { width: 100%; height: 130px; object-fit: contain; display: block; background: #f1f4f8; }
    .shot a { display: block; color: #243b53; font-size: 12px; margin-top: 4px; overflow-wrap: anywhere; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
    th, td { border: 1px solid #d8dee6; padding: 4px 6px; text-align: left; }
    th { background: #edf2f7; }
    .status { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
    .stat { border: 1px solid #e1e7ef; border-radius: 6px; padding: 8px; background: #fbfcfd; }
    .stat strong { display: block; font-size: 12px; color: #627386; }
    pre { margin: 0; min-height: 520px; max-height: 72vh; overflow: auto; white-space: pre-wrap; background: #101820; color: #dce6f2; border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.45; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>YOLOv8 训练控制台</h1>
    <div class="hint">选择训练集、验证集、测试集，后台日志会显示在右侧终端状态中。</div>
  </header>
  <main>
    <div>
      <section>
        <h2>训练</h2>
        <label>YOLO 家族</label>
        <select id="modelFamily"></select>
        <label>模型尺寸</label>
        <select id="modelSize"></select>
        <label>增强参数配方</label>
        <select id="trainPreset"></select>
        <h3>训练集来源</h3>
        <div id="trainSources" class="choices"></div>
        <h3>验证集来源</h3>
        <div id="validSources" class="choices"></div>
        <p class="hint">若存在 <code>golden/images</code>，会显示 <code>dataset/golden</code> 选项。</p>
        <button id="startTrain">开始训练</button>
      </section>

      <section style="margin-top:16px">
        <h2>批量测试</h2>
        <label>权重来源，可多选</label>
        <select id="weightSelect" multiple></select>
        <h3>测试集来源</h3>
        <div id="testSources" class="choices"></div>
        <div class="row" style="margin-top:12px">
          <button id="startValidate">开始验证</button>
          <button id="refreshOptions" class="secondary">刷新选项</button>
        </div>
      </section>

      <section style="margin-top:16px">
        <h2>模型管理</h2>
        <h3>有效模型</h3>
        <div id="activeModels" class="model-list"></div>
        <h3>停用归档区</h3>
        <div id="disabledModels" class="model-list"></div>
        <div class="row" style="margin-top:12px">
          <button id="refreshModels" class="secondary">刷新模型</button>
        </div>
      </section>
    </div>

    <section>
      <h2>终端状态</h2>
      <div class="status">
        <div class="stat"><strong>状态</strong><span id="statusRunning">-</span></div>
        <div class="stat"><strong>任务</strong><span id="statusKind">-</span></div>
        <div class="stat"><strong>开始</strong><span id="statusStarted">-</span></div>
        <div class="stat"><strong>退出码</strong><span id="statusCode">-</span></div>
      </div>
      <div class="row" style="margin-bottom:10px">
        <button id="stopTask">急停</button>
        <button id="clearLogs" class="secondary">清空显示</button>
      </div>
      <pre id="logs"></pre>
    </section>

    <section style="grid-column: 1 / -1">
      <div class="row" style="justify-content:space-between; margin-bottom:10px">
        <h2>运行结果</h2>
        <button id="refreshRuns" class="secondary">刷新结果</button>
      </div>
      <div id="runs" class="runs"></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);

    function checkedValues(containerId) {
      return Array.from(document.querySelectorAll(`#${containerId} input:checked`)).map((item) => item.value);
    }

    function selectedValues(selectId) {
      return Array.from($(selectId).selectedOptions).map((item) => item.value);
    }

    function renderChecks(containerId, options, withSamples=false) {
      const root = $(containerId);
      root.innerHTML = "";
      options.forEach((option) => {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = option.value;
        label.appendChild(input);
        label.append(" " + option.label);
        if (withSamples) {
          const select = document.createElement("select");
          select.dataset.sampleFor = option.value;
          select.style.width = "88px";
          select.style.marginLeft = "8px";
          (window.sampleChoices || [1]).forEach((value) => {
            const choice = document.createElement("option");
            choice.value = String(value);
            choice.textContent = `${value}x`;
            if (Number(value) === 1) choice.selected = true;
            select.appendChild(choice);
          });
          label.appendChild(select);
        }
        root.appendChild(label);
      });
    }

    function checkedSampleMultipliers(containerId) {
      const payload = {};
      Array.from(document.querySelectorAll(`#${containerId} input:checked`)).forEach((input) => {
        const select = document.querySelector(`#${containerId} select[data-sample-for="${CSS.escape(input.value)}"]`);
        if (select) payload[input.value.replaceAll("@", "_")] = Number(select.value);
      });
      return payload;
    }

    async function loadOptions() {
      const response = await fetch("/api/options");
      const data = await response.json();
      window.modelFamilies = data.model_families;
      window.sampleChoices = data.sample_choices;
      $("modelFamily").innerHTML = data.model_families.map((item) => `<option value="${item.value}">${item.label}</option>`).join("");
      updateModelSizes();
      $("trainPreset").innerHTML = data.training_presets.map((item) => `<option value="${item.value}">${item.label}</option>`).join("");
      renderChecks("trainSources", data.datasets, true);
      renderChecks("validSources", data.datasets);
      renderChecks("testSources", data.datasets);
      $("weightSelect").innerHTML = data.weights.map((item) => `<option value="${item.value}">${item.label}</option>`).join("");
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) alert(data.error || "请求失败");
      return data;
    }

    $("startTrain").onclick = () => postJson("/api/train", {
      model_family: $("modelFamily").value,
      model_size: $("modelSize").value,
      augment_preset: $("trainPreset").value,
      train_sources: checkedValues("trainSources"),
      train_sample_multipliers: checkedSampleMultipliers("trainSources"),
      valid_sources: checkedValues("validSources"),
    });

    $("startValidate").onclick = () => postJson("/api/validate", {
      weights: selectedValues("weightSelect"),
      test_sources: checkedValues("testSources"),
    });

    $("refreshOptions").onclick = loadOptions;
    $("modelFamily").onchange = updateModelSizes;
    $("refreshRuns").onclick = loadRuns;
    $("refreshModels").onclick = loadModels;
    $("stopTask").onclick = () => postJson("/api/stop", {});
    $("clearLogs").onclick = () => { $("logs").textContent = ""; };

    function csvTable(rows) {
      if (!rows || rows.length === 0) return "";
      const head = rows[0].map((cell) => `<th>${cell}</th>`).join("");
      const body = rows.slice(1).map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function updateModelSizes() {
      const families = window.modelFamilies || [];
      const selected = families.find((item) => item.value === $("modelFamily").value) || families[0];
      $("modelSize").innerHTML = selected ? selected.sizes.map((item) => `<option value="${item}">${item}</option>`).join("") : "";
    }

    async function loadRuns() {
      const response = await fetch("/api/runs");
      const data = await response.json();
      const root = $("runs");
      if (!data.runs.length) {
        root.innerHTML = `<div class="hint">还没有 runs 结果。</div>`;
        return;
      }
      root.innerHTML = data.runs.map((run) => {
        const images = run.images.map((file) => `
          <div class="shot">
            <img src="${file.url}" alt="${file.name}">
            <a href="${file.url}" target="_blank">${file.name}</a>
          </div>
        `).join("");
        const files = run.files.map((file) => `<a class="file" href="${file.url}" target="_blank">${file.name}</a>`).join("");
        const csv = run.results_csv ? `<h3>${run.results_csv.name} 最近记录</h3>${csvTable(run.results_csv.preview)}` : "";
        return `
          <div class="run">
            <div class="run-title">
              <strong>${run.name}</strong>
              <span class="hint">${run.updated} · ${run.file_count} files</span>
            </div>
            <div class="files">${files}</div>
            ${csv}
            <div class="gallery">${images}</div>
          </div>
        `;
      }).join("");
    }

    function modelRows(models, actionLabel, action) {
      if (!models.length) return `<div class="hint">暂无模型。</div>`;
      return models.map((model) => `
        <div class="model-row">
          <strong>${model.name}</strong>
          <div class="meta">model=${model.base_model || "?"} · preset=${model.preset || "?"} · datasets=${model.datasets || "?"} · ${model.created_at || "?"}</div>
          <button class="secondary" onclick="${action}('${model.value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}')">${actionLabel}</button>
        </div>
      `).join("");
    }

    async function loadModels() {
      const response = await fetch("/api/models");
      const data = await response.json();
      $("activeModels").innerHTML = modelRows(data.active, "移入停用归档区", "archiveModel");
      $("disabledModels").innerHTML = modelRows(data.disabled, "恢复为有效模型", "restoreModel");
    }

    async function archiveModel(path) {
      await postJson("/api/models/archive", {weights: path});
      await loadOptions();
      await loadModels();
    }

    async function restoreModel(path) {
      await postJson("/api/models/restore", {weights: path});
      await loadOptions();
      await loadModels();
    }

    window.archiveModel = archiveModel;
    window.restoreModel = restoreModel;

    async function refreshStatus() {
      const response = await fetch("/api/status");
      const data = await response.json();
      $("statusRunning").textContent = data.running ? "运行中" : "空闲";
      $("statusKind").textContent = data.kind || "-";
      $("statusStarted").textContent = data.started_at || "-";
      $("statusCode").textContent = data.returncode === null ? "-" : data.returncode;
      $("logs").textContent = data.logs.join("");
      $("startTrain").disabled = data.running;
      $("startValidate").disabled = data.running;
      $("stopTask").disabled = !data.running;
    }

    loadOptions();
    loadModels();
    loadRuns();
    refreshStatus();
    setInterval(refreshStatus, 1000);
    setInterval(loadRuns, 5000);
  </script>
</body>
</html>
"""


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(line: str) -> None:
    with STATE_LOCK:
        STATE["logs"].append(line)
        if len(STATE["logs"]) > MAX_LOG_LINES:
            STATE["logs"] = STATE["logs"][-MAX_LOG_LINES:]


def dataset_last_updated(dataset_dir: Path) -> str:
    newest = max((path.stat().st_mtime for path in dataset_dir.rglob("*") if path.is_file()), default=0)
    if newest <= 0:
        return "unknown"
    return datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")


def dataset_options() -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for config_path in sorted(train.DATASETS_DIR.glob("*/data.yaml")):
        name = config_path.parent.name
        options.append({
            "label": f"{name} | updated {dataset_last_updated(config_path.parent)}",
            "value": name,
        })
        golden_dir = config_path.parent / train.GOLDEN_SPLIT / "images"
        if golden_dir.exists():
            options.append({
                "label": f"{name}/{train.GOLDEN_SPLIT} | updated {dataset_last_updated(golden_dir.parent)}",
                "value": f"{name}@{train.GOLDEN_SPLIT}",
            })
    return options


def training_preset_options() -> list[dict[str, str]]:
    return [
        {
            "label": f"{name} | {preset['description']}",
            "value": name,
        }
        for name, preset in train.AUGMENT_PRESETS.items()
    ]


def model_family_options() -> list[dict[str, object]]:
    return [
        {
            "label": f"{family} | {config['description']}",
            "value": family,
            "sizes": list(config["sizes"]),
        }
        for family, config in train.MODEL_FAMILIES.items()
    ]


def archive_weight_options() -> list[dict[str, str]]:
    metadata_by_weight: dict[str, dict] = {}
    for metadata_path in train.ARCHIVE_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata_by_weight[Path(metadata.get("weights", "")).name] = metadata

    options: list[dict[str, str]] = []
    for weight_path in sorted(train.ARCHIVE_DIR.glob("*.pt")):
        metadata = metadata_by_weight.get(weight_path.name, {})
        base_model = metadata.get("base_model", "?")
        preset = metadata.get("augment_preset", "?")
        train_sources = ",".join(metadata.get("train_dataset_sources") or metadata.get("dataset_sources") or [])
        valid_sources = ",".join(metadata.get("val_dataset_sources") or [])
        created_at = metadata.get("created_at", "?")
        label = f"archieve | {weight_path.name} | model={base_model} | preset={preset} | train={train_sources} | valid={valid_sources} | {created_at}"
        options.append({"label": label, "value": str(weight_path)})

    weights_dir = train.ROOT / "runs" / "detect" / "train" / "weights"
    for name in ["last.pt", "best.pt"]:
        path = weights_dir / name
        if path.exists():
            options.append({"label": f"lastrun | {name}", "value": str(path)})
    return options


def metadata_for_weight(weight_path: Path, root: Path) -> list[Path]:
    matches: list[Path] = []
    for metadata_path in sorted(root.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if Path(metadata.get("weights", "")).name == weight_path.name:
            matches.append(metadata_path)
    return matches


def metadata_by_weight(root: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for metadata_path in root.glob("*.json"):
        try:
            item = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata[Path(item.get("weights", "")).name] = item
    return metadata


def model_records(root: Path) -> list[dict[str, str]]:
    metadata = metadata_by_weight(root)
    records: list[dict[str, str]] = []
    for weight_path in sorted(root.glob("*.pt")):
        item = metadata.get(weight_path.name, {})
        datasets = ",".join(item.get("dataset_sources") or [item.get("dataset_name", "")])
        records.append({
            "name": weight_path.name,
            "value": str(weight_path.resolve()),
            "base_model": str(item.get("base_model", "")),
            "preset": str(item.get("augment_preset", "")),
            "datasets": datasets,
            "created_at": str(item.get("created_at", "")),
        })
    return records


def resolve_model_path(raw_path: str, root: Path) -> Path:
    base = root.resolve()
    target = Path(raw_path).expanduser()
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    if not target.is_file() or base not in target.parents or target.suffix != ".pt":
        raise FileNotFoundError(f"模型不存在或不在允许目录中: {raw_path}")
    return target


def move_model_files(weight_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir = weight_path.parent
    related = [weight_path] + metadata_for_weight(weight_path, source_dir)
    for source in related:
        target = target_dir / source.name
        if target.exists():
            raise FileExistsError(f"目标已存在: {target}")
    for source in related:
        shutil.move(str(source), str(target_dir / source.name))


def run_file_url(path: Path) -> str:
    relative = path.relative_to(train.ROOT / "runs").as_posix()
    return f"/run-file/{relative}"


def csv_preview(path: Path, rows: int = 6) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    selected = [lines[0]] + lines[-rows:]
    return [[cell.strip() for cell in line.split(",")] for line in selected]


def run_summaries() -> list[dict]:
    runs_root = train.ROOT / "runs"
    if not runs_root.exists():
        return []

    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    important_names = {
        "args.yaml",
        "results.csv",
        "results.png",
        "comparison_curves.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    }
    summaries: list[dict] = []
    for run_dir in sorted((path for path in runs_root.glob("*/*") if path.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        files = [path for path in run_dir.rglob("*") if path.is_file()]
        if not files:
            continue
        updated = max(path.stat().st_mtime for path in files)
        display_files = [
            path for path in files
            if path.name in important_names or path.suffix.lower() in {".pt", ".yaml", ".csv"}
        ]
        image_files = [
            path for path in files
            if path.suffix.lower() in image_suffixes and path.name != "results.png"
        ]
        preferred_images = [
            path for path in files
            if path.name in {
                "results.png",
                "comparison_curves.png",
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
                "BoxPR_curve.png",
                "BoxF1_curve.png",
                "BoxP_curve.png",
                "BoxR_curve.png",
            }
        ]
        watch_images = [
            path for path in image_files
            if "watch" in path.relative_to(run_dir).parts
        ]
        batch_images = [
            path for path in image_files
            if path.name.startswith("val_batch")
        ]
        other_images = sorted(
            [path for path in image_files if path not in watch_images and path not in batch_images],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:8]
        images = list(dict.fromkeys(preferred_images + sorted(watch_images)[:80] + sorted(batch_images)[:12] + other_images))
        results_csv = run_dir / "results.csv"
        summaries.append({
            "name": run_dir.relative_to(runs_root).as_posix(),
            "updated": datetime.fromtimestamp(updated).strftime("%Y-%m-%d %H:%M:%S"),
            "file_count": len(files),
            "files": [
                {"name": path.relative_to(run_dir).as_posix(), "url": run_file_url(path)}
                for path in sorted(display_files, key=lambda p: p.name)
            ],
            "images": [
                {"name": path.relative_to(run_dir).as_posix(), "url": run_file_url(path)}
                for path in images
            ],
            "results_csv": {
                "name": "results.csv",
                "url": run_file_url(results_csv),
                "preview": csv_preview(results_csv),
            } if results_csv.exists() else None,
        })
    return summaries


def start_process(kind: str, args: list[str], env_updates: dict[str, str]) -> tuple[bool, str | None]:
    global CURRENT_PROCESS
    with STATE_LOCK:
        if STATE["running"]:
            return False, "已有任务正在运行"
        STATE.update({
            "running": True,
            "kind": kind,
            "started_at": now_text(),
            "finished_at": None,
            "returncode": None,
            "logs": [],
        })

    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONUNBUFFERED"] = "1"

    def worker() -> None:
        append_log(f"$ {' '.join(args)}\n")
        process = subprocess.Popen(
            args,
            cwd=train.ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        with CURRENT_PROCESS_LOCK:
            CURRENT_PROCESS = process
        assert process.stdout is not None
        for line in process.stdout:
            append_log(line)
        returncode = process.wait()
        append_log(f"\n[{now_text()}] 任务结束，退出码 {returncode}\n")
        with CURRENT_PROCESS_LOCK:
            if CURRENT_PROCESS is process:
                CURRENT_PROCESS = None
        with STATE_LOCK:
            STATE["running"] = False
            STATE["finished_at"] = now_text()
            STATE["returncode"] = returncode

    threading.Thread(target=worker, daemon=True).start()
    return True, None


def stop_current_process() -> tuple[bool, str]:
    with CURRENT_PROCESS_LOCK:
        process = CURRENT_PROCESS
    if process is None or process.poll() is not None:
        with STATE_LOCK:
            STATE["running"] = False
        return False, "当前没有运行中的任务"

    append_log(f"\n[{now_text()}] 收到急停请求，正在终止任务...\n")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True, "任务已经结束"

    for _ in range(20):
        if process.poll() is not None:
            return True, "任务已终止"
        time.sleep(0.2)

    append_log(f"[{now_text()}] 任务未响应 SIGTERM，发送 SIGKILL...\n")
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True, "已发送强制终止"


@APP.get("/")
def index():
    return render_template_string(PAGE)


@APP.get("/api/options")
def api_options():
    return jsonify({
        "model_families": model_family_options(),
        "training_presets": training_preset_options(),
        "sample_choices": train.sample_choice_options(),
        "datasets": dataset_options(),
        "weights": archive_weight_options(),
    })


@APP.get("/api/status")
def api_status():
    with STATE_LOCK:
        return jsonify(dict(STATE))


@APP.post("/api/stop")
def api_stop():
    ok, message = stop_current_process()
    return jsonify({"ok": ok, "message": message})


@APP.get("/api/runs")
def api_runs():
    return jsonify({"runs": run_summaries()})


@APP.get("/api/models")
def api_models():
    return jsonify({
        "active": model_records(train.ARCHIVE_DIR),
        "disabled": model_records(train.DISABLED_ARCHIVE_DIR),
    })


@APP.post("/api/models/archive")
def api_archive_model():
    payload = request.get_json(force=True)
    try:
        weight_path = resolve_model_path(payload.get("weights", ""), train.ARCHIVE_DIR)
        move_model_files(weight_path, train.DISABLED_ARCHIVE_DIR)
    except (FileNotFoundError, FileExistsError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"ok": True})


@APP.post("/api/models/restore")
def api_restore_model():
    payload = request.get_json(force=True)
    try:
        weight_path = resolve_model_path(payload.get("weights", ""), train.DISABLED_ARCHIVE_DIR)
        move_model_files(weight_path, train.ARCHIVE_DIR)
    except (FileNotFoundError, FileExistsError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"ok": True})


@APP.get("/run-file/<path:relative_path>")
def run_file(relative_path: str):
    runs_root = (train.ROOT / "runs").resolve()
    target = (runs_root / relative_path).resolve()
    if not target.is_file() or runs_root not in target.parents:
        abort(404)
    return send_file(target)


@APP.post("/api/train")
def api_train():
    payload = request.get_json(force=True)
    if payload.get("model_family"):
        try:
            base_model = train.resolve_base_model(payload.get("model_family"), payload.get("model_size"))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
    else:
        base_model = train.resolve_base_model(payload.get("base_model") or train.BASE_MODEL)
    augment_preset = payload.get("augment_preset") or train.DEFAULT_AUGMENT_PRESET
    train_sources = payload.get("train_sources") or []
    valid_sources = payload.get("valid_sources") or []
    sample_multipliers = payload.get("train_sample_multipliers") or {}
    if not train_sources or not valid_sources:
        return jsonify({"error": "训练集和验证集都至少选择一个来源"}), 400
    try:
        train.resolve_augment_preset(augment_preset)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    code = (
        "import os, train\n"
        "train.run_training("
        "os.environ.get('YOLO_BASE_MODEL', train.BASE_MODEL), "
        "os.environ.get(train.TRAIN_PRESET_ENV, train.DEFAULT_AUGMENT_PRESET)"
        ")\n"
    )
    ok, error = start_process(
        "train",
        [sys.executable, "-c", code],
        {
            "YOLO_BASE_MODEL": base_model,
            train.TRAIN_PRESET_ENV: augment_preset,
            train.TRAIN_DATASET_ENV: os.pathsep.join(train_sources),
            train.VALID_DATASET_ENV: os.pathsep.join(valid_sources),
            train.TRAIN_SAMPLE_ENV: json.dumps(sample_multipliers),
        },
    )
    if not ok:
        return jsonify({"error": error}), 409
    return jsonify({"ok": True})


@APP.post("/api/validate")
def api_validate():
    payload = request.get_json(force=True)
    weights = payload.get("weights") or []
    test_sources = payload.get("test_sources") or []
    if isinstance(weights, str):
        weights = [weights]
    if not weights:
        return jsonify({"error": "至少选择一个权重"}), 400
    if not test_sources:
        return jsonify({"error": "测试集至少选择一个来源"}), 400
    code = (
        "import os, train\n"
        "from pathlib import Path\n"
        "weights = [Path(item) for item in os.environ['YOLO_VALIDATE_WEIGHTS'].split(os.pathsep) if item]\n"
        "print('Test weights:')\n"
        "for item in weights:\n"
        "    print(f'- {item}')\n"
        "train.run_batch_testing(weights)\n"
    )
    ok, error = start_process(
        "batch_test",
        [sys.executable, "-c", code],
        {
            "YOLO_VALIDATE_WEIGHTS": os.pathsep.join(weights),
            train.TEST_DATASET_ENV: os.pathsep.join(test_sources),
        },
    )
    if not ok:
        return jsonify({"error": error}), 409
    return jsonify({"ok": True})


def main() -> None:
    APP.run(host="0.0.0.0", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()
