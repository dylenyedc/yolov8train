import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file

import train


APP = Flask(__name__)
BASE_MODEL_CHOICES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
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
    button { border: 1px solid #243b53; border-radius: 6px; padding: 8px 12px; color: #fff; background: #243b53; cursor: pointer; }
    button.secondary { color: #243b53; background: #fff; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .choices { max-height: 210px; overflow: auto; border: 1px solid #e1e7ef; border-radius: 6px; padding: 8px; background: #fbfcfd; }
    .hint { color: #627386; font-size: 13px; }
    .runs { display: grid; gap: 12px; }
    .run { border: 1px solid #e1e7ef; border-radius: 8px; padding: 10px; background: #fbfcfd; }
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
        <label>基础模型</label>
        <select id="baseModel"></select>
        <h3>训练集来源</h3>
        <div id="trainSources" class="choices"></div>
        <h3>验证集来源</h3>
        <div id="validSources" class="choices"></div>
        <p class="hint">若存在 <code>golden/images</code>，会显示 <code>dataset/golden</code> 选项。</p>
        <button id="startTrain">开始训练</button>
      </section>

      <section style="margin-top:16px">
        <h2>测试集验证</h2>
        <label>权重来源</label>
        <select id="weightSelect"></select>
        <h3>测试集来源</h3>
        <div id="testSources" class="choices"></div>
        <div class="row" style="margin-top:12px">
          <button id="startValidate">开始验证</button>
          <button id="refreshOptions" class="secondary">刷新选项</button>
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

    function renderChecks(containerId, options) {
      const root = $(containerId);
      root.innerHTML = "";
      options.forEach((option) => {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = option.value;
        label.appendChild(input);
        label.append(" " + option.label);
        root.appendChild(label);
      });
    }

    async function loadOptions() {
      const response = await fetch("/api/options");
      const data = await response.json();
      $("baseModel").innerHTML = data.base_models.map((item) => `<option value="${item}">${item}</option>`).join("");
      renderChecks("trainSources", data.datasets);
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
      base_model: $("baseModel").value,
      train_sources: checkedValues("trainSources"),
      valid_sources: checkedValues("validSources"),
    });

    $("startValidate").onclick = () => postJson("/api/validate", {
      weights: $("weightSelect").value,
      test_sources: checkedValues("testSources"),
    });

    $("refreshOptions").onclick = loadOptions;
    $("refreshRuns").onclick = loadRuns;
    $("clearLogs").onclick = () => { $("logs").textContent = ""; };

    function csvTable(rows) {
      if (!rows || rows.length === 0) return "";
      const head = rows[0].map((cell) => `<th>${cell}</th>`).join("");
      const body = rows.slice(1).map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
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
    }

    loadOptions();
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
        train_sources = ",".join(metadata.get("train_dataset_sources") or metadata.get("dataset_sources") or [])
        valid_sources = ",".join(metadata.get("val_dataset_sources") or [])
        created_at = metadata.get("created_at", "?")
        label = f"archieve | {weight_path.name} | model={base_model} | train={train_sources} | valid={valid_sources} | {created_at}"
        options.append({"label": label, "value": str(weight_path)})

    weights_dir = train.ROOT / "runs" / "detect" / "train" / "weights"
    for name in ["last.pt", "best.pt"]:
        path = weights_dir / name
        if path.exists():
            options.append({"label": f"lastrun | {name}", "value": str(path)})
    return options


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
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
                "BoxPR_curve.png",
                "BoxF1_curve.png",
                "BoxP_curve.png",
                "BoxR_curve.png",
            }
        ]
        other_images = sorted(image_files, key=lambda p: p.stat().st_mtime, reverse=True)[:8]
        images = list(dict.fromkeys(preferred_images + other_images))
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
        )
        assert process.stdout is not None
        for line in process.stdout:
            append_log(line)
        returncode = process.wait()
        append_log(f"\n[{now_text()}] 任务结束，退出码 {returncode}\n")
        with STATE_LOCK:
            STATE["running"] = False
            STATE["finished_at"] = now_text()
            STATE["returncode"] = returncode

    threading.Thread(target=worker, daemon=True).start()
    return True, None


@APP.get("/")
def index():
    return render_template_string(PAGE)


@APP.get("/api/options")
def api_options():
    return jsonify({
        "base_models": BASE_MODEL_CHOICES,
        "datasets": dataset_options(),
        "weights": archive_weight_options(),
    })


@APP.get("/api/status")
def api_status():
    with STATE_LOCK:
        return jsonify(dict(STATE))


@APP.get("/api/runs")
def api_runs():
    return jsonify({"runs": run_summaries()})


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
    base_model = payload.get("base_model") or train.BASE_MODEL
    train_sources = payload.get("train_sources") or []
    valid_sources = payload.get("valid_sources") or []
    if not train_sources or not valid_sources:
        return jsonify({"error": "训练集和验证集都至少选择一个来源"}), 400
    code = "import os, train; train.run_training(os.environ.get('YOLO_BASE_MODEL', train.BASE_MODEL))"
    ok, error = start_process(
        "train",
        [sys.executable, "-c", code],
        {
            "YOLO_BASE_MODEL": base_model,
            train.TRAIN_DATASET_ENV: os.pathsep.join(train_sources),
            train.VALID_DATASET_ENV: os.pathsep.join(valid_sources),
        },
    )
    if not ok:
        return jsonify({"error": error}), 409
    return jsonify({"ok": True})


@APP.post("/api/validate")
def api_validate():
    payload = request.get_json(force=True)
    weights = payload.get("weights")
    test_sources = payload.get("test_sources") or []
    if not weights:
        return jsonify({"error": "请选择权重"}), 400
    if not test_sources:
        return jsonify({"error": "测试集至少选择一个来源"}), 400
    code = (
        "import os, train\n"
        "from ultralytics import YOLO\n"
        "cfg, name, _ = train.testing_data_config()\n"
        "weights = os.environ['YOLO_VALIDATE_WEIGHTS']\n"
        "print(f'Validate weights: {weights}')\n"
        "print(f'Test config: {cfg}')\n"
        "print(f'Test name: {name}')\n"
        "YOLO(weights).val(data=str(cfg), split='test', device=0, "
        "project=str(train.ROOT / 'runs' / 'detect'), name='test', exist_ok=True)\n"
    )
    ok, error = start_process(
        "validate",
        [sys.executable, "-c", code],
        {
            "YOLO_VALIDATE_WEIGHTS": weights,
            train.TEST_DATASET_ENV: os.pathsep.join(test_sources),
        },
    )
    if not ok:
        return jsonify({"error": error}), 409
    return jsonify({"ok": True})


def main() -> None:
    APP.run(host="127.0.0.1", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()
