import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "testimage"
RESULT_DIR = ROOT / "testimage-result"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def clear_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for item in RESULT_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    clear_result_dir()

    images = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images found in {SOURCE_DIR}")
        return

    weights = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else Path("yolov8n.pt")
    model = YOLO(str(weights))
    model.predict(
        source=[str(path) for path in images],
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        project=str(RESULT_DIR),
        name=".",
        exist_ok=True,
    )
    print(f"Saved detection results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
