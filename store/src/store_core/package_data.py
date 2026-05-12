from importlib.resources import files
from pathlib import Path


def get_default_yolo_weight_path() -> str:
    return str(Path(files("store_core")) / "assets" / "weights" / "train_roof_yolo11n_best.pt")

