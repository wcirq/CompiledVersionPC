from importlib.resources import files
from pathlib import Path


def get_default_yolo_weight_path() -> str:
    return str(Path(files("store_core")) / "assets" / "weights" / "train_roof_yolo11n_best.pt")


def get_default_bm_yolo_weight_path() -> str:
    return str(Path(files("store_core")) / "assets" / "weights" / "best_yolo11n_seg_bm1684x_fp16.bmodel")


def get_default_sophon_seg_demo_path() -> str:
    package_demo_dir = Path(files("store_core")) / "sophon_seg_demo"
    if package_demo_dir.exists():
        return str(package_demo_dir)
    repo_demo_dir = Path(__file__).resolve().parents[3] / "suanneng" / "YOLOv8_plus_seg" / "python"
    return str(repo_demo_dir)
