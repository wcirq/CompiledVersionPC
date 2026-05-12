BUILTIN_MODELS = [
    {
        "name": "fire_smoke",
        "backend": "ultralytics",
        "task_type": "detect",
        "weight_path": "assets/weights/fire-smoke_model.pt",
        "class_names": ["fire", "smoke"],
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "imgsz": 640,
        "max_det": 100,
        "enabled": True,
        "description": "火焰与烟雾目标检测器"
    }
]
