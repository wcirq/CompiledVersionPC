BUILTIN_MODELS = [
    {
        "name": "fire_smoke",
        "backend": "auto",
        "task_type": "detect",
        "class_names": ["fire", "smoke"],
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "imgsz": 640,
        "max_det": 100,
        "enabled": True,
        "description": "火焰与烟雾目标检测器",
        "backends": {
            "sophon_bmcv": {
                "weight_path": "assets/weights/best_yolo11s_bm1684x_fp16.bmodel",
                "python_path": "sophon_demo",
                "dev_id": 0,
            },
            "ultralytics": {
                "weight_path": "assets/weights/fire-smoke_model.pt",
            },
        },
    }
]
