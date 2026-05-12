from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m store_web",
        description="启动列车顶部异物检测管理后台与 HTTP 服务。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务监听地址。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=55555,
        help="服务监听端口。",
    )
    parser.add_argument(
        "--root-dir",
        default="./store_data",
        help="模型仓库根目录，用于保存 models、registry.json、tmp 等数据。",
    )
    parser.add_argument(
        "--yolo-weight-path",
        default=None,
        help="YOLO 列车顶部检测权重路径；不传则使用打包内置权重。",
    )
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        default=0.8,
        help="YOLO 列车顶部检测的置信度阈值。",
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="YOLO 推理设备，例如 cpu、cuda:0；不传则按模型默认逻辑选择。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from store_core.platform import TrainRoofAnomalyStore
    from store_service.server import run_foreground_server

    store = TrainRoofAnomalyStore(
        root_dir=args.root_dir,
        autostart_service=False,
        yolo_weight_path=args.yolo_weight_path,
        yolo_conf_threshold=args.yolo_conf_threshold,
        yolo_device=args.yolo_device,
    )
    run_foreground_server(store.manager, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
