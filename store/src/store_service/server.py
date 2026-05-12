from __future__ import annotations

import socket
import threading
from typing import Any, Dict, Optional

import uvicorn

from .api import build_app


_SERVER_LOCK = threading.Lock()
_SERVER_STATE: Dict[str, Any] = {}


def _find_free_port(host: str, start_port: int) -> int:
    port = int(start_port)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, port)) != 0:
                return port
        port += 1


def ensure_background_server(manager, host: str = "127.0.0.1", start_port: int = 55555) -> Dict[str, Any]:
    with _SERVER_LOCK:
        if _SERVER_STATE.get("thread") and _SERVER_STATE["thread"].is_alive():
            return {
                "host": _SERVER_STATE["host"],
                "port": _SERVER_STATE["port"],
                "url": _SERVER_STATE["url"],
            }

        port = _find_free_port(host, start_port)
        app = build_app(manager)
        config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config=config)

        def run_server() -> None:
            server.run()

        thread = threading.Thread(target=run_server, name="train-roof-store-server", daemon=True)
        thread.start()

        _SERVER_STATE.clear()
        _SERVER_STATE.update(
            {
                "thread": thread,
                "server": server,
                "host": host,
                "port": port,
                "url": f"http://{host}:{port}",
            }
        )
        return {"host": host, "port": port, "url": f"http://{host}:{port}"}


def wait_for_background_server(poll_interval: float = 1.0) -> Dict[str, Any]:
    thread = _SERVER_STATE.get("thread")
    if thread is None or not thread.is_alive():
        raise RuntimeError("Background store service is not running. Start it first with autostart_service=True.")

    try:
        while thread.is_alive():
            thread.join(timeout=max(0.1, float(poll_interval)))
    except KeyboardInterrupt:
        pass

    return {
        "host": _SERVER_STATE.get("host"),
        "port": _SERVER_STATE.get("port"),
        "url": _SERVER_STATE.get("url"),
    }


def run_foreground_server(manager, host: str = "0.0.0.0", port: int = 55555) -> Dict[str, Any]:
    app = build_app(manager)
    uvicorn.run(app, host=host, port=int(port), log_level="info")
    return {"host": host, "port": int(port), "url": f"http://{host}:{int(port)}"}


def main() -> None:
    from store_web.__main__ import main as store_web_main

    store_web_main()
