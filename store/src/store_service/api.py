from __future__ import annotations

import json
import multiprocessing
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from store_core.model_store import ModelStoreManager
from store_core.schemas import RuntimeOptions
from store_infer import get_inference_registry


_EXPORT_TASKS: Dict[str, Dict[str, Any]] = {}
_EXPORT_TASKS_LOCK = threading.Lock()
_TRAIN_TASKS: Dict[str, Dict[str, Any]] = {}
_TRAIN_TASKS_LOCK = threading.Lock()


class TrainRequest(BaseModel):
    model_name: str
    image_dir: str
    save_root_dir: Optional[str] = None
    calibrate_dir: Optional[str] = None
    runtime_options: Dict[str, Any] = Field(default_factory=dict)


class DetectRequest(BaseModel):
    model_id: str
    image_path: Optional[str] = None
    include_heatmap_base64: bool = False
    threshold: Optional[float] = None
    heatmap_include_background: bool = True
    heatmap_zero_below_threshold: Optional[bool] = None


class SampleUpdateRequest(BaseModel):
    contour: List[List[int]]
    note: str = ""
    enabled_tile_ids: List[str] = Field(default_factory=list)


class SampleAddRequest(BaseModel):
    model_id: str
    image_path: Optional[str] = None
    contour: Optional[List[List[int]]] = None
    note: str = ""


class ScanRequest(BaseModel):
    threshold: Optional[float] = None


class TileStateRequest(BaseModel):
    enabled_tile_ids: List[str] = Field(default_factory=list)


class ModelThresholdUpdateRequest(BaseModel):
    threshold: float


class InferenceRequest(BaseModel):
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    imgsz: Optional[int] = None
    max_det: Optional[int] = None
    device: Optional[str] = None
    include_visualization_base64: bool = True


def _parse_runtime_options(payload: Dict[str, Any]) -> RuntimeOptions:
    return RuntimeOptions(**payload)


def _safe_relative_upload_path(filename: str, fallback_prefix: str, index: int) -> Path:
    raw_name = (filename or "").replace("\\", "/").strip("/")
    if not raw_name:
        return Path(fallback_prefix) / f"file_{index:05d}"
    safe_parts = [part for part in Path(raw_name).parts if part not in ("", ".", "..")]
    if not safe_parts:
        return Path(fallback_prefix) / f"file_{index:05d}"
    return Path(*safe_parts)


def _estimate_train_progress(stage: str, event: Dict[str, Any], current_progress: int) -> int:
    stage_progress = {
        "prepare_training": 3,
        "build_tiles_start": 38,
        "build_memory_done": 68,
        "calibrate_start": 74,
        "calibrate_done": 92,
        "train_done": 100,
    }
    if stage in stage_progress:
        return stage_progress[stage]
    if stage in {"preprocess_ok", "preprocess_failed"}:
        total = max(1, int(event.get("total", 1)))
        index = max(0, min(total, int(event.get("index", 0))))
        return max(current_progress, 4 + int(index * 30 / total))
    if stage in {"calibrate_segment_ok", "calibrate_segment_failed"}:
        total = max(1, int(event.get("total", 1)))
        index = max(0, min(total, int(event.get("index", 0))))
        return max(current_progress, 75 + int(index * 12 / total))
    return current_progress


async def _write_uploaded_files(
    files: List[UploadFile],
    target_dir: Path,
    fallback_prefix: str,
) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, upload in enumerate(files, start=1):
        if upload is None:
            continue
        relative_path = _safe_relative_upload_path(upload.filename or "", fallback_prefix=fallback_prefix, index=index)
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = await upload.read()
        if not data:
            continue
        destination.write_bytes(data)
        written += 1
    return written


def _append_train_log(task_id: str, line: str) -> None:
    with _TRAIN_TASKS_LOCK:
        task = _TRAIN_TASKS.get(task_id)
        if task is None:
            return
        task["logs"].append(line)
        if len(task["logs"]) > 500:
            task["logs"] = task["logs"][-500:]


def _train_worker_entry(
    manager_config: Dict[str, Any],
    model_name: str,
    image_dir: str,
    runtime_options_payload: Dict[str, Any],
    calibrate_dir: Optional[str],
    event_queue: Any,
) -> None:
    manager = ModelStoreManager(
        root_dir=manager_config["root_dir"],
        yolo_weight_path=manager_config.get("yolo_weight_path"),
        yolo_conf_threshold=manager_config.get("yolo_conf_threshold", 0.25),
        yolo_device=manager_config.get("yolo_device"),
    )
    runtime_options = RuntimeOptions(**runtime_options_payload)

    def on_progress(event: Dict[str, Any]) -> None:
        event_queue.put({"type": "progress", "event": event})

    try:
        result = manager.train_model(
            model_name=model_name,
            image_dir=image_dir,
            runtime_options=runtime_options,
            calibrate_dir=calibrate_dir,
            progress_callback=on_progress,
        )
        event_queue.put({"type": "completed", "result": result})
    except Exception as exc:
        event_queue.put({"type": "error", "error": str(exc)})


def build_app(manager) -> FastAPI:
    app = FastAPI(title="Train Roof Anomaly Store")
    infer_registry = get_inference_registry()
    static_dir = Path(__file__).resolve().parents[1] / "store_web" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def remove_model_artifacts(model_id: Optional[str]) -> None:
        if not model_id:
            return
        model_dir = Path(manager.models_dir) / model_id
        shutil.rmtree(model_dir, ignore_errors=True)
        registry = manager._read_registry()
        next_models = [item for item in registry.get("models", []) if item.get("model_id") != model_id]
        if len(next_models) != len(registry.get("models", [])):
            registry["models"] = next_models
            manager._save_registry(registry)

    def finalize_stopped_task(task_id: str, reason: str = "训练已停止") -> None:
        with _TRAIN_TASKS_LOCK:
            task = _TRAIN_TASKS.get(task_id)
            if task is None:
                return
            upload_root = task.get("upload_root")
            model_id = task.get("model_id")
            task["status"] = "stopped"
            task["progress"] = 0
            task["message"] = reason
            task["error"] = None
            task["logs"].append(json.dumps({"stage": "stopped", "detail": reason}, ensure_ascii=False))
            task["process"] = None
            task["queue"] = None
        if upload_root:
            shutil.rmtree(upload_root, ignore_errors=True)
        remove_model_artifacts(model_id)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/runtime-options/defaults")
    def get_runtime_options_defaults() -> Dict[str, Any]:
        return {"runtime_options": asdict(RuntimeOptions())}

    @app.get("/api/models")
    def list_models() -> Dict[str, Any]:
        return {"items": manager.list_models()}

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str) -> Dict[str, Any]:
        try:
            return manager.get_model(model_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/models/{model_id}/export-summary")
    def get_export_summary(model_id: str) -> Dict[str, Any]:
        try:
            return manager.get_export_package_summary(model_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/models/{model_id}/export")
    def create_export_model_task(model_id: str, deployment_only: bool = Query(False)) -> Dict[str, Any]:
        try:
            manager.get_model(model_id)
            task_id = f"export_{uuid.uuid4().hex[:16]}"
            filename = f"{model_id}-deploy.zip" if deployment_only else f"{model_id}.zip"
            with _EXPORT_TASKS_LOCK:
                _EXPORT_TASKS[task_id] = {
                    "task_id": task_id,
                    "model_id": model_id,
                    "deployment_only": bool(deployment_only),
                    "status": "pending",
                    "progress": 0,
                    "message": "等待开始压缩",
                    "archive_path": None,
                    "filename": filename,
                    "error": None,
                }

            def run_export() -> None:
                try:
                    with _EXPORT_TASKS_LOCK:
                        _EXPORT_TASKS[task_id]["status"] = "running"
                        _EXPORT_TASKS[task_id]["message"] = "正在压缩部署关键文件..." if deployment_only else "正在压缩模型目录..."

                    def on_progress(current: int, total: int, name: str) -> None:
                        percent = int(current * 100 / max(1, total))
                        with _EXPORT_TASKS_LOCK:
                            task = _EXPORT_TASKS.get(task_id)
                            if task is None:
                                return
                            task["progress"] = percent
                            task["message"] = f"正在压缩：{current}/{total} {name}"

                    archive_path = manager.export_model_archive(
                        model_id=model_id,
                        deployment_only=deployment_only,
                        progress_callback=on_progress,
                    )
                    with _EXPORT_TASKS_LOCK:
                        task = _EXPORT_TASKS[task_id]
                        task["status"] = "ready"
                        task["progress"] = 100
                        task["message"] = "压缩完成，准备下载"
                        task["archive_path"] = archive_path
                except Exception as exc:
                    with _EXPORT_TASKS_LOCK:
                        task = _EXPORT_TASKS.get(task_id)
                        if task is not None:
                            task["status"] = "error"
                            task["error"] = str(exc)
                            task["message"] = f"导出失败：{exc}"

            threading.Thread(target=run_export, name=f"store-export-{task_id}", daemon=True).start()
            return {"task_id": task_id, "model_id": model_id, "deployment_only": bool(deployment_only)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/model-export-tasks/{task_id}")
    def get_export_model_task(task_id: str) -> Dict[str, Any]:
        with _EXPORT_TASKS_LOCK:
            task = _EXPORT_TASKS.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"Unknown export task: {task_id}")
            return {
                "task_id": task["task_id"],
                "model_id": task["model_id"],
                "deployment_only": task.get("deployment_only", False),
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"],
                "filename": task["filename"],
                "error": task["error"],
            }

    @app.get("/api/model-export-tasks/{task_id}/download")
    def download_export_model_task(task_id: str) -> FileResponse:
        with _EXPORT_TASKS_LOCK:
            task = _EXPORT_TASKS.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"Unknown export task: {task_id}")
            if task["status"] != "ready" or not task["archive_path"]:
                raise HTTPException(status_code=400, detail="Export task is not ready for download.")
            archive_path = task["archive_path"]
            filename = task["filename"]
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=filename,
        )

    @app.post("/api/models/import")
    async def import_model(model_file: UploadFile = File(...)) -> Dict[str, Any]:
        try:
            suffix = Path(model_file.filename or "model.zip").suffix or ".zip"
            tmp_path = Path(manager.tmp_dir) / f"import_model_upload{suffix}"
            data = await model_file.read()
            tmp_path.write_bytes(data)
            try:
                return manager.import_model_archive(str(tmp_path))
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/models/{model_id}/threshold")
    def update_model_threshold(model_id: str, request: ModelThresholdUpdateRequest) -> Dict[str, Any]:
        try:
            return manager.update_model_threshold(model_id=model_id, threshold=request.threshold)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str) -> Dict[str, Any]:
        try:
            return manager.delete_model(model_id=model_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/models/{model_id}/prune-assets")
    def prune_model_assets(model_id: str) -> Dict[str, Any]:
        try:
            return manager.prune_model_assets(model_id=model_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/models/{model_id}/samples")
    def list_samples(model_id: str, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
        try:
            return manager.list_samples(model_id=model_id, page=page, page_size=page_size)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/models/{model_id}/samples/{sample_id}")
    def get_sample_detail(model_id: str, sample_id: str) -> Dict[str, Any]:
        try:
            return manager.get_sample_detail(model_id=model_id, sample_id=sample_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/models/{model_id}/samples/{sample_id}/image")
    def get_sample_image(model_id: str, sample_id: str, kind: str = Query("processed")) -> FileResponse:
        try:
            path = manager.get_sample_image_path(model_id=model_id, sample_id=sample_id, kind=kind)
            return FileResponse(path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/models/{model_id}/samples/{sample_id}/tiles/{tile_id}/image")
    def get_sample_tile_image(model_id: str, sample_id: str, tile_id: str) -> FileResponse:
        try:
            path = manager.get_sample_tile_image_path(model_id=model_id, sample_id=sample_id, tile_id=tile_id)
            return FileResponse(path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/train")
    def train_model(request: TrainRequest) -> Dict[str, Any]:
        progress_events: List[Dict[str, Any]] = []

        def collect(payload: Dict[str, Any]) -> None:
            progress_events.append(payload)
            print(json.dumps(payload, ensure_ascii=False))

        try:
            response = manager.train_model(
                model_name=request.model_name,
                image_dir=request.image_dir,
                save_root_dir=request.save_root_dir,
                calibrate_dir=request.calibrate_dir,
                runtime_options=_parse_runtime_options(request.runtime_options),
                progress_callback=collect,
            )
            response["progress_events"] = progress_events
            return response
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/train-tasks")
    async def create_train_task(
        model_name: str = Form(...),
        runtime_options_json: str = Form("{}"),
        train_files: List[UploadFile] = File(...),
        calibrate_files: Optional[List[UploadFile]] = File(None),
    ) -> Dict[str, Any]:
        try:
            runtime_options_payload = json.loads(runtime_options_json or "{}")
            runtime_options = _parse_runtime_options(runtime_options_payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"训练参数解析失败: {exc}") from exc

        if not train_files:
            raise HTTPException(status_code=400, detail="请至少选择一个训练文件。")

        task_id = f"train_{uuid.uuid4().hex[:16]}"
        upload_root = Path(manager.tmp_dir) / "train_tasks" / task_id
        train_dir = upload_root / "train"
        calibrate_dir = upload_root / "calibrate"
        try:
            train_count = await _write_uploaded_files(train_files, train_dir, fallback_prefix="train")
            calibrate_count = await _write_uploaded_files(calibrate_files or [], calibrate_dir, fallback_prefix="calibrate")
        except Exception as exc:
            shutil.rmtree(upload_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"上传训练文件失败: {exc}") from exc

        if train_count <= 0:
            shutil.rmtree(upload_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail="训练目录没有可用图片文件。")

        with _TRAIN_TASKS_LOCK:
            _TRAIN_TASKS[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0,
                "message": "训练文件上传完成，等待开始。",
                "model_name": model_name,
                "model_id": None,
                "error": None,
                "logs": [
                    f"[upload] 训练文件 {train_count} 个",
                    f"[upload] 校准文件 {calibrate_count} 个",
                ],
                "progress_events": [],
                "runtime_options": asdict(runtime_options),
                "train_file_count": train_count,
                "calibrate_file_count": calibrate_count,
                "upload_root": str(upload_root),
                "process": None,
                "queue": None,
            }

        ctx = multiprocessing.get_context("spawn")
        event_queue = ctx.Queue()
        process = ctx.Process(
            target=_train_worker_entry,
            name=f"store-train-{task_id}",
            args=(
                {
                    "root_dir": str(manager.root_dir),
                    "yolo_weight_path": manager.yolo_weight_path,
                    "yolo_conf_threshold": manager.yolo_conf_threshold,
                    "yolo_device": manager.yolo_device,
                },
                model_name,
                str(train_dir),
                asdict(runtime_options),
                str(calibrate_dir) if calibrate_count > 0 else None,
                event_queue,
            ),
            daemon=True,
        )
        process.start()
        with _TRAIN_TASKS_LOCK:
            task = _TRAIN_TASKS.get(task_id)
            if task is not None:
                task["status"] = "running"
                task["progress"] = 1
                task["message"] = "开始训练..."
                task["process"] = process
                task["queue"] = event_queue

        def monitor_train_task() -> None:
            while True:
                with _TRAIN_TASKS_LOCK:
                    task = _TRAIN_TASKS.get(task_id)
                    if task is None:
                        break
                    current_process = task.get("process")
                    current_queue = task.get("queue")
                    current_status = task.get("status")
                if current_status == "stopped":
                    break
                if current_queue is not None:
                    try:
                        message = current_queue.get(timeout=0.5)
                    except Empty:
                        message = None
                    if message:
                        if message.get("type") == "progress":
                            event = message.get("event", {})
                            stage = str(event.get("stage", "unknown"))
                            line = json.dumps(event, ensure_ascii=False)
                            with _TRAIN_TASKS_LOCK:
                                live_task = _TRAIN_TASKS.get(task_id)
                                if live_task is not None:
                                    live_task["progress_events"].append(event)
                                    if len(live_task["progress_events"]) > 500:
                                        live_task["progress_events"] = live_task["progress_events"][-500:]
                                    live_task["logs"].append(line)
                                    if len(live_task["logs"]) > 500:
                                        live_task["logs"] = live_task["logs"][-500:]
                                    live_task["progress"] = _estimate_train_progress(stage, event, int(live_task.get("progress", 0)))
                                    live_task["message"] = f"训练中: {stage}"
                                    if event.get("model_id"):
                                        live_task["model_id"] = event.get("model_id")
                        elif message.get("type") == "completed":
                            result = message.get("result", {})
                            with _TRAIN_TASKS_LOCK:
                                live_task = _TRAIN_TASKS.get(task_id)
                                if live_task is not None:
                                    live_task["status"] = "completed"
                                    live_task["progress"] = 100
                                    live_task["message"] = "训练完成"
                                    live_task["model_id"] = result.get("model_id")
                                    live_task["result"] = result
                                    live_task["logs"].append(json.dumps({"stage": "completed", "model_id": result.get("model_id")}, ensure_ascii=False))
                            shutil.rmtree(upload_root, ignore_errors=True)
                            break
                        elif message.get("type") == "error":
                            detail = message.get("error", "训练失败")
                            with _TRAIN_TASKS_LOCK:
                                live_task = _TRAIN_TASKS.get(task_id)
                                if live_task is not None:
                                    live_task["status"] = "error"
                                    live_task["error"] = detail
                                    live_task["message"] = f"训练失败: {detail}"
                                    live_task["logs"].append(json.dumps({"stage": "error", "detail": detail}, ensure_ascii=False))
                            shutil.rmtree(upload_root, ignore_errors=True)
                            remove_model_artifacts(task.get("model_id") if task else None)
                            break
                if current_process is not None and not current_process.is_alive():
                    current_process.join(timeout=0.1)
                    with _TRAIN_TASKS_LOCK:
                        live_task = _TRAIN_TASKS.get(task_id)
                        if live_task is None or live_task.get("status") in {"completed", "error", "stopped"}:
                            break
                        live_task["status"] = "error"
                        live_task["error"] = "训练进程异常退出"
                        live_task["message"] = "训练进程异常退出"
                        live_task["logs"].append(json.dumps({"stage": "error", "detail": "训练进程异常退出"}, ensure_ascii=False))
                    shutil.rmtree(upload_root, ignore_errors=True)
                    remove_model_artifacts(task.get("model_id") if task else None)
                    break

        threading.Thread(target=monitor_train_task, name=f"store-train-monitor-{task_id}", daemon=True).start()
        return {
            "task_id": task_id,
            "status": "running",
            "progress": 1,
            "message": "训练任务已创建并开始运行",
            "train_file_count": train_count,
            "calibrate_file_count": calibrate_count,
        }

    @app.get("/api/train-tasks/{task_id}")
    def get_train_task(task_id: str) -> Dict[str, Any]:
        with _TRAIN_TASKS_LOCK:
            task = _TRAIN_TASKS.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"Unknown train task: {task_id}")
            return {
                "task_id": task["task_id"],
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"],
                "model_name": task["model_name"],
                "model_id": task.get("model_id"),
                "error": task.get("error"),
                "logs": task.get("logs", []),
                "progress_events": task.get("progress_events", []),
                "runtime_options": task.get("runtime_options", {}),
                "train_file_count": task.get("train_file_count", 0),
                "calibrate_file_count": task.get("calibrate_file_count", 0),
            }

    @app.post("/api/train-tasks/{task_id}/stop")
    def stop_train_task(task_id: str) -> Dict[str, Any]:
        with _TRAIN_TASKS_LOCK:
            task = _TRAIN_TASKS.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"Unknown train task: {task_id}")
            status = task.get("status")
            process = task.get("process")
            if status in {"completed", "error", "stopped"}:
                raise HTTPException(status_code=400, detail=f"当前任务状态不支持停止: {status}")
            task["status"] = "stopping"
            task["message"] = "正在停止训练并清理文件..."
            task["logs"].append(json.dumps({"stage": "stopping", "detail": "收到停止请求"}, ensure_ascii=False))

        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

        finalize_stopped_task(task_id)
        return {"task_id": task_id, "status": "stopped", "message": "训练已停止，已清理临时文件和模型目录。"}

    @app.post("/api/detect")
    async def detect_model(
        model_id: str = Form(...),
        image_path: Optional[str] = Form(None),
        include_heatmap_base64: bool = Form(False),
        threshold: Optional[float] = Form(None),
        heatmap_include_background: bool = Form(True),
        heatmap_zero_below_threshold: Optional[bool] = Form(None),
        image_file: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        try:
            image_bytes = await image_file.read() if image_file is not None else None
            return manager.detect_image(
                model_id=model_id,
                image_path=image_path,
                image_bytes=image_bytes,
                include_heatmap_base64=include_heatmap_base64,
                threshold=threshold,
                heatmap_include_background=heatmap_include_background,
                heatmap_zero_below_threshold=heatmap_zero_below_threshold,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/models/{model_id}/scan-samples")
    def scan_samples(model_id: str, request: ScanRequest) -> Dict[str, Any]:
        try:
            return manager.scan_samples_for_anomalies(model_id=model_id, threshold=request.threshold)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/models/{model_id}/samples/{sample_id}")
    def delete_sample(model_id: str, sample_id: str) -> Dict[str, Any]:
        try:
            return manager.delete_sample(model_id=model_id, sample_id=sample_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/models/{model_id}/samples/{sample_id}")
    def update_sample(model_id: str, sample_id: str, request: SampleUpdateRequest) -> Dict[str, Any]:
        try:
            return manager.update_sample_contour(
                model_id=model_id,
                sample_id=sample_id,
                contour=request.contour,
                note=request.note,
                enabled_tile_ids=request.enabled_tile_ids,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/models/{model_id}/samples/{sample_id}/tiles")
    def update_sample_tiles(model_id: str, sample_id: str, request: TileStateRequest) -> Dict[str, Any]:
        try:
            return manager.update_sample_tiles_enabled(model_id=model_id, sample_id=sample_id, enabled_tile_ids=request.enabled_tile_ids)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/models/{model_id}/samples")
    async def add_sample(
        model_id: str,
        image_path: Optional[str] = Form(None),
        contour_json: Optional[str] = Form(None),
        note: str = Form(""),
        append_max_vectors: int = Form(20),
        image_file: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        contour = json.loads(contour_json) if contour_json else None
        try:
            image_bytes = await image_file.read() if image_file is not None else None
            return manager.add_positive_sample(
                model_id=model_id,
                image_path=image_path,
                image_bytes=image_bytes,
                contour=contour,
                note=note,
                append_max_vectors=append_max_vectors,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/extract-contours")
    async def extract_contours(
        image_path: Optional[str] = Form(None),
        image_file: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        try:
            image_bytes = await image_file.read() if image_file is not None else None
            return manager.extract_roof_contours(image_path=image_path, image_bytes=image_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/inference/models")
    def list_inference_models() -> Dict[str, Any]:
        return {"items": infer_registry.list_models()}

    @app.get("/api/inference/models/{model_name}")
    def get_inference_model(model_name: str) -> Dict[str, Any]:
        try:
            return infer_registry.get_model_info(model_name)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/inference/{model_name}")
    async def run_inference(
        model_name: str,
        image_path: Optional[str] = Form(None),
        conf_threshold: Optional[float] = Form(None),
        iou_threshold: Optional[float] = Form(None),
        imgsz: Optional[int] = Form(None),
        max_det: Optional[int] = Form(None),
        device: Optional[str] = Form(None),
        include_visualization_base64: bool = Form(True),
        image_file: Optional[UploadFile] = File(None),
    ) -> Dict[str, Any]:
        try:
            image_bytes = await image_file.read() if image_file is not None else None
            return infer_registry.run(
                model_name=model_name,
                image_path=image_path,
                image_bytes=image_bytes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                include_visualization_base64=include_visualization_base64,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
