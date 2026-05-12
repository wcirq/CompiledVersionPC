from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from store_core.schemas import RuntimeOptions
from store_infer import get_inference_registry


_EXPORT_TASKS: Dict[str, Dict[str, Any]] = {}
_EXPORT_TASKS_LOCK = threading.Lock()


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
    max_det: Optional[int] = None
    device: Optional[str] = None
    include_visualization_base64: bool = True


def _parse_runtime_options(payload: Dict[str, Any]) -> RuntimeOptions:
    return RuntimeOptions(**payload)


def build_app(manager) -> FastAPI:
    app = FastAPI(title="Train Roof Anomaly Store")
    infer_registry = get_inference_registry()
    static_dir = Path(__file__).resolve().parents[1] / "store_web" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok"}

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
                max_det=max_det,
                device=device,
                include_visualization_base64=include_visualization_base64,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
