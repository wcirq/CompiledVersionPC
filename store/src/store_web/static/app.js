const state = {
  activeView: "store",
  selectedModelId: null,
  selectedModelThreshold: null,
  selectedModelSampleAssetsAvailable: false,
  selectedModelStorageMessage: "",
  selectedInferenceModelName: null,
  selectedInferenceModelConfig: null,
  page: 1,
  pageSize: 10,
  total: 0,
  appendEditor: null,
  updateEditor: null,
  appendFile: null,
  currentSampleDetail: null,
  sourcePreviewImage: null,
  detectFile: null,
  inferFile: null,
  appendPreviewUrl: "",
  imagePreview: {
    naturalWidth: 0,
    naturalHeight: 0,
    scale: 1,
    minScale: 1,
    maxScale: 8,
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
  },
};

function setActiveView(view) {
  state.activeView = view === "infer" ? "infer" : "store";
  document.getElementById("openStoreViewBtn").classList.toggle("active", state.activeView === "store");
  document.getElementById("openInferViewBtn").classList.toggle("active", state.activeView === "infer");
  document.getElementById("storeSidebarPanel").classList.toggle("hidden", state.activeView !== "store");
  document.getElementById("inferSidebarPanel").classList.toggle("hidden", state.activeView !== "infer");
  document.getElementById("storeWorkbench").classList.toggle("hidden", state.activeView !== "store");
  document.getElementById("inferWorkbench").classList.toggle("hidden", state.activeView !== "infer");
}

function formatThresholdLabel(baseLabel, threshold) {
  return Number.isFinite(threshold)
    ? `${baseLabel}（默认 ${Number(threshold).toFixed(4)}）`
    : `${baseLabel}（默认未设置）`;
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  if (value < 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function updateThresholdInputs(threshold) {
  const detectLabel = document.getElementById("detectThresholdLabel");
  const appendLabel = document.getElementById("appendThresholdLabel");
  const detectInput = document.getElementById("detectThreshold");
  const appendInput = document.getElementById("appendThreshold");
  detectLabel.textContent = formatThresholdLabel("Threshold", threshold);
  appendLabel.textContent = formatThresholdLabel("Threshold", threshold);
  detectInput.placeholder = Number.isFinite(threshold)
    ? `留空则使用默认值 ${Number(threshold).toFixed(4)}`
    : "留空则使用模型默认值";
  appendInput.placeholder = Number.isFinite(threshold)
    ? `留空则使用默认值 ${Number(threshold).toFixed(4)}`
    : "留空则使用模型默认值";
}

async function api(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function setJson(id, data) {
  document.getElementById(id).textContent = JSON.stringify(data, null, 2);
}

function setInferenceResultJson(data) {
  const payload = { ...data };
  if (typeof payload.visualization_base64 === "string" && payload.visualization_base64) {
    payload.visualization_base64 = `[base64 omitted, length=${payload.visualization_base64.length}]`;
  }
  document.getElementById("inferResultJson").textContent = JSON.stringify(payload, null, 2);
}

function openModal(id) {
  document.getElementById(id).classList.remove("hidden");
}

function closeModal(id) {
  document.getElementById(id).classList.add("hidden");
}

function openImagePreview(src, title) {
  if (!src) {
    return;
  }
  const image = document.getElementById("imagePreviewImage");
  const titleNode = document.getElementById("imagePreviewTitle");
  image.src = src;
  titleNode.textContent = title || "图片预览";
  openModal("imagePreviewModal");
}

function getImagePreviewElements() {
  return {
    body: document.getElementById("imagePreviewBody"),
    image: document.getElementById("imagePreviewImage"),
  };
}

function clampImagePreviewOffsets() {
  const { body } = getImagePreviewElements();
  const preview = state.imagePreview;
  const scaledWidth = preview.naturalWidth * preview.scale;
  const scaledHeight = preview.naturalHeight * preview.scale;
  const bodyWidth = body.clientWidth;
  const bodyHeight = body.clientHeight;

  if (scaledWidth <= bodyWidth) {
    preview.offsetX = (bodyWidth - scaledWidth) / 2;
  } else {
    const minX = bodyWidth - scaledWidth;
    preview.offsetX = Math.min(0, Math.max(minX, preview.offsetX));
  }

  if (scaledHeight <= bodyHeight) {
    preview.offsetY = (bodyHeight - scaledHeight) / 2;
  } else {
    const minY = bodyHeight - scaledHeight;
    preview.offsetY = Math.min(0, Math.max(minY, preview.offsetY));
  }
}

function renderImagePreview() {
  const { image } = getImagePreviewElements();
  const preview = state.imagePreview;
  image.style.transform = `translate(${preview.offsetX}px, ${preview.offsetY}px) scale(${preview.scale})`;
}

function fitImagePreviewToViewport() {
  const { body } = getImagePreviewElements();
  const preview = state.imagePreview;
  if (!preview.naturalWidth || !preview.naturalHeight || !body.clientWidth || !body.clientHeight) {
    return;
  }
  const scaleX = body.clientWidth / preview.naturalWidth;
  const scaleY = body.clientHeight / preview.naturalHeight;
  preview.minScale = Math.min(scaleX, scaleY, 1);
  preview.scale = preview.minScale;
  preview.offsetX = (body.clientWidth - preview.naturalWidth * preview.scale) / 2;
  preview.offsetY = (body.clientHeight - preview.naturalHeight * preview.scale) / 2;
  renderImagePreview();
}

function zoomImagePreview(deltaScale, clientX, clientY) {
  const { body } = getImagePreviewElements();
  const preview = state.imagePreview;
  if (!preview.naturalWidth || !preview.naturalHeight) {
    return;
  }

  const rect = body.getBoundingClientRect();
  const pointX = clientX - rect.left;
  const pointY = clientY - rect.top;
  const nextScale = Math.min(preview.maxScale, Math.max(preview.minScale, preview.scale * deltaScale));
  if (Math.abs(nextScale - preview.scale) < 1e-6) {
    return;
  }

  const imageX = (pointX - preview.offsetX) / preview.scale;
  const imageY = (pointY - preview.offsetY) / preview.scale;
  preview.scale = nextScale;
  preview.offsetX = pointX - imageX * preview.scale;
  preview.offsetY = pointY - imageY * preview.scale;
  clampImagePreviewOffsets();
  renderImagePreview();
}

function resetImagePreviewState() {
  const { body, image } = getImagePreviewElements();
  state.imagePreview.naturalWidth = 0;
  state.imagePreview.naturalHeight = 0;
  state.imagePreview.scale = 1;
  state.imagePreview.minScale = 1;
  state.imagePreview.offsetX = 0;
  state.imagePreview.offsetY = 0;
  state.imagePreview.dragging = false;
  body.classList.remove("dragging");
  image.removeAttribute("src");
  image.style.transform = "";
}

function refreshAppendEntryState() {
  const input = document.getElementById("appendImageFile");
  const entry = document.getElementById("appendUploadEntry");
  const subtitle = document.getElementById("appendUploadSubtitle");
  const enabled = Boolean(state.selectedModelId);
  input.disabled = !enabled;
  entry.classList.toggle("disabled", !enabled);
  subtitle.textContent = enabled
    ? "选择图片后弹出轮廓提取与编辑窗口"
    : "请先选择模型";
}

function refreshDetectEntryState() {
  const input = document.getElementById("detectImageFile");
  const entry = document.getElementById("detectUploadEntry");
  const subtitle = document.getElementById("detectUploadSubtitle");
  const meta = document.getElementById("detectResultMeta");
  const rerunBtn = document.getElementById("rerunDetectBtn");
  const enabled = Boolean(state.selectedModelId);
  input.disabled = !enabled;
  rerunBtn.disabled = !enabled;
  entry.classList.toggle("disabled", !enabled);
  subtitle.textContent = enabled
    ? "选择图片后开始异物检测"
    : "请先选择模型";
  if (!enabled) {
    meta.textContent = "请先选择模型";
    document.getElementById("modelThresholdInput").value = "";
    document.getElementById("modelThresholdInput").disabled = true;
    document.getElementById("saveModelThresholdBtn").disabled = true;
    document.getElementById("detectThreshold").value = "";
    document.getElementById("appendThreshold").value = "";
    document.getElementById("detectHeatmapIncludeBackground").checked = true;
    document.getElementById("detectHeatmapZeroBelowThreshold").checked = true;
    state.selectedModelThreshold = null;
    updateThresholdInputs(null);
  }
}

function refreshSamplesEntryState() {
  const btn = document.getElementById("openSamplesBtn");
  const pruneBtn = document.getElementById("pruneModelAssetsBtn");
  const hint = document.getElementById("samplesEntryHint");
  const hasModel = Boolean(state.selectedModelId);
  const available = hasModel && state.selectedModelSampleAssetsAvailable;
  btn.disabled = !available;
  pruneBtn.disabled = !hasModel;
  btn.textContent = available ? "查看向量库样本" : "样本文件不可查看";
  const meta = document.getElementById("samplesModalMeta");
  if (!hasModel) {
    meta.textContent = "请选择模型后查看";
    hint.textContent = "请选择模型后查看";
    pruneBtn.textContent = "精简模型文件";
  } else if (!state.selectedModelSampleAssetsAvailable) {
    meta.textContent = `当前模型: ${state.selectedModelId}`;
    hint.textContent = `${state.selectedModelStorageMessage || "当前模型缺少样本文件，无法查看向量库样本。"} 如需释放空间，模型已经是精简状态。`;
    pruneBtn.textContent = "模型已精简";
  } else {
    meta.textContent = `当前模型: ${state.selectedModelId}`;
    hint.textContent = "可查看原始样本、处理图和子图。也可执行“精简模型文件”，删除不影响检测和后续追加正样本的样本派生文件。";
    pruneBtn.textContent = "精简模型文件";
  }
}

function refreshModelTransferActions() {
  const exportBtn = document.getElementById("exportModelBtn");
  exportBtn.disabled = !state.selectedModelId;
}

function refreshInferenceEntryState() {
  const input = document.getElementById("inferImageFile");
  const entry = document.getElementById("inferUploadEntry");
  const subtitle = document.getElementById("inferUploadSubtitle");
  const rerunBtn = document.getElementById("rerunInferenceBtn");
  const enabled = Boolean(state.selectedInferenceModelName);
  input.disabled = !enabled;
  rerunBtn.disabled = !enabled;
  entry.classList.toggle("disabled", !enabled);
  subtitle.textContent = enabled
    ? "选择图片后开始检测"
    : "请先选择检测模型";
  if (!enabled) {
    document.getElementById("inferConfThreshold").value = "";
    document.getElementById("inferIouThreshold").value = "";
    document.getElementById("inferImgsz").value = "";
    document.getElementById("inferMaxDet").value = "";
    document.getElementById("inferDevice").value = "";
  }
}

function resetInferenceResultState(message = "请先选择检测模型") {
  document.getElementById("inferOriginalImage").src = "";
  document.getElementById("inferOriginalImage").classList.add("hidden");
  document.getElementById("inferAnnotatedImage").src = "";
  document.getElementById("inferAnnotatedImage").classList.add("hidden");
  document.getElementById("inferResultMeta").textContent = message;
  document.getElementById("inferResultJson").textContent = "等待推理";
}

function setInferenceBusyState(busy) {
  const btn = document.getElementById("rerunInferenceBtn");
  btn.disabled = busy || !state.selectedInferenceModelName;
  btn.textContent = busy ? "推理中..." : "重新推理";
}

function syncInferenceInputsFromConfig(config) {
  document.getElementById("inferConfThreshold").placeholder =
    config ? `默认 ${Number(config.conf_threshold).toFixed(2)}` : "留空则使用模型默认值";
  document.getElementById("inferIouThreshold").placeholder =
    config ? `默认 ${Number(config.iou_threshold).toFixed(2)}` : "留空则使用模型默认值";
  document.getElementById("inferImgsz").placeholder =
    config ? `默认 ${config.imgsz}` : "留空则使用模型默认值";
  document.getElementById("inferMaxDet").placeholder =
    config ? `默认 ${config.max_det}` : "留空则使用模型默认值";
}

function setModelTransferStatus({ visible = false, title = "模型传输中", percent = 0, message = "准备开始" } = {}) {
  const root = document.getElementById("modelTransferStatus");
  const titleNode = document.getElementById("modelTransferTitle");
  const percentNode = document.getElementById("modelTransferPercent");
  const barNode = document.getElementById("modelTransferBar");
  const messageNode = document.getElementById("modelTransferMessage");
  root.classList.toggle("hidden", !visible);
  titleNode.textContent = title;
  const normalizedPercent = Math.max(0, Math.min(100, Number(percent) || 0));
  percentNode.textContent = `${normalizedPercent.toFixed(0)}%`;
  barNode.style.width = `${normalizedPercent}%`;
  messageNode.textContent = message;
}

function resetModelTransferStatus() {
  setModelTransferStatus({ visible: false, percent: 0, message: "准备开始" });
}

function refreshModelThresholdEditor() {
  const input = document.getElementById("modelThresholdInput");
  const button = document.getElementById("saveModelThresholdBtn");
  const deleteBtn = document.getElementById("deleteModelBtn");
  const enabled = Boolean(state.selectedModelId);
  input.disabled = !enabled;
  button.disabled = !enabled;
  deleteBtn.disabled = !enabled;
  input.placeholder = enabled ? "输入新的默认 threshold" : "选择模型后可修改";
  input.value = Number.isFinite(state.selectedModelThreshold) ? String(state.selectedModelThreshold) : "";
}

function setAppendHeatmapState({ message = "等待检测", imageUrl = "", visible = false } = {}) {
  const meta = document.getElementById("appendHeatmapMeta");
  const image = document.getElementById("appendHeatmapImage");
  meta.textContent = message;
  image.src = imageUrl || "";
  image.classList.toggle("hidden", !visible || !imageUrl);
}

function updateAppendPreview() {
  const preview = document.getElementById("appendPreviewImage");
  const placeholder = document.getElementById("appendPreviewPlaceholder");
  if (!state.appendEditor || !state.appendEditor.image) {
    preview.src = "";
    preview.classList.add("hidden");
    placeholder.classList.remove("hidden");
    state.appendPreviewUrl = "";
    return;
  }
  state.appendPreviewUrl = state.appendEditor.toDataUrl();
  preview.src = state.appendPreviewUrl;
  preview.classList.remove("hidden");
  placeholder.classList.add("hidden");
}

function resetAppendModalState() {
  document.getElementById("appendImageFile").value = "";
  document.getElementById("appendThreshold").value =
    Number.isFinite(state.selectedModelThreshold) ? String(state.selectedModelThreshold) : "";
  document.getElementById("appendMaxVectors").value = "20";
  document.getElementById("appendHeatmapIncludeBackground").checked = true;
  document.getElementById("appendHeatmapZeroBelowThreshold").checked = true;
  state.appendFile = null;
  state.appendPreviewUrl = "";
  state.appendEditor.reset();
  document.getElementById("appendPreviewImage").src = "";
  document.getElementById("appendPreviewImage").classList.add("hidden");
  document.getElementById("appendPreviewPlaceholder").classList.remove("hidden");
  setAppendHeatmapState();
}

function resetDetectResultState(message = "等待检测") {
  document.getElementById("detectHeatmapImage").src = "";
  document.getElementById("detectHeatmapImage").classList.add("hidden");
  document.getElementById("detectAnnotatedImage").src = "";
  document.getElementById("detectAnnotatedImage").classList.add("hidden");
  document.getElementById("detectResultMeta").textContent = message;
}

function setDetectBusyState(busy) {
  const btn = document.getElementById("rerunDetectBtn");
  btn.disabled = busy || !state.selectedModelId;
  btn.textContent = busy ? "检测中..." : "重新检测";
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function buildAnnotatedDetectImage(file, result) {
  const img = await loadImageFromFile(file);
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const baseSize = Math.max(img.width, img.height);
  const roofLineWidth = Math.max(2, Math.round(baseSize / 500));
  const anomalyLineWidth = Math.max(2, Math.round(baseSize / 500));
  const boxLineWidth = Math.max(2, Math.round(baseSize / 450));
  const fontSize = Math.max(20, Math.round(baseSize / 64));
  const fontPaddingX = Math.max(6, Math.round(fontSize * 0.3));
  const fontPaddingY = Math.max(4, Math.round(fontSize * 0.22));
  const labelHeight = fontSize + fontPaddingY * 2;
  const labelOffsetY = Math.max(8, Math.round(fontSize * 0.35));

  for (const contour of result.roof_contours || []) {
    if (!Array.isArray(contour) || contour.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(contour[0][0], contour[0][1]);
    contour.slice(1).forEach((pt) => ctx.lineTo(pt[0], pt[1]));
    ctx.closePath();
    ctx.strokeStyle = "#00b26f";
    ctx.lineWidth = roofLineWidth;
    ctx.stroke();
  }

  (result.anomaly_regions || []).forEach((region, index) => {
    const contour = region.contour || [];
    if (Array.isArray(contour) && contour.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(contour[0][0], contour[0][1]);
      contour.slice(1).forEach((pt) => ctx.lineTo(pt[0], pt[1]));
      ctx.closePath();
      ctx.strokeStyle = "#e13a2f";
      ctx.lineWidth = anomalyLineWidth;
      ctx.stroke();
    }

    const box = region.box || [];
    if (box.length === 4) {
      const [x1, y1, x2, y2] = box;
      ctx.strokeStyle = "#ffd24a";
      ctx.lineWidth = boxLineWidth;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const score = Number.isFinite(region.score) ? Number(region.score).toFixed(3) : "-";
      const label = `A${index + 1}:${score}`;
      ctx.font = `${fontSize}px sans-serif`;
      ctx.textBaseline = "top";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "#ffd24a";
      ctx.fillRect(
        x1,
        Math.max(0, y1 - labelHeight - labelOffsetY),
        textWidth + fontPaddingX * 2,
        labelHeight,
      );
      ctx.fillStyle = "#1b1f1d";
      ctx.fillText(label, x1 + fontPaddingX, Math.max(0, y1 - labelHeight - labelOffsetY) + fontPaddingY);
    }
  });

  return canvas.toDataURL("image/jpeg", 0.92);
}

async function buildInferenceAnnotatedImage(file, result) {
  const img = await loadImageFromFile(file);
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const baseSize = Math.max(img.width, img.height);
  const lineWidth = Math.max(2, Math.round(baseSize / 500));
  const fontSize = Math.max(20, Math.round(baseSize / 64));
  const fontPaddingX = Math.max(6, Math.round(fontSize * 0.3));
  const fontPaddingY = Math.max(4, Math.round(fontSize * 0.22));
  const labelHeight = fontSize + fontPaddingY * 2;

  (result.detections || []).forEach((item, index) => {
    const [x1, y1, x2, y2] = item.box || [];
    if (![x1, y1, x2, y2].every(Number.isFinite)) {
      return;
    }
    const label = `${index + 1}. ${item.class_name || item.class_id} ${(Number(item.confidence) || 0).toFixed(3)}`;
    const color = item.class_name === "fire" ? "#237cff" : "#47c4ff";
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.font = `${fontSize}px "Source Han Sans SC", sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const labelY = Math.max(0, y1 - labelHeight - 6);
    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, textWidth + fontPaddingX * 2, labelHeight);
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x1 + fontPaddingX, labelY + labelHeight / 2);
  });

  return canvas.toDataURL("image/jpeg", 0.92);
}

function pointDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function pointLineDistance(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) {
    return pointDistance(point, start);
  }
  const numerator = Math.abs((dy * point.x) - (dx * point.y) + (end.x * start.y) - (end.y * start.x));
  const denominator = Math.hypot(dx, dy);
  return numerator / denominator;
}

function rdpSimplify(points, epsilon) {
  if (points.length <= 2) {
    return points.slice();
  }

  let maxDistance = 0;
  let splitIndex = -1;
  const start = points[0];
  const end = points[points.length - 1];

  for (let i = 1; i < points.length - 1; i += 1) {
    const distance = pointLineDistance(points[i], start, end);
    if (distance > maxDistance) {
      maxDistance = distance;
      splitIndex = i;
    }
  }

  if (maxDistance <= epsilon || splitIndex === -1) {
    return [start, end];
  }

  const left = rdpSimplify(points.slice(0, splitIndex + 1), epsilon);
  const right = rdpSimplify(points.slice(splitIndex), epsilon);
  return left.slice(0, -1).concat(right);
}

function simplifyPolygonPoints(points) {
  if (!Array.isArray(points) || points.length <= 3) {
    return Array.isArray(points) ? points.slice() : [];
  }

  const normalized = points.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  const bounds = normalized.reduce((acc, point) => ({
    minX: Math.min(acc.minX, point.x),
    minY: Math.min(acc.minY, point.y),
    maxX: Math.max(acc.maxX, point.x),
    maxY: Math.max(acc.maxY, point.y),
  }), {
    minX: normalized[0].x,
    minY: normalized[0].y,
    maxX: normalized[0].x,
    maxY: normalized[0].y,
  });
  const size = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, 1);
  const epsilon = Math.max(2, size * 0.008);
  const openPolyline = normalized.concat([normalized[0]]);
  const approx = rdpSimplify(openPolyline, epsilon).slice(0, -1);

  if (approx.length < 3) {
    return normalized;
  }

  const simplified = [];
  const tolerance = 0.75;

  for (let i = 0; i < approx.length; i += 1) {
    const prev = approx[(i - 1 + approx.length) % approx.length];
    const curr = approx[i];
    const next = approx[(i + 1) % approx.length];
    const v1x = curr.x - prev.x;
    const v1y = curr.y - prev.y;
    const v2x = next.x - curr.x;
    const v2y = next.y - curr.y;
    const cross = (v1x * v2y) - (v1y * v2x);
    if (Math.abs(cross) > tolerance || simplified.length < 2) {
      simplified.push(curr);
    }
  }

  return simplified.length >= 3 ? simplified : approx;
}

function simplifyPolygons(polygons) {
  return (polygons || [])
    .map((poly) => simplifyPolygonPoints(poly))
    .filter((poly) => poly.length >= 3);
}

function createPolygonEditor(canvasId, textareaId, options = {}) {
  const canvas = document.getElementById(canvasId);
  const textarea = document.getElementById(textareaId);
  const ctx = canvas.getContext("2d");
  const editor = {
    canvas,
    textarea,
    ctx,
    image: null,
    polygons: [],
    selectedPolygon: -1,
    draggingPoint: null,
    drawMode: false,
    tempPolygon: [],
    drawStart: null,
    scaleX: 1,
    scaleY: 1,
    polygonKinds: [],
    rectangleOnly: Boolean(options.rectangleOnly),
    replaceOnDraw: Boolean(options.replaceOnDraw),
    previewScaleLimit: Number.isFinite(Number(options.previewScaleLimit)) ? Number(options.previewScaleLimit) : 1,
    colorResolver: typeof options.colorResolver === "function" ? options.colorResolver : null,
    onChange: typeof options.onChange === "function" ? options.onChange : null,
  };

  function buildRectangle(start, end) {
    const x1 = Math.min(start.x, end.x);
    const y1 = Math.min(start.y, end.y);
    const x2 = Math.max(start.x, end.x);
    const y2 = Math.max(start.y, end.y);
    return [
      { x: x1, y: y1 },
      { x: x2, y: y1 },
      { x: x2, y: y2 },
      { x: x1, y: y2 },
    ];
  }

  function resizeCanvas(w, h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }

  function syncTextarea(simplify = false) {
    if (simplify) {
      const simplified = simplifyPolygons(editor.polygons);
      editor.polygonKinds = simplified.map((_, idx) => editor.polygonKinds[idx] || "manual");
      editor.polygons = simplified;
      if (editor.selectedPolygon >= editor.polygons.length) {
        editor.selectedPolygon = editor.polygons.length - 1;
      }
    }
    textarea.value = JSON.stringify(
      editor.polygons.map((poly) => poly.map((pt) => [Math.round(pt.x * editor.scaleX), Math.round(pt.y * editor.scaleY)])),
    );
  }

  function drawPolygon(poly, color, fill = false) {
    if (!poly || poly.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(poly[0].x, poly[0].y);
    poly.slice(1).forEach((pt) => ctx.lineTo(pt.x, pt.y));
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (fill) {
      ctx.fillStyle = `${color}14`;
      ctx.fill();
    }
    poly.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!editor.image) {
      ctx.fillStyle = "#7d817d";
      ctx.font = "16px sans-serif";
      ctx.fillText(editor.rectangleOnly ? "选择图片后会自动提取轮廓，也可框选矩形区域" : "选择图片后会自动提取轮廓，也可手工新建轮廓点", 20, 36);
      return;
    }
    ctx.drawImage(editor.image, 0, 0, canvas.width, canvas.height);
    editor.polygons.forEach((poly, idx) => {
      const kind = editor.polygonKinds[idx] || "manual";
      const colors = editor.colorResolver
        ? editor.colorResolver({ index: idx, selected: idx === editor.selectedPolygon, kind })
        : {
          stroke: idx === editor.selectedPolygon ? "#ff5a36" : "#1f8c6f",
          fill: idx === editor.selectedPolygon,
        };
      drawPolygon(poly, colors.stroke, Boolean(colors.fill));
    });
    if (editor.tempPolygon.length > 0) {
      drawPolygon(editor.tempPolygon, "#375ef5", false);
    }
    if (editor.onChange) {
      editor.onChange(editor);
    }
  }

  function getMousePos(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function hitPoint(pos) {
    for (let p = 0; p < editor.polygons.length; p += 1) {
      for (let i = 0; i < editor.polygons[p].length; i += 1) {
        if (pointDistance(pos, editor.polygons[p][i]) < 10) {
          return { polygonIndex: p, pointIndex: i };
        }
      }
    }
    return null;
  }

  function hitPolygon(pos) {
    for (let p = editor.polygons.length - 1; p >= 0; p -= 1) {
      const poly = editor.polygons[p];
      ctx.beginPath();
      ctx.moveTo(poly[0].x, poly[0].y);
      poly.slice(1).forEach((pt) => ctx.lineTo(pt.x, pt.y));
      ctx.closePath();
      if (ctx.isPointInPath(pos.x, pos.y)) {
        return p;
      }
    }
    return -1;
  }

  editor.loadImageFromUrl = (url) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const bounds = canvas.parentElement.getBoundingClientRect();
      const maxWidth = Math.max(240, Math.floor(bounds.width) - 4);
      const computed = window.getComputedStyle(canvas);
      const maxHeight = Number.parseFloat(computed.maxHeight) || 340;
      const scale = Math.min(maxWidth / img.width, maxHeight / img.height, editor.previewScaleLimit);
      const width = Math.max(1, Math.round(img.width * scale));
      const height = Math.max(1, Math.round(img.height * scale));
      resizeCanvas(width, height);
      canvas.dataset.imageWidth = String(img.width);
      canvas.dataset.imageHeight = String(img.height);
      console.debug(`[${canvasId}] image=${img.width}x${img.height} canvas=${width}x${height} scale=${scale.toFixed(4)}`);
      editor.scaleX = img.width / width;
      editor.scaleY = img.height / height;
      editor.image = img;
      editor.polygons = [];
      editor.polygonKinds = [];
      editor.selectedPolygon = -1;
      editor.tempPolygon = [];
      editor.drawMode = false;
      syncTextarea(true);
      redraw();
      resolve();
    };
    img.onerror = reject;
    img.src = url;
  });

  editor.setPolygonEntries = (entries) => {
    editor.polygons = simplifyPolygons(
      (entries || []).map((entry) => (entry.points || []).map((pt) => ({
        x: pt[0] / editor.scaleX,
        y: pt[1] / editor.scaleY,
      }))),
    );
    editor.polygonKinds = (entries || []).slice(0, editor.polygons.length).map((entry) => entry.kind || "manual");
    editor.selectedPolygon = -1;
    editor.tempPolygon = [];
    syncTextarea(true);
    redraw();
  };

  editor.setPolygons = (payload, kind = "manual") => {
    editor.setPolygonEntries((payload || []).map((points) => ({ points, kind })));
  };

  editor.getPolygons = () => simplifyPolygons(editor.polygons)
    .map((poly) => poly.map((pt) => [Math.round(pt.x * editor.scaleX), Math.round(pt.y * editor.scaleY)]));

  editor.enableDrawMode = () => {
    editor.drawMode = true;
    editor.tempPolygon = [];
    editor.drawStart = null;
    if (editor.rectangleOnly && editor.replaceOnDraw) {
      editor.polygons = [];
      editor.polygonKinds = [];
      syncTextarea(true);
    }
    editor.selectedPolygon = -1;
    redraw();
  };

  editor.deleteSelected = () => {
    if (editor.selectedPolygon < 0) return;
    editor.polygons.splice(editor.selectedPolygon, 1);
    editor.polygonKinds.splice(editor.selectedPolygon, 1);
    editor.selectedPolygon = -1;
    syncTextarea(true);
    redraw();
  };

  editor.toDataUrl = () => canvas.toDataURL("image/jpeg", 0.92);

  canvas.addEventListener("mousedown", (event) => {
    if (!editor.image) return;
    const pos = getMousePos(event);
    if (editor.drawMode && editor.rectangleOnly) {
      editor.drawStart = pos;
      editor.tempPolygon = buildRectangle(pos, pos);
      redraw();
      return;
    }
    const point = hitPoint(pos);
    if (point) {
      editor.selectedPolygon = point.polygonIndex;
      editor.draggingPoint = point;
      redraw();
      return;
    }
    if (editor.drawMode) {
      editor.tempPolygon.push(pos);
      redraw();
      return;
    }
    const polygonIndex = hitPolygon(pos);
    editor.selectedPolygon = polygonIndex;
    redraw();
  });

  canvas.addEventListener("mousemove", (event) => {
    if (editor.drawMode && editor.rectangleOnly && editor.drawStart) {
      const pos = getMousePos(event);
      editor.tempPolygon = buildRectangle(editor.drawStart, pos);
      redraw();
      return;
    }
    if (!editor.draggingPoint) return;
    const pos = getMousePos(event);
    const { polygonIndex, pointIndex } = editor.draggingPoint;
    editor.polygons[polygonIndex][pointIndex] = pos;
    syncTextarea();
    redraw();
  });

  window.addEventListener("mouseup", () => {
    if (editor.drawMode && editor.rectangleOnly && editor.drawStart) {
      const rect = editor.tempPolygon.slice();
      const width = Math.abs(rect[1].x - rect[0].x);
      const height = Math.abs(rect[3].y - rect[0].y);
      editor.drawStart = null;
      editor.tempPolygon = [];
      editor.drawMode = false;
      if (width >= 3 && height >= 3) {
        if (editor.replaceOnDraw) {
          editor.polygons = [];
          editor.polygonKinds = [];
        }
        editor.polygons.push(rect);
        editor.polygonKinds.push("manual");
        editor.selectedPolygon = editor.polygons.length - 1;
        syncTextarea(true);
      }
      redraw();
      return;
    }
    editor.draggingPoint = null;
  });

  canvas.addEventListener("dblclick", () => {
    if (!editor.drawMode || editor.tempPolygon.length < 3) return;
    const nextPolygons = simplifyPolygons([editor.tempPolygon.slice()]);
    editor.polygons.push(...nextPolygons);
    editor.polygonKinds.push(...nextPolygons.map(() => "manual"));
    editor.selectedPolygon = editor.polygons.length - 1;
    editor.tempPolygon = [];
    editor.drawMode = false;
    syncTextarea(true);
    redraw();
  });

  textarea.addEventListener("change", () => {
    if (!editor.image) return;
    try {
      const payload = JSON.parse(textarea.value || "[]");
      editor.setPolygons(payload);
    } catch (_) {
    }
  });

  redraw();
  return editor;
}

function createAppendRectEditor(canvasId, textareaId, options = {}) {
  const canvas = document.getElementById(canvasId);
  const textarea = document.getElementById(textareaId);
  const ctx = canvas.getContext("2d");
  const editor = {
    canvas,
    textarea,
    ctx,
    image: null,
    autoPolygons: [],
    manualPolygons: [],
    selectedManualIndex: -1,
    drawMode: false,
    drawStart: null,
    tempPolygon: [],
    viewportScale: 1,
    fitScale: 1,
    offsetX: 0,
    offsetY: 0,
    dragMode: null,
    dragPoint: null,
    lastCanvasPos: null,
    lastImagePos: null,
    onChange: typeof options.onChange === "function" ? options.onChange : null,
  };

  function buildRectangle(start, end) {
    const x1 = Math.min(start.x, end.x);
    const y1 = Math.min(start.y, end.y);
    const x2 = Math.max(start.x, end.x);
    const y2 = Math.max(start.y, end.y);
    return [
      { x: x1, y: y1 },
      { x: x2, y: y1 },
      { x: x2, y: y2 },
      { x: x1, y: y2 },
    ];
  }

  function getCanvasBounds() {
    const parent = canvas.parentElement;
    const bounds = parent.getBoundingClientRect();
    const computed = window.getComputedStyle(canvas);
    const maxHeight = Number.parseFloat(computed.maxHeight) || 980;
    const minHeight = Number.parseFloat(computed.minHeight) || 420;
    const width = Math.max(320, Math.floor(bounds.width) - 4);
    const height = Math.max(minHeight, Math.min(maxHeight, Math.floor(window.innerHeight * 0.78)));
    return { width, height };
  }

  function resizeCanvas() {
    const { width, height } = getCanvasBounds();
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }

  function clampPoint(point) {
    if (!editor.image) {
      return point;
    }
    return {
      x: Math.min(editor.image.width, Math.max(0, point.x)),
      y: Math.min(editor.image.height, Math.max(0, point.y)),
    };
  }

  function imageToCanvas(point) {
    return {
      x: editor.offsetX + (point.x * editor.viewportScale),
      y: editor.offsetY + (point.y * editor.viewportScale),
    };
  }

  function canvasToImage(point) {
    return clampPoint({
      x: (point.x - editor.offsetX) / editor.viewportScale,
      y: (point.y - editor.offsetY) / editor.viewportScale,
    });
  }

  function getMousePos(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function syncTextarea() {
    const payload = editor.getPolygons();
    textarea.value = JSON.stringify(payload);
  }

  function refreshCursor() {
    if (editor.drawMode) {
      canvas.style.cursor = "crosshair";
      return;
    }
    if (editor.dragMode === "pan") {
      canvas.style.cursor = "grabbing";
      return;
    }
    if (editor.dragMode === "polygon") {
      canvas.style.cursor = "move";
      return;
    }
    if (editor.dragMode === "point") {
      canvas.style.cursor = "grabbing";
      return;
    }
    canvas.style.cursor = "grab";
  }

  function triggerChange() {
    syncTextarea();
    if (editor.onChange) {
      editor.onChange(editor);
    }
  }

  function drawPolygon(poly, strokeStyle, fillStyle = "", showHandles = false) {
    if (!poly || poly.length < 2) {
      return;
    }
    const canvasPoints = poly.map(imageToCanvas);
    ctx.beginPath();
    ctx.moveTo(canvasPoints[0].x, canvasPoints[0].y);
    canvasPoints.slice(1).forEach((pt) => ctx.lineTo(pt.x, pt.y));
    ctx.closePath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (fillStyle) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
    if (showHandles) {
      canvasPoints.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = strokeStyle;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
  }

  function fitImageToViewport() {
    if (!editor.image) {
      return;
    }
    resizeCanvas();
    editor.fitScale = Math.min(canvas.width / editor.image.width, canvas.height / editor.image.height, 1);
    editor.viewportScale = editor.fitScale;
    editor.offsetX = (canvas.width - editor.image.width * editor.viewportScale) / 2;
    editor.offsetY = (canvas.height - editor.image.height * editor.viewportScale) / 2;
  }

  function clampOffsets() {
    if (!editor.image) {
      return;
    }
    const scaledWidth = editor.image.width * editor.viewportScale;
    const scaledHeight = editor.image.height * editor.viewportScale;
    if (scaledWidth <= canvas.width) {
      editor.offsetX = (canvas.width - scaledWidth) / 2;
    } else {
      editor.offsetX = Math.min(0, Math.max(canvas.width - scaledWidth, editor.offsetX));
    }
    if (scaledHeight <= canvas.height) {
      editor.offsetY = (canvas.height - scaledHeight) / 2;
    } else {
      editor.offsetY = Math.min(0, Math.max(canvas.height - scaledHeight, editor.offsetY));
    }
  }

  function redraw() {
    refreshCursor();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!editor.image) {
      ctx.fillStyle = "#7d817d";
      ctx.font = "16px sans-serif";
      ctx.fillText("选择图片后可进入大图标注，支持滚轮缩放与拖动。", 20, 36);
      return;
    }

    ctx.drawImage(
      editor.image,
      editor.offsetX,
      editor.offsetY,
      editor.image.width * editor.viewportScale,
      editor.image.height * editor.viewportScale,
    );

    editor.autoPolygons.forEach((poly) => {
      drawPolygon(poly, "#1f8c6f", "rgba(31, 140, 111, 0.08)", false);
    });

    editor.manualPolygons.forEach((poly, idx) => {
      const selected = idx === editor.selectedManualIndex;
      drawPolygon(
        poly,
        "#ff5a36",
        selected ? "rgba(255, 90, 54, 0.12)" : "",
        selected,
      );
    });

    if (editor.tempPolygon.length > 0) {
      drawPolygon(editor.tempPolygon, "#375ef5", "", false);
    }
  }

  function isPointInPolygon(point, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
      const xi = poly[i].x;
      const yi = poly[i].y;
      const xj = poly[j].x;
      const yj = poly[j].y;
      const intersect = ((yi > point.y) !== (yj > point.y))
        && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-6) + xi);
      if (intersect) {
        inside = !inside;
      }
    }
    return inside;
  }

  function hitManualPoint(canvasPoint) {
    for (let polygonIndex = 0; polygonIndex < editor.manualPolygons.length; polygonIndex += 1) {
      const poly = editor.manualPolygons[polygonIndex];
      for (let pointIndex = 0; pointIndex < poly.length; pointIndex += 1) {
        const pt = imageToCanvas(poly[pointIndex]);
        if (pointDistance(canvasPoint, pt) <= 10) {
          return { polygonIndex, pointIndex };
        }
      }
    }
    return null;
  }

  function hitManualPolygon(imagePoint) {
    for (let polygonIndex = editor.manualPolygons.length - 1; polygonIndex >= 0; polygonIndex -= 1) {
      if (isPointInPolygon(imagePoint, editor.manualPolygons[polygonIndex])) {
        return polygonIndex;
      }
    }
    return -1;
  }

  function clampPolygonDelta(poly, dx, dy) {
    const xs = poly.map((pt) => pt.x);
    const ys = poly.map((pt) => pt.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const clampedDx = Math.min(editor.image.width - maxX, Math.max(-minX, dx));
    const clampedDy = Math.min(editor.image.height - maxY, Math.max(-minY, dy));
    return { dx: clampedDx, dy: clampedDy };
  }

  function zoomAt(clientX, clientY, factor) {
    if (!editor.image) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const canvasPoint = { x: clientX - rect.left, y: clientY - rect.top };
    const imagePoint = canvasToImage(canvasPoint);
    const minScale = editor.fitScale;
    const maxScale = editor.fitScale * 8;
    const nextScale = Math.min(maxScale, Math.max(minScale, editor.viewportScale * factor));
    if (Math.abs(nextScale - editor.viewportScale) < 1e-6) {
      return;
    }
    editor.viewportScale = nextScale;
    editor.offsetX = canvasPoint.x - imagePoint.x * editor.viewportScale;
    editor.offsetY = canvasPoint.y - imagePoint.y * editor.viewportScale;
    clampOffsets();
    redraw();
  }

  editor.loadImageFromUrl = (url) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      editor.image = img;
      editor.autoPolygons = [];
      editor.manualPolygons = [];
      editor.selectedManualIndex = -1;
      editor.tempPolygon = [];
      editor.drawMode = false;
      fitImageToViewport();
      triggerChange();
      redraw();
      resolve();
    };
    img.onerror = reject;
    img.src = url;
  });

  editor.setPolygonEntries = (entries) => {
    editor.autoPolygons = [];
    editor.manualPolygons = [];
    (entries || []).forEach((entry) => {
      const target = entry.kind === "manual" ? editor.manualPolygons : editor.autoPolygons;
      target.push((entry.points || []).map((pt) => clampPoint({ x: Number(pt[0]), y: Number(pt[1]) })));
    });
    editor.selectedManualIndex = editor.manualPolygons.length > 0 ? 0 : -1;
    triggerChange();
    redraw();
  };

  editor.setPolygons = (payload, kind = "manual") => {
    editor.setPolygonEntries((payload || []).map((points) => ({ points, kind })));
  };

  editor.getPolygons = () => {
    const source = editor.manualPolygons.length > 0 ? editor.manualPolygons : editor.autoPolygons;
    return source.map((poly) => poly.map((pt) => [Math.round(pt.x), Math.round(pt.y)]));
  };

  editor.enableDrawMode = () => {
    if (!editor.image) {
      return;
    }
    editor.drawMode = true;
    editor.drawStart = null;
    editor.tempPolygon = [];
    redraw();
  };

  editor.deleteSelected = () => {
    if (editor.selectedManualIndex < 0) {
      return;
    }
    editor.manualPolygons.splice(editor.selectedManualIndex, 1);
    editor.selectedManualIndex = editor.manualPolygons.length > 0
      ? Math.min(editor.selectedManualIndex, editor.manualPolygons.length - 1)
      : -1;
    triggerChange();
    redraw();
  };

  editor.toDataUrl = () => {
    if (!editor.image) {
      return "";
    }
    const maxLongSide = 1600;
    const scale = Math.min(maxLongSide / Math.max(editor.image.width, editor.image.height), 1);
    const width = Math.max(1, Math.round(editor.image.width * scale));
    const height = Math.max(1, Math.round(editor.image.height * scale));
    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = width;
    previewCanvas.height = height;
    const previewCtx = previewCanvas.getContext("2d");
    previewCtx.drawImage(editor.image, 0, 0, width, height);
    const drawPreviewPolygon = (poly, strokeStyle, fillStyle = "") => {
      if (!poly || poly.length < 2) {
        return;
      }
      previewCtx.beginPath();
      previewCtx.moveTo(poly[0].x * scale, poly[0].y * scale);
      poly.slice(1).forEach((pt) => previewCtx.lineTo(pt.x * scale, pt.y * scale));
      previewCtx.closePath();
      previewCtx.strokeStyle = strokeStyle;
      previewCtx.lineWidth = Math.max(2, Math.round(Math.max(width, height) / 700));
      previewCtx.stroke();
      if (fillStyle) {
        previewCtx.fillStyle = fillStyle;
        previewCtx.fill();
      }
    };
    editor.autoPolygons.forEach((poly) => drawPreviewPolygon(poly, "#1f8c6f", "rgba(31, 140, 111, 0.08)"));
    editor.manualPolygons.forEach((poly) => drawPreviewPolygon(poly, "#ff5a36", "rgba(255, 90, 54, 0.12)"));
    return previewCanvas.toDataURL("image/jpeg", 0.92);
  };

  editor.refreshLayout = ({ fit = false } = {}) => {
    resizeCanvas();
    if (!editor.image) {
      redraw();
      return;
    }
    if (fit) {
      fitImageToViewport();
    } else {
      clampOffsets();
    }
    redraw();
  };

  editor.reset = () => {
    editor.image = null;
    editor.autoPolygons = [];
    editor.manualPolygons = [];
    editor.selectedManualIndex = -1;
    editor.drawMode = false;
    editor.drawStart = null;
    editor.tempPolygon = [];
    editor.dragMode = null;
    editor.dragPoint = null;
    editor.lastCanvasPos = null;
    editor.lastImagePos = null;
    textarea.value = "";
    resizeCanvas();
    redraw();
  };

  canvas.addEventListener("wheel", (event) => {
    if (!editor.image) {
      return;
    }
    event.preventDefault();
    zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 1.12 : (1 / 1.12));
  }, { passive: false });

  canvas.addEventListener("mousedown", (event) => {
    if (!editor.image || event.button !== 0) {
      return;
    }
    const canvasPoint = getMousePos(event);
    const imagePoint = canvasToImage(canvasPoint);

    if (editor.drawMode) {
      editor.drawStart = imagePoint;
      editor.tempPolygon = buildRectangle(imagePoint, imagePoint);
      redraw();
      return;
    }

    const hitPoint = hitManualPoint(canvasPoint);
    if (hitPoint) {
      editor.selectedManualIndex = hitPoint.polygonIndex;
      editor.dragMode = "point";
      editor.dragPoint = hitPoint;
      redraw();
      return;
    }

    const manualPolygonIndex = hitManualPolygon(imagePoint);
    if (manualPolygonIndex >= 0) {
      editor.selectedManualIndex = manualPolygonIndex;
      editor.dragMode = "polygon";
      editor.lastImagePos = imagePoint;
      redraw();
      return;
    }

    editor.selectedManualIndex = -1;
    editor.dragMode = "pan";
    editor.lastCanvasPos = canvasPoint;
    redraw();
  });

  canvas.addEventListener("mousemove", (event) => {
    if (!editor.image) {
      return;
    }
    const canvasPoint = getMousePos(event);
    const imagePoint = canvasToImage(canvasPoint);

    if (editor.drawMode && editor.drawStart) {
      editor.tempPolygon = buildRectangle(editor.drawStart, imagePoint);
      redraw();
      return;
    }

    if (editor.dragMode === "point" && editor.dragPoint) {
      const { polygonIndex, pointIndex } = editor.dragPoint;
      editor.manualPolygons[polygonIndex][pointIndex] = clampPoint(imagePoint);
      triggerChange();
      redraw();
      return;
    }

    if (editor.dragMode === "polygon" && editor.selectedManualIndex >= 0 && editor.lastImagePos) {
      const polygon = editor.manualPolygons[editor.selectedManualIndex];
      const delta = clampPolygonDelta(
        polygon,
        imagePoint.x - editor.lastImagePos.x,
        imagePoint.y - editor.lastImagePos.y,
      );
      editor.manualPolygons[editor.selectedManualIndex] = polygon.map((pt) => ({
        x: pt.x + delta.dx,
        y: pt.y + delta.dy,
      }));
      editor.lastImagePos = imagePoint;
      triggerChange();
      redraw();
      return;
    }

    if (editor.dragMode === "pan" && editor.lastCanvasPos) {
      editor.offsetX += canvasPoint.x - editor.lastCanvasPos.x;
      editor.offsetY += canvasPoint.y - editor.lastCanvasPos.y;
      editor.lastCanvasPos = canvasPoint;
      clampOffsets();
      redraw();
    }
  });

  window.addEventListener("mouseup", () => {
    if (editor.drawMode && editor.drawStart) {
      const rect = editor.tempPolygon.slice();
      const width = Math.abs(rect[1].x - rect[0].x);
      const height = Math.abs(rect[3].y - rect[0].y);
      editor.drawMode = false;
      editor.drawStart = null;
      editor.tempPolygon = [];
      if (width >= 3 && height >= 3) {
        editor.manualPolygons = [rect];
        editor.selectedManualIndex = 0;
        triggerChange();
      }
      redraw();
      return;
    }
    editor.dragMode = null;
    editor.dragPoint = null;
    editor.lastCanvasPos = null;
    editor.lastImagePos = null;
  });

  window.addEventListener("resize", () => {
    resizeCanvas();
    if (editor.image) {
      fitImageToViewport();
    }
    redraw();
  });

  resizeCanvas();
  redraw();
  return editor;
}

function drawTileSourcePreview(rawImageUrl, tile, sampleContour = null) {
  const canvas = document.getElementById("tileSourceCanvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    const maxWidth = canvas.parentElement.clientWidth - 4;
    const scale = Math.min(maxWidth / img.width, 560 / img.height, 1);
    const width = Math.max(1, Math.round(img.width * scale));
    const height = Math.max(1, Math.round(img.height * scale));
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);

    if (sampleContour && sampleContour.length > 2) {
      ctx.beginPath();
      ctx.moveTo(sampleContour[0][0] / img.width * width, sampleContour[0][1] / img.height * height);
      sampleContour.slice(1).forEach((pt) => ctx.lineTo(pt[0] / img.width * width, pt[1] / img.height * height));
      ctx.closePath();
      ctx.strokeStyle = "#10a37f";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    const sampleBox = tile.sample_bbox || [0, 0, img.width, img.height];
    const [sampleX1, sampleY1] = sampleBox;
    const [x1, y1, x2, y2] = tile.box;
    const absX1 = sampleX1 + x1;
    const absY1 = sampleY1 + y1;
    const absX2 = sampleX1 + x2;
    const absY2 = sampleY1 + y2;
    const rx = absX1 / img.width * width;
    const ry = absY1 / img.height * height;
    const rw = (absX2 - absX1) / img.width * width;
    const rh = (absY2 - absY1) / img.height * height;
    ctx.fillStyle = "rgba(255,90,54,0.18)";
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeStyle = "#ff5a36";
    ctx.lineWidth = 3;
    ctx.strokeRect(rx, ry, rw, rh);
  };
  img.src = rawImageUrl;
}

function showTilePreview(sampleId, sampleContour, tile) {
  const preview = document.getElementById("tilePreviewImage");
  preview.src = `/api/models/${state.selectedModelId}/samples/${sampleId}/tiles/${tile.tile_id}/image`;
  preview.classList.remove("hidden");
  document.getElementById("tilePreviewMeta").textContent =
    `tile_id=${tile.tile_id} box=${JSON.stringify(tile.box)} sample_bbox=${JSON.stringify(tile.sample_bbox || [])} enabled=${tile.enabled !== false}`;
  drawTileSourcePreview(
    `/api/models/${state.selectedModelId}/samples/${sampleId}/image?kind=raw`,
    tile,
    sampleContour,
  );
  openModal("tilePreviewModal");
}

function renderTileCanvas(detail) {
  const root = document.getElementById("tileCanvas");
  root.innerHTML = "";
  const canvasSize = detail.canvas_size || [0, 0];
  root.style.width = "100%";
  root.style.height = `${Math.max(canvasSize[1], 260)}px`;
  state.currentSampleDetail = detail;

  (detail.tiles || []).forEach((tile) => {
    const [x1, y1, x2, y2] = tile.display_box;
    const div = document.createElement("div");
    div.className = `tile-item${tile.enabled === false ? " disabled" : ""}`;
    div.style.left = `${x1}px`;
    div.style.top = `${y1}px`;
    div.style.width = `${Math.max(10, x2 - x1)}px`;
    div.style.height = `${Math.max(10, y2 - y1)}px`;
    div.dataset.tileId = tile.tile_id;
    div.onclick = () => {
      tile.enabled = tile.enabled === false;
      div.classList.toggle("disabled", tile.enabled === false);
      showTilePreview(detail.sample.sample_id, detail.sample.contour, tile);
    };
    const img = document.createElement("img");
    img.src = `/api/models/${state.selectedModelId}/samples/${detail.sample.sample_id}/tiles/${tile.tile_id}/image`;
    div.appendChild(img);
    root.appendChild(div);
  });
}

async function loadModels() {
  const data = await api("/api/models");
  const root = document.getElementById("modelsList");
  root.innerHTML = "";
  data.items.forEach((item) => {
    const card = document.createElement("div");
    card.className = `model-card${item.model_id === state.selectedModelId ? " active" : ""}`;
    card.innerHTML = `
      <strong>${item.model_name}</strong>
      <div>${item.model_id}</div>
      <div>当前版本: ${item.current_version_id}</div>
      <div>版本数: ${item.versions.length}</div>
    `;
    card.onclick = async () => {
      state.selectedModelId = item.model_id;
      refreshAppendEntryState();
      refreshDetectEntryState();
      refreshSamplesEntryState();
      state.page = 1;
      await loadModelDetail();
      if (state.selectedModelSampleAssetsAvailable) {
        await loadSamples();
      } else {
        document.getElementById("samplesTable").innerHTML = "";
      }
      await loadModels();
    };
    root.appendChild(card);
  });
  refreshAppendEntryState();
  refreshDetectEntryState();
  refreshSamplesEntryState();
  refreshModelTransferActions();
}

async function loadInferenceModels() {
  const data = await api("/api/inference/models");
  const root = document.getElementById("inferenceModelsList");
  root.innerHTML = "";
  data.items.forEach((item) => {
    const card = document.createElement("div");
    card.className = `model-card${item.name === state.selectedInferenceModelName ? " active" : ""}`;
    card.innerHTML = `
      <strong>${item.name}</strong>
      <div>任务: ${item.task_type}</div>
      <div>后端: ${item.backend}</div>
      <small>${item.description || "内置检测模型"}</small>
    `;
    card.onclick = async () => {
      state.selectedInferenceModelName = item.name;
      await loadInferenceModelDetail();
      await loadInferenceModels();
    };
    root.appendChild(card);
  });
  refreshInferenceEntryState();
}

async function loadInferenceModelDetail() {
  if (!state.selectedInferenceModelName) {
    state.selectedInferenceModelConfig = null;
    document.getElementById("inferModelInfo").textContent = "请选择推理模型";
    resetInferenceResultState();
    refreshInferenceEntryState();
    return;
  }
  const data = await api(`/api/inference/models/${state.selectedInferenceModelName}`);
  state.selectedInferenceModelConfig = data;
  setJson("inferModelInfo", data);
  syncInferenceInputsFromConfig(data);
  document.getElementById("inferResultMeta").textContent = `当前模型: ${data.name} (${data.task_type})`;
  refreshInferenceEntryState();
}

async function loadModelDetail() {
  if (!state.selectedModelId) return;
  const data = await api(`/api/models/${state.selectedModelId}`);
  setJson("modelInfo", data);
  state.selectedModelSampleAssetsAvailable = Boolean(data.storage_status?.sample_assets_available);
  state.selectedModelStorageMessage = data.storage_status?.message || "";
  const currentVersion = Array.isArray(data.versions)
    ? data.versions.find((item) => item.version_id === data.current_version_id)
    : null;
  const threshold = currentVersion?.threshold;
  state.selectedModelThreshold = Number.isFinite(Number(threshold)) ? Number(threshold) : null;
  updateThresholdInputs(state.selectedModelThreshold);
  refreshModelThresholdEditor();
  refreshSamplesEntryState();
  document.getElementById("detectThreshold").value = threshold ?? "";
  document.getElementById("appendThreshold").value = threshold ?? "";
}

async function saveModelThreshold() {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  const raw = document.getElementById("modelThresholdInput").value.trim();
  if (raw === "") {
    alert("请输入默认 threshold");
    return;
  }
  const threshold = Number(raw);
  if (!Number.isFinite(threshold)) {
    alert("默认 threshold 格式不正确");
    return;
  }
  await api(`/api/models/${state.selectedModelId}/threshold`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threshold }),
  });
  state.selectedModelThreshold = threshold;
  updateThresholdInputs(state.selectedModelThreshold);
  refreshModelThresholdEditor();
  document.getElementById("detectThreshold").value = String(threshold);
  document.getElementById("appendThreshold").value = String(threshold);
  await loadModelDetail();
  await loadModels();
}

function resetSelectedModelState() {
  state.selectedModelId = null;
  state.selectedModelThreshold = null;
  state.selectedModelSampleAssetsAvailable = false;
  state.selectedModelStorageMessage = "";
  state.currentSampleDetail = null;
  state.page = 1;
  state.total = 0;
  document.getElementById("modelInfo").textContent = "请选择模型";
  document.getElementById("modelThresholdInput").value = "";
  document.getElementById("detectThreshold").value = "";
  document.getElementById("appendThreshold").value = "";
  document.getElementById("samplesTable").innerHTML = "";
  document.getElementById("pageLabel").textContent = "1 / 1";
  refreshAppendEntryState();
  refreshDetectEntryState();
  refreshSamplesEntryState();
  refreshModelTransferActions();
  refreshModelThresholdEditor();
}

async function deleteCurrentModel() {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  const modelId = state.selectedModelId;
  const confirmedModelId = window.prompt(
    `请输入模型 ID 以确认删除。\n模型 ID: ${modelId}\n该操作会删除模型目录，无法恢复。`,
    "",
  );
  if (confirmedModelId === null) {
    return;
  }
  if (confirmedModelId.trim() !== modelId) {
    alert("输入的模型 ID 不匹配，已取消删除。");
    return;
  }
  await api(`/api/models/${modelId}`, { method: "DELETE" });
  if (state.selectedModelId === modelId) {
    resetSelectedModelState();
  }
  await loadModels();
}

async function pruneCurrentModelAssets() {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  if (!state.selectedModelSampleAssetsAvailable) {
    alert("当前模型已经是精简状态，无需重复执行。");
    return;
  }
  const modelId = state.selectedModelId;
  const confirmedModelId = window.prompt(
    `请输入模型 ID 以确认精简文件。\n模型 ID: ${modelId}\n将删除 raw、processed、tiles 等样本派生文件，不影响检测和后续追加正样本，但历史向量库样本将无法查看和维护。`,
    "",
  );
  if (confirmedModelId === null) {
    return;
  }
  if (confirmedModelId.trim() !== modelId) {
    alert("输入的模型 ID 不匹配，已取消精简。");
    return;
  }
  const result = await api(`/api/models/${modelId}/prune-assets`, { method: "POST" });
  await loadModelDetail();
  document.getElementById("samplesTable").innerHTML = "";
  alert(`精简完成，删除 ${result.deleted_file_count} 个文件，释放 ${formatBytes(result.released_bytes)}。`);
  await loadModels();
}

async function importModelArchive(file) {
  setModelTransferStatus({
    visible: true,
    title: "导入模型",
    percent: 0,
    message: "准备上传模型包...",
  });
  const data = await new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("model_file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/models/import", true);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = (event.loaded / event.total) * 100;
        setModelTransferStatus({
          visible: true,
          title: "导入模型",
          percent,
          message: `正在上传模型包：${Math.round(event.loaded / 1024)}KB / ${Math.round(event.total / 1024)}KB`,
        });
      } else {
        setModelTransferStatus({
          visible: true,
          title: "导入模型",
          percent: 0,
          message: "正在上传模型包...",
        });
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        setModelTransferStatus({
          visible: true,
          title: "导入模型",
          percent: 100,
          message: "上传完成，服务器正在处理模型包...",
        });
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (error) {
          reject(error);
        }
        return;
      }
      try {
        const payload = JSON.parse(xhr.responseText);
        reject(new Error(payload.detail || "Request failed"));
      } catch (_) {
        reject(new Error(xhr.statusText || "Request failed"));
      }
    };
    xhr.onerror = () => reject(new Error("网络错误，模型导入失败"));
    xhr.send(form);
  });
  state.selectedModelId = data.model_id;
  await loadModelDetail();
  if (state.selectedModelSampleAssetsAvailable) {
    await loadSamples();
  } else {
    document.getElementById("samplesTable").innerHTML = "";
  }
  await loadModels();
  setModelTransferStatus({
    visible: true,
    title: "导入模型",
    percent: 100,
    message: "模型导入完成",
  });
  window.setTimeout(resetModelTransferStatus, 1200);
}

async function loadExportSummary() {
  const fullOption = document.getElementById("exportFullOption");
  const deployOption = document.getElementById("exportDeployOption");
  const modeHint = document.getElementById("exportModeHint");
  const fullNode = document.getElementById("exportFullSummary");
  const deployNode = document.getElementById("exportDeploySummary");
  const fullRadio = fullOption.querySelector('input[name="exportMode"][value="full"]');
  const deployRadio = deployOption.querySelector('input[name="exportMode"][value="deploy"]');
  const canExportFull = Boolean(state.selectedModelSampleAssetsAvailable);
  fullOption.classList.toggle("hidden", !canExportFull);
  deployOption.classList.remove("hidden");
  if (canExportFull) {
    modeHint.textContent = "请选择导出方式。";
    if (!fullRadio.checked && !deployRadio.checked) {
      fullRadio.checked = true;
    }
  } else {
    modeHint.textContent = "当前模型仅包含部署检测关键文件，不能导出完整模型包。若后续手动追加了正样本，完整模型包选项会重新出现。";
    deployRadio.checked = true;
  }
  fullNode.textContent = "正在读取完整模型包大小...";
  deployNode.textContent = "正在读取部署模型包大小...";
  try {
    const data = await api(`/api/models/${state.selectedModelId}/export-summary`);
    if (canExportFull) {
      fullNode.textContent =
        `包含模型目录全部文件，共 ${data.full.file_count} 个文件，约 ${formatBytes(data.full.total_size_bytes)}。可继续查看向量库样本、编辑样本、重建模型。`;
    } else {
      fullNode.textContent = "当前模型缺少完整样本文件，不显示完整模型包导出。";
    }
    deployNode.textContent =
      `仅导出部署检测关键文件，共 ${data.deploy.file_count} 个文件，约 ${formatBytes(data.deploy.total_size_bytes)}。导入后无法查看向量库样本，也不能做样本维护。`;
  } catch (error) {
    fullNode.textContent = canExportFull ? `读取大小失败：${error.message}` : "当前模型缺少完整样本文件，不显示完整模型包导出。";
    deployNode.textContent = "仅导出部署检测所需关键文件。导入后无法查看向量库样本，也不能做样本维护。";
  }
}

function getSelectedExportMode() {
  const selected = document.querySelector('input[name="exportMode"]:checked');
  return selected && selected.value === "deploy" ? "deploy" : "full";
}

async function exportCurrentModel(mode = "full") {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  closeModal("exportModal");
  const deploymentOnly = mode === "deploy";
  setModelTransferStatus({
    visible: true,
    title: deploymentOnly ? "导出部署模型" : "导出模型",
    percent: 0,
    message: "正在创建导出任务...",
  });
  const task = await api(`/api/models/${state.selectedModelId}/export?deployment_only=${deploymentOnly ? "true" : "false"}`, {
    method: "POST",
  });
  let taskStatus = null;
  while (true) {
    taskStatus = await api(`/api/model-export-tasks/${task.task_id}`);
    if (taskStatus.status === "error") {
      resetModelTransferStatus();
      throw new Error(taskStatus.error || taskStatus.message || "导出模型失败");
    }
    setModelTransferStatus({
      visible: true,
      title: deploymentOnly ? "导出部署模型" : "导出模型",
      percent: taskStatus.progress ?? 0,
      message: taskStatus.message || (deploymentOnly ? "正在压缩部署关键文件..." : "正在压缩模型目录..."),
    });
    if (taskStatus.status === "ready") {
      break;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 400));
  }

  setModelTransferStatus({
    visible: true,
    title: deploymentOnly ? "导出部署模型" : "导出模型",
    percent: 100,
    message: "压缩完成，正在下载模型包...",
  });

  const response = await fetch(`/api/model-export-tasks/${task.task_id}/download`);
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {
    }
    resetModelTransferStatus();
    throw new Error(message || "导出模型失败");
  }
  const total = Number(response.headers.get("content-length") || 0);
  const reader = response.body?.getReader();
  if (!reader) {
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = deploymentOnly ? `${state.selectedModelId}-deploy.zip` : `${state.selectedModelId}.zip`;
    link.click();
    URL.revokeObjectURL(objectUrl);
    setModelTransferStatus({
      visible: true,
      title: deploymentOnly ? "导出部署模型" : "导出模型",
      percent: 100,
      message: "模型导出完成",
    });
    window.setTimeout(resetModelTransferStatus, 1200);
    return;
  }
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    const percent = total > 0 ? (loaded / total) * 100 : 0;
    setModelTransferStatus({
      visible: true,
      title: deploymentOnly ? "导出部署模型" : "导出模型",
      percent,
      message: total > 0
        ? `正在下载模型包：${Math.round(loaded / 1024)}KB / ${Math.round(total / 1024)}KB`
        : `正在下载模型包：${Math.round(loaded / 1024)}KB`,
    });
  }
  const blob = new Blob(chunks, { type: "application/zip" });
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = (taskStatus && taskStatus.filename) ? taskStatus.filename : `${state.selectedModelId}.zip`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
  setModelTransferStatus({
    visible: true,
    title: deploymentOnly ? "导出部署模型" : "导出模型",
    percent: 100,
    message: "模型导出完成",
  });
  window.setTimeout(resetModelTransferStatus, 1200);
}

async function loadSamples() {
  if (!state.selectedModelId) return;
  const data = await api(`/api/models/${state.selectedModelId}/samples?page=${state.page}&page_size=${state.pageSize}`);
  state.total = data.total;
  const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));
  document.getElementById("pageLabel").textContent = `${data.page} / ${totalPages}`;
  const root = document.getElementById("samplesTable");
  root.innerHTML = "";
  for (const item of data.items) {
    const detail = await api(`/api/models/${state.selectedModelId}/samples/${item.sample_id}`);
    const card = document.createElement("div");
    const rawExists = detail.sample_file_status?.raw_exists !== false;
    const processedExists = detail.sample_file_status?.processed_exists !== false;
    const tilesComplete = detail.sample_file_status?.tile_images_complete !== false;
    const sampleFilesAvailable = rawExists && processedExists && tilesComplete;
    card.className = `sample-card${sampleFilesAvailable ? "" : " unavailable"}`;
    card.innerHTML = `
      <div><strong>${item.sample_id}</strong></div>
      <div>源图: ${item.source_image_name}</div>
      <div>bbox: ${JSON.stringify(item.bbox)}</div>
      <div>tile_count: ${item.tile_count ?? "-"}</div>
      <div>last_scan_score: ${item.last_scan_score ?? "-"}</div>
      <div>last_scan_is_anomaly: ${item.last_scan_is_anomaly ?? "-"}</div>
      <div>note: ${item.note || "-"}</div>
      <div>样本文件: ${sampleFilesAvailable ? "完整" : "缺失，无法查看或编辑"}</div>
      <div class="tile-thumb-grid"></div>
      <div class="sample-actions">
        <button type="button" data-edit="${item.sample_id}">编辑</button>
        <button type="button" class="danger" data-delete="${item.sample_id}">删除</button>
      </div>
    `;
    const grid = card.querySelector(".tile-thumb-grid");
    (detail.tiles || []).slice(0, 12).forEach((tile) => {
      const thumb = document.createElement("div");
      thumb.className = "tile-thumb";
      thumb.innerHTML = `
        <img src="/api/models/${state.selectedModelId}/samples/${item.sample_id}/tiles/${tile.tile_id}/image" alt="${tile.tile_id}">
        <span>${tile.tile_id.replace("tile_", "#")}</span>
      `;
      thumb.onclick = () => showTilePreview(item.sample_id, detail.sample.contour, tile);
      grid.appendChild(thumb);
    });
    card.querySelector("[data-edit]").onclick = async () => {
      if (!sampleFilesAvailable) {
        alert(detail.sample_file_status?.message || "当前样本缺少原图或子图文件，无法查看和编辑。");
        return;
      }
      document.getElementById("updateSampleId").value = item.sample_id;
      document.getElementById("updateContour").value = JSON.stringify([detail.sample.contour || []]);
      document.getElementById("updateNote").value = detail.sample.note || "";
      await state.updateEditor.loadImageFromUrl(`/api/models/${state.selectedModelId}/samples/${item.sample_id}/image?kind=raw`);
      state.updateEditor.setPolygons([detail.sample.contour || []]);
      renderTileCanvas(detail);
      openModal("editModal");
    };
    card.querySelector("[data-delete]").onclick = async () => {
      if (!confirm(`确认删除样本 ${item.sample_id} 吗？`)) return;
      await deleteSample(item.sample_id);
    };
    root.appendChild(card);
  }
}

async function deleteSample(sampleId) {
  await api(`/api/models/${state.selectedModelId}/samples/${sampleId}`, { method: "DELETE" });
  await loadModelDetail();
  await loadSamples();
}

async function scanSamples() {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  const data = await api(`/api/models/${state.selectedModelId}/scan-samples`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  alert(`扫描完成，异常样本数: ${data.flagged_count}`);
  await loadSamples();
}

async function runDetectImage() {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  const file = state.detectFile || document.getElementById("detectImageFile").files[0];
  if (!file) {
    alert("请选择图片");
    return;
  }
  state.detectFile = file;
  resetDetectResultState("检测中...");
  setDetectBusyState(true);
  try {
    const form = new FormData();
    form.append("model_id", state.selectedModelId);
    const includeHeatmap = document.getElementById("detectIncludeHeatmap").checked;
    const heatmapIncludeBackground = document.getElementById("detectHeatmapIncludeBackground").checked;
    const heatmapZeroBelowThreshold = document.getElementById("detectHeatmapZeroBelowThreshold").checked;
    const thresholdRaw = document.getElementById("detectThreshold").value.trim();
    form.append("include_heatmap_base64", includeHeatmap ? "true" : "false");
    form.append("heatmap_include_background", heatmapIncludeBackground ? "true" : "false");
    form.append("heatmap_zero_below_threshold", heatmapZeroBelowThreshold ? "true" : "false");
    if (thresholdRaw !== "") {
      form.append("threshold", thresholdRaw);
    }
    form.append("image_file", file);
    const result = await api("/api/detect", { method: "POST", body: form });
    const heatmapUrl = includeHeatmap && result.heatmap_base64 ? `data:image/jpeg;base64,${result.heatmap_base64}` : "";
    const annotatedUrl = await buildAnnotatedDetectImage(file, result);
    const anomalyCount = Array.isArray(result.anomaly_regions) ? result.anomaly_regions.length : 0;
    const scoreText = Number.isFinite(result.score) ? Number(result.score).toFixed(4) : "-";
    document.getElementById("detectHeatmapImage").src = heatmapUrl;
    document.getElementById("detectHeatmapImage").classList.toggle("hidden", !heatmapUrl);
    document.getElementById("detectAnnotatedImage").src = annotatedUrl;
    document.getElementById("detectAnnotatedImage").classList.remove("hidden");
    document.getElementById("detectResultMeta").textContent =
      `score=${scoreText} threshold=${result.threshold ?? "-"} 异常区域=${anomalyCount} 结果=${result.is_anomaly ? "异常" : "正常"}`;
  } finally {
    setDetectBusyState(false);
  }
}

async function runInferenceImage() {
  if (!state.selectedInferenceModelName) {
    alert("请先选择检测模型");
    return;
  }
  const file = state.inferFile || document.getElementById("inferImageFile").files[0];
  if (!file) {
    alert("请选择图片");
    return;
  }
  state.inferFile = file;
  resetInferenceResultState("推理中...");
  setInferenceBusyState(true);
  try {
    const form = new FormData();
    const confRaw = document.getElementById("inferConfThreshold").value.trim();
    const iouRaw = document.getElementById("inferIouThreshold").value.trim();
    const imgszRaw = document.getElementById("inferImgsz").value.trim();
    const maxDetRaw = document.getElementById("inferMaxDet").value.trim();
    const deviceRaw = document.getElementById("inferDevice").value.trim();
    const includeVisualization = document.getElementById("inferIncludeVisualization").checked;
    if (confRaw !== "") {
      form.append("conf_threshold", confRaw);
    }
    if (iouRaw !== "") {
      form.append("iou_threshold", iouRaw);
    }
    if (imgszRaw !== "") {
      form.append("imgsz", imgszRaw);
    }
    if (maxDetRaw !== "") {
      form.append("max_det", maxDetRaw);
    }
    if (deviceRaw !== "") {
      form.append("device", deviceRaw);
    }
    form.append("include_visualization_base64", includeVisualization ? "true" : "false");
    form.append("image_file", file);
    const result = await api(`/api/inference/${state.selectedInferenceModelName}`, { method: "POST", body: form });
    const originalUrl = URL.createObjectURL(file);
    const annotatedUrl = result.visualization_base64
      ? `data:image/jpeg;base64,${result.visualization_base64}`
      : await buildInferenceAnnotatedImage(file, result);
    const count = Number.isFinite(result.count) ? result.count : (Array.isArray(result.detections) ? result.detections.length : 0);
    document.getElementById("inferOriginalImage").src = originalUrl;
    document.getElementById("inferOriginalImage").classList.remove("hidden");
    document.getElementById("inferAnnotatedImage").src = annotatedUrl;
    document.getElementById("inferAnnotatedImage").classList.remove("hidden");
    document.getElementById("inferResultMeta").textContent =
      `模型=${result.model_name} 检测数=${count} conf=${result.conf_threshold} iou=${result.iou_threshold} imgsz=${result.imgsz}${result.visualization_base64 ? "" : " 前端兜底渲染"}`;
    setInferenceResultJson(result);
  } finally {
    setInferenceBusyState(false);
  }
}

async function autoExtractForAppend() {
  const file = state.appendFile || document.getElementById("appendImageFile").files[0];
  if (!file) {
    alert("请选择图片");
    return;
  }
  state.appendFile = file;
  openModal("appendModal");
  setAppendHeatmapState({ message: "YOLO 轮廓提取完成后开始异常检测..." });
  const form = new FormData();
  form.append("image_file", file);
  const data = await api("/api/extract-contours", { method: "POST", body: form });
  await state.appendEditor.loadImageFromUrl(URL.createObjectURL(file));
  state.appendEditor.setPolygonEntries((data.items || []).map((item) => ({ points: item.contour, kind: "yolo" })));
  updateAppendPreview();

  const detectForm = new FormData();
  detectForm.append("model_id", state.selectedModelId);
  detectForm.append("include_heatmap_base64", "true");
  detectForm.append(
    "heatmap_include_background",
    document.getElementById("appendHeatmapIncludeBackground").checked ? "true" : "false",
  );
  detectForm.append(
    "heatmap_zero_below_threshold",
    document.getElementById("appendHeatmapZeroBelowThreshold").checked ? "true" : "false",
  );
  const appendThresholdRaw = document.getElementById("appendThreshold").value.trim();
  if (appendThresholdRaw !== "") {
    detectForm.append("threshold", appendThresholdRaw);
  }
  detectForm.append("image_file", file);
  const detectData = await api("/api/detect", { method: "POST", body: detectForm });
  const heatmapUrl = detectData.heatmap_base64
    ? `data:image/jpeg;base64,${detectData.heatmap_base64}`
    : "";
  const anomalyCount = Array.isArray(detectData.anomaly_regions) ? detectData.anomaly_regions.length : 0;
  const scoreText = Number.isFinite(detectData.score) ? Number(detectData.score).toFixed(4) : "-";
  setAppendHeatmapState({
    message: `score=${scoreText} threshold=${detectData.threshold ?? "-"} 异常区域=${anomalyCount} 结果=${detectData.is_anomaly ? "异常" : "正常"}`,
    imageUrl: heatmapUrl,
    visible: Boolean(heatmapUrl),
  });
}

async function appendSample() {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  const file = state.appendFile || document.getElementById("appendImageFile").files[0];
  if (!file) {
    alert("请选择图片");
    return;
  }
  const polygons = state.appendEditor.getPolygons();
  if (!polygons.length) {
    alert("至少需要一个轮廓");
    return;
  }
  const form = new FormData();
  form.append("image_file", file);
  form.append("contour_json", JSON.stringify(polygons));
  const appendMaxVectors = Math.max(1, parseInt(document.getElementById("appendMaxVectors").value || "20", 10) || 20);
  form.append("append_max_vectors", String(appendMaxVectors));
  const data = await api(`/api/models/${state.selectedModelId}/samples`, { method: "POST", body: form });
  alert(`追加完成，新增 ${data.added_count} 条样本，加入 ${data.added_vector_count ?? "-"} 条向量`);
  closeModal("appendModal");
  resetAppendModalState();
  await loadModelDetail();
  await loadSamples();
}

async function saveTileStateOnly() {
  if (!state.selectedModelId || !state.currentSampleDetail) {
    alert("请先选择样本");
    return;
  }
  const enabledTileIds = (state.currentSampleDetail.tiles || []).filter((tile) => tile.enabled !== false).map((tile) => tile.tile_id);
  await api(`/api/models/${state.selectedModelId}/samples/${state.currentSampleDetail.sample.sample_id}/tiles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled_tile_ids: enabledTileIds }),
  });
  alert("子图启停已保存");
  await loadModelDetail();
  await loadSamples();
}

async function updateSample() {
  if (!state.selectedModelId) {
    alert("请先选择模型和样本");
    return;
  }
  const sampleId = document.getElementById("updateSampleId").value.trim();
  if (!sampleId) {
    alert("请先从样本列表进入编辑");
    return;
  }
  const polygons = state.updateEditor.getPolygons();
  if (!polygons.length) {
    alert("至少保留一个轮廓");
    return;
  }
  const enabledTileIds = (state.currentSampleDetail?.tiles || []).filter((tile) => tile.enabled !== false).map((tile) => tile.tile_id);
  await api(`/api/models/${state.selectedModelId}/samples/${sampleId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contour: polygons[0],
      note: document.getElementById("updateNote").value || "",
      enabled_tile_ids: enabledTileIds,
    }),
  });
  alert("样本已更新，已执行局部向量优化");
  const detail = await api(`/api/models/${state.selectedModelId}/samples/${sampleId}`);
  state.currentSampleDetail = detail;
  renderTileCanvas(detail);
  await loadModelDetail();
  await loadSamples();
}

document.getElementById("refreshModelsBtn").onclick = loadModels;
document.getElementById("refreshInferenceModelsBtn").onclick = loadInferenceModels;
document.getElementById("openStoreViewBtn").onclick = () => setActiveView("store");
document.getElementById("openInferViewBtn").onclick = () => setActiveView("infer");
document.getElementById("openSamplesBtn").onclick = async () => {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  if (!state.selectedModelSampleAssetsAvailable) {
    alert(state.selectedModelStorageMessage || "当前模型缺少样本文件，无法查看向量库样本。");
    return;
  }
  await loadSamples();
  openModal("samplesModal");
};
document.getElementById("prevPageBtn").onclick = async () => {
  state.page = Math.max(1, state.page - 1);
  await loadSamples();
};
document.getElementById("nextPageBtn").onclick = async () => {
  const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
  state.page = Math.min(totalPages, state.page + 1);
  await loadSamples();
};

document.getElementById("appendImageFile").addEventListener("change", async (event) => {
  state.appendFile = event.target.files[0];
  if (state.appendFile) {
    await autoExtractForAppend();
  }
});
document.getElementById("detectImageFile").addEventListener("change", async (event) => {
  state.detectFile = event.target.files[0];
  if (state.detectFile) {
    await runDetectImage();
  }
});
document.getElementById("inferImageFile").addEventListener("change", async (event) => {
  state.inferFile = event.target.files[0];
  if (state.inferFile) {
    await runInferenceImage();
  }
});
document.getElementById("rerunDetectBtn").onclick = runDetectImage;
document.getElementById("rerunInferenceBtn").onclick = runInferenceImage;
document.getElementById("saveModelThresholdBtn").onclick = saveModelThreshold;
document.getElementById("deleteModelBtn").onclick = deleteCurrentModel;
document.getElementById("pruneModelAssetsBtn").onclick = pruneCurrentModelAssets;
document.getElementById("exportModelBtn").onclick = () => {
  if (!state.selectedModelId) {
    alert("请先选择模型");
    return;
  }
  openModal("exportModal");
  loadExportSummary();
};
document.getElementById("confirmExportBtn").onclick = () => exportCurrentModel(getSelectedExportMode());
document.getElementById("importModelFile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  try {
    await importModelArchive(file);
    alert("模型导入成功");
  } catch (error) {
    resetModelTransferStatus();
    alert(error.message);
  } finally {
    event.target.value = "";
  }
});

document.getElementById("appendAutoExtractBtn").onclick = autoExtractForAppend;
document.getElementById("openAppendEditorBtn").onclick = () => {
  if (!state.appendEditor.image) {
    alert("请先上传图片并完成 YOLO 自动提取");
    return;
  }
  openModal("appendEditorModal");
  requestAnimationFrame(() => state.appendEditor.refreshLayout({ fit: true }));
};
document.getElementById("appendNewPolygonBtn").onclick = () => state.appendEditor.enableDrawMode();
document.getElementById("appendDeletePolygonBtn").onclick = () => state.appendEditor.deleteSelected();
document.getElementById("updateNewPolygonBtn").onclick = () => state.updateEditor.enableDrawMode();
document.getElementById("updateDeletePolygonBtn").onclick = () => state.updateEditor.deleteSelected();
document.getElementById("saveTileStateBtn").onclick = saveTileStateOnly;

document.getElementById("appendForm").onsubmit = async (event) => {
  event.preventDefault();
  await appendSample();
};

document.getElementById("updateForm").onsubmit = async (event) => {
  event.preventDefault();
  await updateSample();
};

document.querySelectorAll("[data-close-modal]").forEach((element) => {
  element.onclick = () => {
    const modalId = element.getAttribute("data-close-modal");
    closeModal(modalId);
    if (modalId === "appendModal") {
      closeModal("appendEditorModal");
      resetAppendModalState();
    } else if (modalId === "imagePreviewModal") {
      resetImagePreviewState();
    }
  };
});

document.getElementById("appendPreviewCard").addEventListener("click", () => {
  if (!state.appendEditor.image) {
    return;
  }
  openModal("appendEditorModal");
  requestAnimationFrame(() => state.appendEditor.refreshLayout({ fit: true }));
});

document.getElementById("detectHeatmapImage").addEventListener("click", (event) => {
  if (event.currentTarget.classList.contains("hidden")) {
    return;
  }
  openImagePreview(event.currentTarget.src, "热力图放大预览");
});

document.getElementById("detectAnnotatedImage").addEventListener("click", (event) => {
  if (event.currentTarget.classList.contains("hidden")) {
    return;
  }
  openImagePreview(event.currentTarget.src, "异常结果框放大预览");
});

document.getElementById("inferOriginalImage").addEventListener("click", (event) => {
  if (event.currentTarget.classList.contains("hidden")) {
    return;
  }
  openImagePreview(event.currentTarget.src, "检测原图");
});

document.getElementById("inferAnnotatedImage").addEventListener("click", (event) => {
  if (event.currentTarget.classList.contains("hidden")) {
    return;
  }
  openImagePreview(event.currentTarget.src, "检测可视化结果");
});

document.getElementById("appendHeatmapImage").addEventListener("click", (event) => {
  if (event.currentTarget.classList.contains("hidden")) {
    return;
  }
  openImagePreview(event.currentTarget.src, "异常检测热力图放大预览");
});

document.getElementById("imagePreviewImage").addEventListener("load", (event) => {
  state.imagePreview.naturalWidth = event.currentTarget.naturalWidth;
  state.imagePreview.naturalHeight = event.currentTarget.naturalHeight;
  fitImagePreviewToViewport();
});

document.getElementById("imagePreviewBody").addEventListener("wheel", (event) => {
  event.preventDefault();
  zoomImagePreview(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX, event.clientY);
}, { passive: false });

document.getElementById("imagePreviewBody").addEventListener("mousedown", (event) => {
  if (event.button !== 0 || !state.imagePreview.naturalWidth) {
    return;
  }
  event.preventDefault();
  state.imagePreview.dragging = true;
  state.imagePreview.dragStartX = event.clientX;
  state.imagePreview.dragStartY = event.clientY;
  document.getElementById("imagePreviewBody").classList.add("dragging");
});

window.addEventListener("mousemove", (event) => {
  if (!state.imagePreview.dragging) {
    return;
  }
  const deltaX = event.clientX - state.imagePreview.dragStartX;
  const deltaY = event.clientY - state.imagePreview.dragStartY;
  state.imagePreview.dragStartX = event.clientX;
  state.imagePreview.dragStartY = event.clientY;
  state.imagePreview.offsetX += deltaX;
  state.imagePreview.offsetY += deltaY;
  clampImagePreviewOffsets();
  renderImagePreview();
});

window.addEventListener("mouseup", () => {
  if (!state.imagePreview.dragging) {
    return;
  }
  state.imagePreview.dragging = false;
  document.getElementById("imagePreviewBody").classList.remove("dragging");
});

window.addEventListener("resize", () => {
  if (document.getElementById("imagePreviewModal").classList.contains("hidden")) {
    return;
  }
  fitImagePreviewToViewport();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.getElementById("imagePreviewModal").classList.contains("hidden")) {
    closeModal("imagePreviewModal");
    resetImagePreviewState();
  }
});

state.appendEditor = createAppendRectEditor("appendEditorCanvas", "appendContour", {
  onChange: () => updateAppendPreview(),
});
state.updateEditor = createPolygonEditor("updateCanvas", "updateContour");
setActiveView("store");
refreshAppendEntryState();
refreshDetectEntryState();
refreshSamplesEntryState();
refreshInferenceEntryState();
refreshModelThresholdEditor();
refreshModelTransferActions();
setAppendHeatmapState();
resetDetectResultState();
resetInferenceResultState();
setDetectBusyState(false);
setInferenceBusyState(false);

loadModels().catch((err) => {
  console.error(err);
  alert(err.message);
});
loadInferenceModels().catch((err) => {
  console.error(err);
  document.getElementById("inferResultMeta").textContent = err.message;
});
