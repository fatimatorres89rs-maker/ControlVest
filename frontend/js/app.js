import {
  FACE_CONFIDENCE_THRESHOLD,
  UNREGISTERED_STUDENT_LABEL,
  getFaceModelLabels,
  identifyFace,
  initFaceModel,
} from "./face_auth.js";

// ============================================================================
// CONFIGURACIÓN PRINCIPAL (ajusta la URL del backend o los umbrales aquí)
// ============================================================================
// URL base del servidor backend FastAPI/Flask (sin barra diagonal al final)
const DEFAULT_BACKEND_URL = "http://localhost:8000";

// Rutas locales del modelo Teachable Machine de Vestimentas
const CLOTHING_MODEL_BASE = "./model/Vestimentas/";
const CLOTHING_MODEL_URL = `${CLOTHING_MODEL_BASE}model.json`;
const CLOTHING_METADATA_URL = `${CLOTHING_MODEL_BASE}metadata.json`;
const CLOTHING_WEIGHTS_URL = `${CLOTHING_MODEL_BASE}weights.bin`;

// Umbral de confianza y duración continua de la regla anti-falsos positivos
const CLOTHING_CLEAR_THRESHOLD = 0.75; // >= 75% para considerar detección clara en pantalla
const CLOTHING_CONFIDENCE_THRESHOLD = 0.85; // >= 85% para confirmar "Incumplimiento"
const STABILITY_DURATION_MS = 3000; // 3 segundos continuos

// Si deseas fijar manualmente una etiqueta distinta a "Incumplimiento", reemplaza `null`
const MANUAL_INFRACTION_LABEL_OVERRIDE = null;

// Frecuencia de actualización del reconocimiento facial en pantalla (ms)
const FACE_PREVIEW_INTERVAL_MS = 400;

const state = {
  clothingModel: null,
  clothingLabels: [],
  clothingImageSize: 224,
  infractionLabel: "Incumplimiento",
  faceModelReady: false,
  stream: null,
  isRunning: false,
  rafId: null,
  violationStartMs: null,
  sentInCurrentWindow: false,
  isSendingPost: false,
  lastFaceEvalMs: 0,
  lastFaceResult: {
    studentName: UNREGISTERED_STUDENT_LABEL,
    rawClassName: null,
    probability: 0,
    isRegistered: false,
    predictions: [],
  },
  backendOnline: false,
  localFallbackHistory: [],
};

const dom = {
  video: document.getElementById("webcam"),
  canvas: document.getElementById("captureCanvas"),
  placeholder: document.getElementById("videoPlaceholder"),
  btnStartCam: document.getElementById("btnStartCam"),
  btnStopCam: document.getElementById("btnStopCam"),
  btnRefreshHistory: document.getElementById("btnRefreshHistory"),
  btnExportCsv: document.getElementById("btnExportCsv"),
  statusClothingModel: document.getElementById("statusClothingModel"),
  statusFaceModel: document.getElementById("statusFaceModel"),
  statusBackend: document.getElementById("statusBackend"),
  statusTensors: document.getElementById("statusTensors"),
  currentClothingState: document.getElementById("currentClothingState"),
  currentClothingConf: document.getElementById("currentClothingConf"),
  stabilityCounter: document.getElementById("stabilityCounter"),
  stabilityProgressBar: document.getElementById("stabilityProgressBar"),
  progressTrack: document.querySelector(".progress-track"),
  alertBanner: document.getElementById("alertBanner"),
  clothingBarsGrid: document.getElementById("clothingBarsGrid"),
  detectedFaceName: document.getElementById("detectedFaceName"),
  detectedFaceConf: document.getElementById("detectedFaceConf"),
  faceBarsList: document.getElementById("faceBarsList"),
  backendUrlInput: document.getElementById("backendUrlInput"),
  infractionLabelSelect: document.getElementById("infractionLabelSelect"),
  historyStatusText: document.getElementById("historyStatusText"),
  historyTableBody: document.getElementById("historyTableBody"),
};

function getBackendUrl() {
  return (dom.backendUrlInput.value || DEFAULT_BACKEND_URL).trim().replace(/\/+$/, "");
}

function setPill(el, text, variant = "neutral") {
  el.textContent = text;
  el.className = `pill pill--${variant}`;
}

/**
 * Comprueba que la cámara tenga un stream activo y un fotograma de video válido.
 */
function hasActiveVideoStream() {
  if (!state.isRunning || !state.stream) return false;
  const videoTrack = state.stream.getVideoTracks()[0];
  if (!videoTrack || videoTrack.readyState !== "live") return false;
  return (
    !dom.video.paused &&
    !dom.video.ended &&
    dom.video.readyState >= 2 &&
    dom.video.videoWidth > 0 &&
    dom.video.videoHeight > 0
  );
}

/**
 * Reinicia todas las barras, textos de estado, contador de estabilidad e identidad facial
 * a estado inactivo (0%) mientras la cámara está apagada o detenida.
 */
function resetAnalysisUI() {
  state.violationStartMs = null;
  state.sentInCurrentWindow = false;
  state.lastFaceEvalMs = 0;
  state.lastFaceResult = {
    studentName: UNREGISTERED_STUDENT_LABEL,
    rawClassName: null,
    probability: 0,
    isRegistered: false,
    predictions: [],
  };

  dom.currentClothingState.textContent = "Cámara detenida";
  dom.currentClothingConf.textContent = "0.0%";
  dom.stabilityCounter.textContent = "0.0s / 3.0s";
  dom.stabilityProgressBar.style.width = "0%";
  dom.progressTrack.setAttribute("aria-valuenow", "0");
  dom.alertBanner.textContent = "Cámara detenida / Sin análisis activo.";
  dom.alertBanner.className = "alert-banner alert-banner--idle";

  dom.detectedFaceName.textContent = UNREGISTERED_STUDENT_LABEL;
  dom.detectedFaceConf.textContent = "Cámara apagada — Sin análisis facial activo.";

  if (state.clothingLabels.length > 0) {
    renderProbabilityBars(
      dom.clothingBarsGrid,
      state.clothingLabels.map((className) => ({ className, probability: 0 })),
      state.infractionLabel,
      false
    );
  }

  const faceLabels = getFaceModelLabels();
  if (faceLabels.length > 0) {
    renderProbabilityBars(
      dom.faceBarsList,
      faceLabels.map((className) => ({ className, probability: 0 })),
      null,
      true
    );
  }
}

/**
 * Auto-detecta la etiqueta de incumplimiento desde metadata.json buscando por
 * coincidencia explícita (ej. "incumplimiento") para nunca seleccionar "Cumple_Formal".
 */
function detectInfractionLabel(labels) {
  if (MANUAL_INFRACTION_LABEL_OVERRIDE && labels.includes(MANUAL_INFRACTION_LABEL_OVERRIDE)) {
    return MANUAL_INFRACTION_LABEL_OVERRIDE;
  }

  const exactIncumplimiento = labels.find((label) =>
    label.toLowerCase().includes("incumplimiento")
  );
  if (exactIncumplimiento) {
    return exactIncumplimiento;
  }

  const fallbackMatch = labels.find((label) => {
    const clean = label
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase();
    return (
      !clean.includes("cumple_") &&
      /(incumpl|no_?cumple|infrac|falta|violat|prohib|informal)/.test(clean)
    );
  });

  return fallbackMatch || "Incumplimiento";
}

function populateInfractionSelector(labels, selectedLabel) {
  dom.infractionLabelSelect.innerHTML = "";
  for (const label of labels) {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    opt.selected = label === selectedLabel;
    dom.infractionLabelSelect.appendChild(opt);
  }
  if (labels.includes(selectedLabel)) {
    dom.infractionLabelSelect.value = selectedLabel;
  }
}

function renderProbabilityBars(container, predictions, highlightLabel = null, isFace = false) {
  container.innerHTML = "";
  const sorted = isFace
    ? [...predictions].sort((a, b) => b.probability - a.probability)
    : predictions;

  for (const item of sorted) {
    const pct = (item.probability * 100).toFixed(1);
    let fillClass = "";
    if (isFace) {
      fillClass = item.probability >= FACE_CONFIDENCE_THRESHOLD ? "bar-fill--ok" : "bar-fill--accent";
    } else if (item.className === highlightLabel) {
      fillClass = "bar-fill--danger";
    } else if (item.className.toLowerCase().includes("cumple")) {
      fillClass = "bar-fill--ok";
    }

    const card = document.createElement("div");
    card.className = "bar-card";
    card.innerHTML = `
      <div class="bar-card__row">
        <span class="bar-card__label" title="${item.className}">${item.className}</span>
        <strong>${pct}%</strong>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${fillClass}" style="width: ${pct}%"></div>
      </div>
    `;
    container.appendChild(card);
  }
}

/**
 * Carga el modelo de vestimentas desde ./model/Vestimentas/ usando tmImage.load(),
 * aplicando cache-busting para evitar archivos obsoletos al actualizar el modelo.
 */
async function initClothingModel() {
  if (!window.tmImage || typeof window.tmImage.load !== "function") {
    throw new Error("La librería @teachablemachine/image (tmImage) no está cargada.");
  }

  const cacheBuster = `?v=${Date.now()}`;

  // Revalida weights.bin en caché antes de cargar model.json
  await fetch(CLOTHING_WEIGHTS_URL, { cache: "no-store" }).catch(() => null);

  const metaResp = await fetch(`${CLOTHING_METADATA_URL}${cacheBuster}`, { cache: "no-store" });
  if (!metaResp.ok) {
    throw new Error(`No se pudo leer ${CLOTHING_METADATA_URL} (HTTP ${metaResp.status})`);
  }

  const metadata = await metaResp.json();
  state.clothingModel = await window.tmImage.load(
    `${CLOTHING_MODEL_URL}${cacheBuster}`,
    `${CLOTHING_METADATA_URL}${cacheBuster}`
  );

  state.clothingLabels = Array.isArray(metadata.labels)
    ? metadata.labels
    : state.clothingModel.getClassLabels?.() || [];
  if (typeof metadata.imageSize === "number") {
    state.clothingImageSize = metadata.imageSize;
  }

  state.infractionLabel = detectInfractionLabel(state.clothingLabels);
  populateInfractionSelector(state.clothingLabels, state.infractionLabel);

  renderProbabilityBars(
    dom.clothingBarsGrid,
    state.clothingLabels.map((className) => ({ className, probability: 0 })),
    state.infractionLabel,
    false
  );
}

/**
 * Predice la vestimenta sobre el <canvas> actual liberando todos los tensores
 * en cada ciclo mediante tf.tidy() y outputTensor.dispose().
 */
async function predictClothingFrame() {
  if (!state.clothingModel || !hasActiveVideoStream()) return [];

  if (window.tf && state.clothingModel.model) {
    let outputTensor = null;
    try {
      outputTensor = window.tf.tidy(() => {
        const pixels = window.tf.browser.fromPixels(dom.canvas);
        const resized = window.tf.image.resizeBilinear(pixels, [
          state.clothingImageSize,
          state.clothingImageSize,
        ]);
        const normalized = resized.toFloat().div(127.5).sub(1.0).expandDims(0);
        return state.clothingModel.model.predict(normalized);
      });

      const probs = await outputTensor.data();
      return state.clothingLabels.map((className, idx) => ({
        className,
        probability: Number(probs[idx] ?? 0),
      }));
    } finally {
      if (outputTensor) {
        outputTensor.dispose();
      }
      dom.statusTensors.textContent = `Tensores: ${window.tf.memory().numTensors}`;
    }
  }

  const fallbackPreds = await state.clothingModel.predict(dom.canvas);
  if (window.tf) {
    dom.statusTensors.textContent = `Tensores: ${window.tf.memory().numTensors}`;
  }
  return fallbackPreds;
}

/**
 * Filtra las probabilidades de Vestimentas:
 * - Si ninguna clase supera CLOTHING_CLEAR_THRESHOLD, retorna todas en 0.0%.
 * - Si hay detección clara, destaca únicamente la clase ganadora y deja las demás en 0.0%.
 */
function filterClothingPredictions(predictions) {
  if (!predictions || predictions.length === 0) {
    return { hasClearDetection: false, top: null, displayPredictions: [] };
  }

  const top = predictions.reduce(
    (best, current) => (current.probability > best.probability ? current : best),
    predictions[0]
  );

  const hasClearDetection = top.probability >= CLOTHING_CLEAR_THRESHOLD;
  const displayPredictions = predictions.map((item) => ({
    className: item.className,
    probability: hasClearDetection && item.className === top.className ? top.probability : 0,
  }));

  return {
    hasClearDetection,
    top: hasClearDetection ? top : null,
    displayPredictions,
  };
}

/**
 * Dibuja el frame del <video> en el <canvas> junto con un indicador HUD ligero.
 */
function drawVideoToCanvas(filteredResult) {
  const ctx = dom.canvas.getContext("2d");
  const width = dom.video.videoWidth || 640;
  const height = dom.video.videoHeight || 480;

  if (dom.canvas.width !== width || dom.canvas.height !== height) {
    dom.canvas.width = width;
    dom.canvas.height = height;
  }

  ctx.drawImage(dom.video, 0, 0, width, height);

  if (filteredResult) {
    const top = filteredResult.top;
    const isInfraction =
      top &&
      top.className === state.infractionLabel &&
      top.probability >= CLOTHING_CONFIDENCE_THRESHOLD;

    const statusLabel = top
      ? `${top.className} (${(top.probability * 100).toFixed(1)}%)`
      : "Sin detección clara";

    ctx.fillStyle = isInfraction ? "rgba(239, 68, 68, 0.88)" : "rgba(15, 23, 42, 0.78)";
    ctx.fillRect(12, 12, Math.min(width - 24, 450), 34);
    ctx.fillStyle = "#ffffff";
    ctx.font = "600 14px Segoe UI, sans-serif";
    ctx.fillText(`${state.lastFaceResult.studentName} | ${statusLabel}`, 22, 34);
  }
}

function updateFaceUI(faceResult) {
  dom.detectedFaceName.textContent = faceResult.studentName;
  if (faceResult.isRegistered) {
    dom.detectedFaceConf.textContent = `Coincidencia: ${(faceResult.probability * 100).toFixed(
      1
    )}% (umbral ${(FACE_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%)`;
  } else {
    dom.detectedFaceConf.textContent = `Sin rostro enfocado (umbral ${(
      FACE_CONFIDENCE_THRESHOLD * 100
    ).toFixed(0)}%).`;
  }

  if (faceResult.predictions.length > 0) {
    renderProbabilityBars(dom.faceBarsList, faceResult.predictions, null, true);
  }
}

/**
 * Regla de estabilidad anti-falsos positivos:
 * Solo confirma infracción si la clase de incumplimiento >= 85% durante 3 segundos continuos.
 */
async function evaluateStabilityRule(filteredResult, nowMs) {
  if (!filteredResult || !filteredResult.hasClearDetection || !filteredResult.top) {
    state.violationStartMs = null;
    state.sentInCurrentWindow = false;
    dom.currentClothingState.textContent = "Sin detección clara";
    dom.currentClothingConf.textContent = "0.0%";
    dom.stabilityCounter.textContent = "0.0s / 3.0s";
    dom.stabilityProgressBar.style.width = "0%";
    dom.progressTrack.setAttribute("aria-valuenow", "0");
    dom.alertBanner.textContent = "En espera: sin detección clara de vestimenta.";
    dom.alertBanner.className = "alert-banner alert-banner--idle";
    return;
  }

  const top = filteredResult.top;
  const infractionProb = top.className === state.infractionLabel ? top.probability : 0;

  dom.currentClothingState.textContent = top.className;
  dom.currentClothingConf.textContent = `${(top.probability * 100).toFixed(1)}%`;

  const isInfractionActive =
    top.className === state.infractionLabel && infractionProb >= CLOTHING_CONFIDENCE_THRESHOLD;

  if (!isInfractionActive) {
    state.violationStartMs = null;
    state.sentInCurrentWindow = false;
    dom.stabilityCounter.textContent = "0.0s / 3.0s";
    dom.stabilityProgressBar.style.width = "0%";
    dom.progressTrack.setAttribute("aria-valuenow", "0");

    if (top.className.toLowerCase().includes("sin")) {
      dom.alertBanner.textContent = "Sin alumno detectado en el encuadre.";
      dom.alertBanner.className = "alert-banner alert-banner--idle";
    } else if (top.className.toLowerCase().includes("cumple")) {
      dom.alertBanner.textContent = `Vestimenta conforme (${top.className}).`;
      dom.alertBanner.className = "alert-banner alert-banner--ok";
    } else {
      dom.alertBanner.textContent = `Monitoreando (${state.infractionLabel} < 85%). Temporizador reiniciado.`;
      dom.alertBanner.className = "alert-banner alert-banner--idle";
    }
    return;
  }

  if (state.violationStartMs === null) {
    state.violationStartMs = nowMs;
  }

  const elapsed = nowMs - state.violationStartMs;
  const clamped = Math.min(elapsed, STABILITY_DURATION_MS);
  const progressPct = Math.round((clamped / STABILITY_DURATION_MS) * 100);

  dom.stabilityCounter.textContent = `${(clamped / 1000).toFixed(1)}s / 3.0s`;
  dom.stabilityProgressBar.style.width = `${progressPct}%`;
  dom.progressTrack.setAttribute("aria-valuenow", String(progressPct));

  if (elapsed < STABILITY_DURATION_MS) {
    dom.alertBanner.textContent = `Posible incumplimiento detectado: validando estabilidad (${(
      clamped / 1000
    ).toFixed(1)}s / 3.0s)...`;
    dom.alertBanner.className = "alert-banner alert-banner--counting";
    return;
  }

  if (!state.sentInCurrentWindow && !state.isSendingPost) {
    state.sentInCurrentWindow = true;
    await handleConfirmedInfraction(infractionProb);
  }
}

/**
 * Al confirmarse los 3s >= 85%, identifica al alumno con face_auth.js y envía POST /api/infracciones.
 */
async function handleConfirmedInfraction(confidence) {
  if (!hasActiveVideoStream()) return;
  state.isSendingPost = true;
  try {
    const faceResult = await identifyFace(
      dom.canvas,
      FACE_CONFIDENCE_THRESHOLD,
      hasActiveVideoStream()
    );
    if (!hasActiveVideoStream()) return;
    state.lastFaceResult = faceResult;
    updateFaceUI(faceResult);

    const payload = {
      nombre_alumno: faceResult.studentName,
      estado_detectado: state.infractionLabel,
      confianza: Number(confidence.toFixed(4)),
      fecha_hora: new Date().toISOString(),
    };

    const response = await fetch(`${getBackendUrl()}/api/infracciones`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await response.json().catch(() => ({}));

    if (response.ok) {
      state.backendOnline = true;
      setPill(dom.statusBackend, "Backend: Conectado", "ok");
      dom.alertBanner.textContent = `Infracción registrada: ${payload.nombre_alumno} (${(
        confidence * 100
      ).toFixed(1)}%).`;
      dom.alertBanner.className = "alert-banner alert-banner--violation";
      await loadHistory();
    } else if (response.status === 429) {
      dom.alertBanner.textContent =
        body.mensaje || `Cooldown activo para ${payload.nombre_alumno}. Registro omitido.`;
      dom.alertBanner.className = "alert-banner alert-banner--counting";
    } else {
      throw new Error(body.detail || `HTTP ${response.status}`);
    }
  } catch (err) {
    // Manejo elegante si el backend aún no está encendido
    state.backendOnline = false;
    setPill(dom.statusBackend, "Backend: Offline", "warn");
    const fallbackItem = {
      id: `local-${state.localFallbackHistory.length + 1}`,
      nombre_alumno: state.lastFaceResult.studentName,
      estado_detectado: state.infractionLabel,
      confianza: Number(confidence.toFixed(4)),
      fecha_hora: new Date().toISOString(),
    };
    state.localFallbackHistory.unshift(fallbackItem);
    renderHistoryTable(state.localFallbackHistory);
    dom.historyStatusText.textContent =
      "Backend no disponible aún. Mostrando registros locales temporales en memoria.";
    dom.alertBanner.textContent = `Infracción confirmada (${fallbackItem.nombre_alumno}), guardada localmente (backend apagado).`;
    dom.alertBanner.className = "alert-banner alert-banner--violation";
    console.warn("[ControlVest] Backend no disponible en POST /api/infracciones:", err);
  } finally {
    state.isSendingPost = false;
  }
}

/**
 * Bucle principal de tiempo real usando requestAnimationFrame().
 * Solo ejecuta inferencias cuando existe un stream de video activo y válido.
 */
async function predictionLoop(timestamp) {
  if (!state.isRunning) return;

  if (hasActiveVideoStream()) {
    drawVideoToCanvas(null);

    const rawClothingPreds = await predictClothingFrame();
    if (!hasActiveVideoStream() || rawClothingPreds.length === 0) {
      return;
    }

    const filteredClothing = filterClothingPredictions(rawClothingPreds);
    renderProbabilityBars(
      dom.clothingBarsGrid,
      filteredClothing.displayPredictions,
      state.infractionLabel,
      false
    );

    if (state.faceModelReady && timestamp - state.lastFaceEvalMs >= FACE_PREVIEW_INTERVAL_MS) {
      state.lastFaceEvalMs = timestamp;
      const faceResult = await identifyFace(
        dom.canvas,
        FACE_CONFIDENCE_THRESHOLD,
        hasActiveVideoStream()
      );
      if (!hasActiveVideoStream()) {
        return;
      }
      state.lastFaceResult = faceResult;
      updateFaceUI(state.lastFaceResult);
    }

    drawVideoToCanvas(filteredClothing);
    await evaluateStabilityRule(filteredClothing, timestamp);
  }

  if (state.isRunning) {
    state.rafId = requestAnimationFrame(predictionLoop);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Tu navegador no soporta navigator.mediaDevices.getUserMedia().");
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    dom.video.srcObject = state.stream;
    await dom.video.play();

    dom.placeholder.classList.add("hidden");
    dom.btnStartCam.disabled = true;
    dom.btnStopCam.disabled = false;

    state.isRunning = true;
    state.lastFaceEvalMs = -FACE_PREVIEW_INTERVAL_MS;
    state.rafId = requestAnimationFrame(predictionLoop);
  } catch (err) {
    dom.alertBanner.textContent = `No se pudo iniciar la webcam: ${err.message}`;
    dom.alertBanner.className = "alert-banner alert-banner--violation";
  }
}

function stopCamera() {
  state.isRunning = false;
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  if (state.stream) {
    for (const track of state.stream.getTracks()) {
      track.stop();
    }
    state.stream = null;
  }
  dom.video.pause();
  dom.video.srcObject = null;
  dom.placeholder.classList.remove("hidden");
  dom.btnStartCam.disabled = false;
  dom.btnStopCam.disabled = true;

  const ctx = dom.canvas.getContext("2d");
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);

  resetAnalysisUI();
}

function renderHistoryTable(rows) {
  dom.historyTableBody.innerHTML = "";
  if (!rows || rows.length === 0) {
    dom.historyTableBody.innerHTML = `
      <tr><td colspan="5" class="empty-cell">No hay infracciones registradas todavía.</td></tr>
    `;
    return;
  }

  for (const row of rows) {
    const tr = document.createElement("tr");
    const conf = Number(row.confianza ?? 0);
    const confStr = conf <= 1 ? `${(conf * 100).toFixed(1)}%` : `${conf.toFixed(1)}%`;
    const dateText = row.fecha_hora ? new Date(row.fecha_hora).toLocaleString("es-PE") : "-";
    tr.innerHTML = `
      <td>${row.id ?? "-"}</td>
      <td><strong>${row.nombre_alumno}</strong></td>
      <td>${row.estado_detectado}</td>
      <td>${confStr}</td>
      <td title="${row.fecha_hora || ""}">${dateText}</td>
    `;
    dom.historyTableBody.appendChild(tr);
  }
}

async function loadHistory() {
  const url = `${getBackendUrl()}/api/infracciones`;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const json = await resp.json();
    const rows = Array.isArray(json) ? json : Array.isArray(json.data) ? json.data : [];

    state.backendOnline = true;
    setPill(dom.statusBackend, "Backend: Conectado", "ok");
    dom.historyStatusText.textContent = `Conectado a ${url} (${rows.length} registros).`;
    renderHistoryTable(rows);
    return rows;
  } catch {
    state.backendOnline = false;
    setPill(dom.statusBackend, "Backend: Offline", "warn");
    dom.historyStatusText.textContent =
      "Backend apagado o inaccesible. Se reintentará al refrescar o confirmar infracciones.";
    renderHistoryTable(state.localFallbackHistory);
    return state.localFallbackHistory;
  }
}

async function exportHistoryCsv() {
  const baseUrl = getBackendUrl();
  try {
    const resp = await fetch(`${baseUrl}/api/infracciones?format=csv`);
    if (resp.ok && resp.headers.get("content-type")?.includes("csv")) {
      const blob = await resp.blob();
      triggerCsvDownload(blob, "reporte_infracciones_controlvest.csv");
      return;
    }
  } catch {
    // Fallback a generación local en cliente si el backend aún no está encendido
  }

  const rows = await loadHistory();
  const header = ["id", "nombre_alumno", "estado_detectado", "confianza", "fecha_hora"];
  const lines = [header.join(",")];

  for (const r of rows) {
    const line = [
      r.id ?? "",
      `"${String(r.nombre_alumno ?? "").replace(/"/g, '""')}"`,
      `"${String(r.estado_detectado ?? "").replace(/"/g, '""')}"`,
      r.confianza ?? "",
      `"${String(r.fecha_hora ?? "").replace(/"/g, '""')}"`,
    ].join(",");
    lines.push(line);
  }

  const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  triggerCsvDownload(blob, "reporte_infracciones_controlvest.csv");
}

function triggerCsvDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindEvents() {
  dom.btnStartCam.addEventListener("click", startCamera);
  dom.btnStopCam.addEventListener("click", stopCamera);
  dom.btnRefreshHistory.addEventListener("click", loadHistory);
  dom.btnExportCsv.addEventListener("click", exportHistoryCsv);

  dom.backendUrlInput.addEventListener("change", loadHistory);
  dom.infractionLabelSelect.addEventListener("change", (e) => {
    state.infractionLabel = e.target.value;
    state.violationStartMs = null;
    state.sentInCurrentWindow = false;
  });
}

async function bootstrap() {
  bindEvents();

  try {
    await initClothingModel();
    setPill(
      dom.statusClothingModel,
      `Vestimentas: Listo (${state.clothingLabels.length} clases)`,
      "ok"
    );
  } catch (err) {
    console.error("[ControlVest] Error cargando modelo de Vestimentas:", err);
    setPill(dom.statusClothingModel, "Vestimentas: Error ./model/Vestimentas/", "danger");
    dom.alertBanner.textContent =
      "No se pudo cargar ./model/Vestimentas/. Sirve frontend/ mediante un servidor HTTP local.";
    dom.alertBanner.className = "alert-banner alert-banner--violation";
  }

  try {
    const faceInit = await initFaceModel();
    state.faceModelReady = true;
    setPill(dom.statusFaceModel, `Rostros: Listo (${faceInit.labels.length} clases)`, "ok");
  } catch (err) {
    console.error("[ControlVest] Error cargando modelo de Rostros:", err);
    setPill(dom.statusFaceModel, "Rostros: Error ./model/Rostros/", "danger");
  }

  resetAnalysisUI();
  await loadHistory();
}

window.addEventListener("DOMContentLoaded", bootstrap);
