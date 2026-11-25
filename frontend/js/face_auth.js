/**
 * Módulo de identificación facial para ControlVest (js/face_auth.js).
 *
 * Carga el modelo de Teachable Machine ubicado en ./model/Rostros/ usando tmImage.load()
 * (13 alumnos registrados en metadata.json), evalúa el frame actual liberando tensores
 * y devuelve el nombre del alumno con mayor probabilidad.
 * Si ninguna clase supera el umbral configurado, devuelve "Alumno No Registrado".
 */

// ============================================================================
// CONFIGURACIÓN DE IDENTIFICACIÓN FACIAL (ajustar según pruebas locales)
// ============================================================================
export const FACE_MODEL_BASE = "./model/Rostros/";
export const FACE_MODEL_URL = `${FACE_MODEL_BASE}model.json`;
export const FACE_METADATA_URL = `${FACE_MODEL_BASE}metadata.json`;
export const FACE_WEIGHTS_URL = `${FACE_MODEL_BASE}weights.bin`;

/**
 * Umbral mínimo de confianza requerido para asignar el nombre de un alumno registrado.
 * Si la probabilidad máxima es menor a este valor, se retorna "Alumno No Registrado".
 */
export const FACE_CONFIDENCE_THRESHOLD = 0.75;

export const UNREGISTERED_STUDENT_LABEL = "Alumno No Registrado";

let faceModel = null;
let faceLabels = [];

/**
 * Carga el modelo Teachable Machine de rostros desde ./model/Rostros/
 * evitando respuestas obsoletas de caché al actualizar los archivos del modelo.
 */
export async function initFaceModel() {
  if (!window.tmImage || typeof window.tmImage.load !== "function") {
    throw new Error("La librería @teachablemachine/image (tmImage) no está disponible.");
  }

  const cacheBuster = `?v=${Date.now()}`;

  // Revalida weights.bin en caché antes de cargar model.json
  await fetch(FACE_WEIGHTS_URL, { cache: "no-store" }).catch(() => null);

  const metaResp = await fetch(`${FACE_METADATA_URL}${cacheBuster}`, { cache: "no-store" });
  if (!metaResp.ok) {
    throw new Error(`No se pudo leer ${FACE_METADATA_URL} (HTTP ${metaResp.status})`);
  }

  const metadata = await metaResp.json();
  faceLabels = Array.isArray(metadata.labels) ? metadata.labels : [];

  faceModel = await window.tmImage.load(
    `${FACE_MODEL_URL}${cacheBuster}`,
    `${FACE_METADATA_URL}${cacheBuster}`
  );

  if (faceLabels.length === 0 && typeof faceModel.getClassLabels === "function") {
    faceLabels = faceModel.getClassLabels();
  }

  return {
    model: faceModel,
    labels: [...faceLabels],
  };
}

/**
 * Retorna las etiquetas cargadas desde ./model/Rostros/metadata.json.
 */
export function getFaceModelLabels() {
  return [...faceLabels];
}

/**
 * Verifica que el elemento de origen (<video> o <canvas>) contenga un fotograma
 * válido y activo antes de ejecutar inferencia sobre él.
 */
function hasValidVideoOrCanvasFrame(sourceElement) {
  if (!sourceElement) return false;

  if (typeof HTMLVideoElement !== "undefined" && sourceElement instanceof HTMLVideoElement) {
    return (
      !sourceElement.paused &&
      !sourceElement.ended &&
      sourceElement.readyState >= 2 &&
      sourceElement.videoWidth > 0 &&
      sourceElement.videoHeight > 0 &&
      Boolean(sourceElement.srcObject)
    );
  }

  if (typeof HTMLCanvasElement !== "undefined" && sourceElement instanceof HTMLCanvasElement) {
    return sourceElement.width > 0 && sourceElement.height > 0;
  }

  return true;
}

/**
 * Retorna una respuesta vacía/inactiva con todas las clases en 0%.
 */
function buildIdleFaceResult() {
  return {
    studentName: UNREGISTERED_STUDENT_LABEL,
    rawClassName: null,
    probability: 0,
    isRegistered: false,
    predictions: faceLabels.map((className) => ({ className, probability: 0 })),
  };
}

/**
 * Evalúa el frame actual (<canvas> o <video>) liberando tensores en cada ciclo
 * mediante tf.tidy() y outputTensor.dispose().
 */
export async function identifyFace(
  sourceElement,
  threshold = FACE_CONFIDENCE_THRESHOLD,
  isStreamActive = true
) {
  if (!faceModel || !isStreamActive || !hasValidVideoOrCanvasFrame(sourceElement)) {
    return buildIdleFaceResult();
  }

  let predictions = [];
  if (window.tf && faceModel.model) {
    const inputSize = faceModel._metadata?.imageSize || 224;
    let outputTensor = null;

    try {
      outputTensor = window.tf.tidy(() => {
        const pixels = window.tf.browser.fromPixels(sourceElement);
        const resized = window.tf.image.resizeBilinear(pixels, [inputSize, inputSize]);
        const normalized = resized.toFloat().div(127.5).sub(1.0).expandDims(0);
        return faceModel.model.predict(normalized);
      });

      const probs = await outputTensor.data();
      const labels =
        faceLabels.length > 0
          ? faceLabels
          : Array.from({ length: probs.length }, (_, i) => `Alumno_${i + 1}`);

      predictions = labels.map((className, index) => ({
        className,
        probability: Number(probs[index] ?? 0),
      }));
    } finally {
      if (outputTensor) {
        outputTensor.dispose();
      }
    }
  } else {
    predictions = await faceModel.predict(sourceElement);
  }

  if (!predictions || predictions.length === 0) {
    return {
      studentName: UNREGISTERED_STUDENT_LABEL,
      rawClassName: null,
      probability: 0,
      isRegistered: false,
      predictions: [],
    };
  }

  const best = predictions.reduce(
    (top, current) => (current.probability > top.probability ? current : top),
    predictions[0]
  );

  // Al no existir clase "Desconocido" en las 13 clases, cualquier rostro por debajo
  // del umbral se clasifica estrictamente como "Alumno No Registrado" y todas las
  // barras se muestran en 0.0% para evitar ruido uniforme (~7.7%).
  const isRegistered = best.probability >= threshold;
  const displayPredictions = predictions.map((item) => ({
    className: item.className,
    probability: isRegistered && item.className === best.className ? best.probability : 0,
  }));

  return {
    studentName: isRegistered ? best.className : UNREGISTERED_STUDENT_LABEL,
    rawClassName: best.className,
    probability: isRegistered ? best.probability : 0,
    isRegistered,
    predictions: displayPredictions,
  };
}
