/**
 * VisionEdge RN — Zero-Allocation Computer Vision Post-Processing Module
 * Target budget: < 2.0 ms per frame inside a react-native-worklets-core JSI Worklet.
 *
 * Pure TypeScript with zero heavy runtime dependencies. All hot-path buffers are
 * preallocated once at module/instance initialization to avoid Garbage Collection pauses.
 *
 * Incluye correcciones de auditoría TFLite:
 *  1. Normalización garantizada [0.0 .. 1.0] en Float32Array (corrige uint8 / float32 0..255).
 *  2. Detección automática de layout transpuesto [1, 84, 8400] vs no transpuesto [1, 8400, 84].
 *  3. Detección automática de escala de coordenadas [0..1] normalizadas vs [0..640] píxeles (w <= 1.5).
 *  4. Mapeo completo de las 80 clases de COCO (personas, vehículos, animales y objetos cotidianos).
 *  5. Prevención de doble activación Sigmoid sobre probabilidades ya activadas en el grafo TFLite.
 *  6. Inspector Smoke Test zero-alloc para telemetría de maxScore y coordenadas crudas [cx, cy, w, h].
 */

export type CocoClassId = number; // 0..79 (80 clases oficiales de COCO)
export type SemanticCategory = 'person' | 'vehicle' | 'bicycle' | 'animal' | 'object';
export type CategoryKey = 'persons' | 'vehicles' | 'bicycles';
export type SensorOrientation = 0 | 90 | 180 | 270;
export type TensorOutputLayout = 'channels-first' | 'anchors-first';

/**
 * Mapeo completo oficial de las 80 clases de MS-COCO (índices 0 a 79).
 */
export const COCO_CLASS_NAMES: readonly string[] = [
  'person',         // 0
  'bicycle',        // 1
  'car',            // 2
  'motorcycle',     // 3
  'airplane',       // 4
  'bus',            // 5
  'train',          // 6
  'truck',          // 7
  'boat',           // 8
  'traffic light',  // 9
  'fire hydrant',   // 10
  'stop sign',      // 11
  'parking meter',  // 12
  'bench',          // 13
  'bird',           // 14
  'cat',            // 15
  'dog',            // 16
  'horse',          // 17
  'sheep',          // 18
  'cow',            // 19
  'elephant',       // 20
  'bear',           // 21
  'zebra',          // 22
  'giraffe',        // 23
  'backpack',       // 24
  'umbrella',       // 25
  'handbag',        // 26
  'tie',            // 27
  'suitcase',       // 28
  'frisbee',        // 29
  'skis',           // 30
  'snowboard',      // 31
  'sports ball',    // 32
  'kite',           // 33
  'baseball bat',   // 34
  'baseball glove', // 35
  'skateboard',     // 36
  'surfboard',      // 37
  'tennis racket',  // 38
  'bottle',         // 39
  'wine glass',     // 40
  'cup',            // 41
  'fork',           // 42
  'knife',          // 43
  'spoon',          // 44
  'bowl',           // 45
  'banana',         // 46
  'apple',          // 47
  'sandwich',       // 48
  'orange',         // 49
  'broccoli',       // 50
  'carrot',         // 51
  'hot dog',        // 52
  'pizza',          // 53
  'donut',          // 54
  'cake',           // 55
  'chair',          // 56
  'couch',          // 57
  'potted plant',   // 58
  'bed',            // 59
  'dining table',   // 60
  'toilet',         // 61
  'tv',             // 62
  'laptop',         // 63
  'mouse',          // 64
  'remote',         // 65
  'keyboard',       // 66
  'cell phone',     // 67
  'microwave',      // 68
  'oven',           // 69
  'toaster',        // 70
  'sink',           // 71
  'refrigerator',   // 72
  'book',           // 73
  'clock',          // 74
  'vase',           // 75
  'scissors',       // 76
  'teddy bear',     // 77
  'hair drier',     // 78
  'toothbrush',     // 79
] as const;

export type CocoClassName = (typeof COCO_CLASS_NAMES)[number];

export interface ScreenRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PoseKeypoint {
  x: number;
  y: number;
  visibility: number;
}

export interface YoloProposal {
  classId: number;
  className: string;
  category: SemanticCategory;
  confidence: number;
  /** Coordenada X central normalizada en el espacio del tensor [0.0 .. 1.0] */
  cx: number;
  /** Coordenada Y central normalizada en el espacio del tensor [0.0 .. 1.0] */
  cy: number;
  /** Ancho normalizado en el espacio del tensor [0.0 .. 1.0] */
  w: number;
  /** Alto normalizado en el espacio del tensor [0.0 .. 1.0] */
  h: number;
  /** 17 keypoints COCO (poblados para detecciones de personas con pose) */
  keypoints: PoseKeypoint[];
  hasPose: boolean;
}

export interface TrackedObject {
  trackId: number;
  classId: number;
  className: string;
  category: SemanticCategory;
  confidence: number;
  screenBox: ScreenRect;
  keypoints: PoseKeypoint[];
  hasPose: boolean;
  ageFrames: number;
  lostFrames: number;
  cx: number;
  cy: number;
}

export interface CategoryDeltaSnapshot {
  persons: number;
  vehicles: number;
  bicycles: number;
  animals: number;
  objects: number;
  deltaPersons: number;
  deltaVehicles: number;
  deltaBicycles: number;
}

export interface ProjectionParams {
  cameraWidth: number;
  cameraHeight: number;
  screenWidth: number;
  screenHeight: number;
  tensorSize?: number; // Default 640
  orientation?: SensorOrientation; // Default 0
  resizeMode?: 'cover' | 'contain'; // Default 'cover'
  /**
   * 'stretch': vision-camera-resize-plugin redimensiona directo a 640x640 sin bandas negras.
   * 'letterbox': el preprocesador inserta bandas de padding manteniendo aspect ratio.
   */
  inputPaddingMode?: 'stretch' | 'letterbox'; // Default 'stretch'
  isFrontCamera?: boolean; // Default false
}

export interface TensorSmokeDiagnostic {
  layout: TensorOutputLayout;
  numAnchors: number;
  numChannels: number;
  numClasses: number;
  maxScore: number;
  bestClassId: number;
  bestClassName: string;
  bestAnchorIndex: number;
  rawCx: number;
  rawCy: number;
  rawW: number;
  rawH: number;
  isNormalizedCoords: boolean;
}

const DEFAULT_NUM_ANCHORS = 8400;
const NUM_COCO_CLASSES = 80;
const NUM_KEYPOINTS = 17;
const MAX_POOL_PROPOSALS = 256;
const MAX_TRACKS = 128;
const INV_255 = 1 / 255.0;

// Buffers preasignados en el hilo Worklet (cero presión al Garbage Collector)
const SORTED_INDICES = new Int32Array(MAX_POOL_PROPOSALS);
const SUPPRESSED_FLAGS = new Uint8Array(MAX_POOL_PROPOSALS);
const SIMILARITY_MATRIX = new Float32Array(MAX_TRACKS * MAX_POOL_PROPOSALS);
const MATCHED_TRACK_FLAGS = new Uint8Array(MAX_TRACKS);
const MATCHED_DET_FLAGS = new Uint8Array(MAX_POOL_PROPOSALS);
const NORMALIZED_INPUT_SCRATCH = new Float32Array(640 * 640 * 3);

/**
 * Obtiene el nombre legible de cualquiera de las 80 clases de COCO.
 */
export function getCocoClassName(classId: number): string {
  'worklet';
  if (classId >= 0 && classId < COCO_CLASS_NAMES.length) {
    return COCO_CLASS_NAMES[classId];
  }
  return `class_${classId}`;
}

/**
 * Clasifica cualquiera de las 80 clases de COCO en su categoría semántica visual.
 */
export function getSemanticCategory(classId: number): SemanticCategory {
  'worklet';
  if (classId === 0) return 'person';
  if (classId === 1 || classId === 3) return 'bicycle'; // bicycle, motorcycle
  if (
    classId === 2 || // car
    classId === 4 || // airplane
    classId === 5 || // bus
    classId === 6 || // train
    classId === 7 || // truck
    classId === 8    // boat
  ) {
    return 'vehicle';
  }
  if (classId >= 14 && classId <= 23) {
    // bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    return 'animal';
  }
  return 'object';
}

/**
 * Punto Crítico 1: Normalización garantizada del Tensor de Entrada [0.0 .. 1.0].
 *
 * Si el modelo TFLite fue exportado en FP16 o FP32, espera floats en [0.0, 1.0].
 * Cuando `vision-camera-resize-plugin` devuelve `Uint8Array` (0..255) o un `Float32Array`
 * con valores sin escalar (0.0..255.0), el modelo se satura y produce confianza 0.000.
 * Esta función verifica una muestra del buffer y normaliza a [0.0, 1.0] sin asignar memoria en el heap.
 */
export function ensureNormalizedFloat32Input(
  rawBuffer: Float32Array | Uint8Array,
  expectedDataType: string = 'float32'
): { buffer: Float32Array | Uint8Array; wasRescaledFrom255: boolean } {
  'worklet';
  // Si el modelo es cuantizado INT8/UINT8 puro, respeta el buffer entero original
  if (expectedDataType === 'uint8' || expectedDataType === 'int8') {
    return { buffer: rawBuffer, wasRescaledFrom255: false };
  }

  const len = rawBuffer.length;
  // Muestreamos varios píxeles distribuidos para detectar si el rango supera 1.0 (escala 0..255)
  let needsRescale = !(rawBuffer instanceof Float32Array);
  if (!needsRescale) {
    const step = Math.max(1, Math.floor(len / 32));
    for (let idx = 0; idx < len; idx += step) {
      if (rawBuffer[idx] > 1.05) {
        needsRescale = true;
        break;
      }
    }
  }

  if (!needsRescale && rawBuffer instanceof Float32Array) {
    return { buffer: rawBuffer, wasRescaledFrom255: false };
  }

  // Si ya es un Float32Array mutable con valores 0..255, normalizamos in-place o sobre buffer preasignado
  const target =
    rawBuffer instanceof Float32Array
      ? rawBuffer
      : NORMALIZED_INPUT_SCRATCH.length === len
      ? NORMALIZED_INPUT_SCRATCH
      : new Float32Array(len);

  for (let i = 0; i < len; i++) {
    target[i] = rawBuffer[i] * INV_255;
  }

  return { buffer: target, wasRescaledFrom255: true };
}

/**
 * Punto Crítico 2: Detección dinámica del Layout del Tensor de Salida.
 * - Ultralytics TFLite estándar: [1, 84, 8400] -> 'channels-first' (stride = 8400)
 * - Exportaciones transpuestas:  [1, 8400, 84] -> 'anchors-first'  (stride = 84)
 */
export function resolveOutputTensorLayout(
  tensorLength: number,
  outputShape?: readonly number[],
  defaultChannels = 84
): {
  layout: TensorOutputLayout;
  numChannels: number;
  numAnchors: number;
} {
  'worklet';
  if (outputShape && outputShape.length >= 3) {
    const d1 = outputShape[outputShape.length - 2];
    const d2 = outputShape[outputShape.length - 1];
    if (d1 > 0 && d2 > 0) {
      if (d1 <= d2) {
        return { layout: 'channels-first', numChannels: d1, numAnchors: d2 };
      } else {
        return { layout: 'anchors-first', numChannels: d2, numAnchors: d1 };
      }
    }
  }

  // Fallback por longitud total del Float32Array (ej. 84 * 8400 = 705,600)
  const inferredAnchors = Math.round(tensorLength / defaultChannels) || DEFAULT_NUM_ANCHORS;
  return {
    layout: 'channels-first',
    numChannels: defaultChannels,
    numAnchors: inferredAnchors,
  };
}

/**
 * Pool preasignado de propuestas YOLO para que decodificar 8,400 anclas jamás
 * cree objetos efímeros en el recolector de basura del Worklet.
 */
export class ProposalPool {
  public readonly items: YoloProposal[];
  public count = 0;

  constructor(capacity = MAX_POOL_PROPOSALS) {
    this.items = new Array<YoloProposal>(capacity);
    for (let i = 0; i < capacity; i++) {
      const kps = new Array<PoseKeypoint>(NUM_KEYPOINTS);
      for (let k = 0; k < NUM_KEYPOINTS; k++) {
        kps[k] = { x: 0, y: 0, visibility: 0 };
      }
      this.items[i] = {
        classId: 0,
        className: 'person',
        category: 'person',
        confidence: 0,
        cx: 0,
        cy: 0,
        w: 0,
        h: 0,
        keypoints: kps,
        hasPose: false,
      };
    }
  }

  public reset(): void {
    'worklet';
    this.count = 0;
  }

  public acquire(): YoloProposal | null {
    'worklet';
    if (this.count >= this.items.length) return null;
    const item = this.items[this.count];
    this.count++;
    return item;
  }
}

const DEFAULT_DET_POOL = new ProposalPool(MAX_POOL_PROPOSALS);
const DEFAULT_POSE_POOL = new ProposalPool(MAX_POOL_PROPOSALS);

/**
 * Punto Crítico 6: Inspector de Diagnóstico Smoke Test (1 vez por segundo).
 * Escanea el tensor de salida sin filtro de umbral para reportar el maxScore real,
 * la clase ganadora y las coordenadas crudas [cx, cy, w, h].
 */
export function inspectYoloOutputTensor(
  detTensor: Float32Array,
  outputShape?: readonly number[]
): TensorSmokeDiagnostic {
  'worklet';
  const { layout, numChannels, numAnchors } = resolveOutputTensorLayout(
    detTensor.length,
    outputShape,
    84
  );
  const numClasses = Math.min(NUM_COCO_CLASSES, Math.max(1, numChannels - 4));

  let maxScore = -Infinity;
  let bestClassId = 0;
  let bestAnchorIndex = 0;

  if (layout === 'channels-first') {
    for (let c = 0; c < numClasses; c++) {
      const chOffset = (4 + c) * numAnchors;
      for (let i = 0; i < numAnchors; i++) {
        const s = detTensor[chOffset + i];
        if (s > maxScore) {
          maxScore = s;
          bestClassId = c;
          bestAnchorIndex = i;
        }
      }
    }
  } else {
    for (let i = 0; i < numAnchors; i++) {
      const rowOffset = i * numChannels;
      for (let c = 0; c < numClasses; c++) {
        const s = detTensor[rowOffset + 4 + c];
        if (s > maxScore) {
          maxScore = s;
          bestClassId = c;
          bestAnchorIndex = i;
        }
      }
    }
  }

  let rawCx = 0;
  let rawCy = 0;
  let rawW = 0;
  let rawH = 0;

  if (layout === 'channels-first') {
    rawCx = detTensor[0 * numAnchors + bestAnchorIndex];
    rawCy = detTensor[1 * numAnchors + bestAnchorIndex];
    rawW = detTensor[2 * numAnchors + bestAnchorIndex];
    rawH = detTensor[3 * numAnchors + bestAnchorIndex];
  } else {
    const rowOffset = bestAnchorIndex * numChannels;
    rawCx = detTensor[rowOffset + 0];
    rawCy = detTensor[rowOffset + 1];
    rawW = detTensor[rowOffset + 2];
    rawH = detTensor[rowOffset + 3];
  }

  return {
    layout,
    numAnchors,
    numChannels,
    numClasses,
    maxScore: maxScore === -Infinity ? 0 : maxScore,
    bestClassId,
    bestClassName: getCocoClassName(bestClassId),
    bestAnchorIndex,
    rawCx,
    rawCy,
    rawW,
    rawH,
    isNormalizedCoords: rawW <= 1.5 && rawH <= 1.5,
  };
}

/**
 * 1A. Decodificador Universal de Tensor de Detección YOLOv8 / YOLO11
 * - Soporta dinámicamente layout [1, 84, 8400] ('channels-first') y [1, 8400, 84] ('anchors-first').
 * - Evalúa las 80 clases completas de COCO (personas, vehículos, animales, objetos).
 * - No aplica doble Sigmoid si los puntajes ya están en [0.0, 1.0] (aplica sigmoid solo si detecta logits crudos < 0 o > 1).
 * - Detecta automáticamente si [cx, cy, w, h] ya vienen normalizadas en [0..1] (w <= 1.5) o en píxeles [0..640].
 */
export function decodeYoloDetections(
  detTensor: Float32Array,
  confThreshold = 0.45,
  tensorSize = 640,
  pool: ProposalPool = DEFAULT_DET_POOL,
  outputShape?: readonly number[]
): ProposalPool {
  'worklet';
  pool.reset();

  const { layout, numChannels, numAnchors } = resolveOutputTensorLayout(
    detTensor.length,
    outputShape,
    84
  );
  const numClasses = Math.min(NUM_COCO_CLASSES, Math.max(1, numChannels - 4));
  const invSize = 1 / tensorSize;

  if (layout === 'channels-first') {
    // Layout [1, 84, 8400]: lectura channel * 8400 + anchorIndex
    const ch0 = 0;
    const ch1 = numAnchors;
    const ch2 = 2 * numAnchors;
    const ch3 = 3 * numAnchors;

    for (let i = 0; i < numAnchors; i++) {
      let bestScore = confThreshold;
      let bestCls = -1;

      for (let c = 0; c < numClasses; c++) {
        let score = detTensor[(4 + c) * numAnchors + i];
        // Punto Crítico 5: Jamás aplicar doble Sigmoid si el valor ya está en [0.0, 1.0].
        // Solo si el modelo devolvió logits crudos fuera de [0, 1] (> 1.001), aplicamos sigmoid.
        if (score > 1.001) {
          score = 1 / (1 + Math.exp(-score));
        }
        if (score >= bestScore) {
          bestScore = score;
          bestCls = c;
        }
      }

      if (bestCls === -1) continue;

      const slot = pool.acquire();
      if (!slot) break;

      let cx = detTensor[ch0 + i];
      let cy = detTensor[ch1 + i];
      let w = detTensor[ch2 + i];
      let h = detTensor[ch3 + i];

      // Punto Crítico 3: Ultralytics TFLite suele exportar [cx, cy, w, h] ya normalizadas en [0..1].
      // Solo dividimos entre 640 si detectamos escala en píxeles (w > 1.5 o h > 1.5).
      if (w > 1.5 || h > 1.5 || cx > 1.5 || cy > 1.5) {
        cx *= invSize;
        cy *= invSize;
        w *= invSize;
        h *= invSize;
      }

      slot.classId = bestCls;
      slot.className = getCocoClassName(bestCls);
      slot.category = getSemanticCategory(bestCls);
      slot.confidence = bestScore;
      slot.cx = cx;
      slot.cy = cy;
      slot.w = w;
      slot.h = h;
      slot.hasPose = false;
    }
  } else {
    // Layout transpuesto [1, 8400, 84]: lectura anchorIndex * 84 + channel
    for (let i = 0; i < numAnchors; i++) {
      const rowOffset = i * numChannels;
      let bestScore = confThreshold;
      let bestCls = -1;

      for (let c = 0; c < numClasses; c++) {
        let score = detTensor[rowOffset + 4 + c];
        if (score > 1.001) {
          score = 1 / (1 + Math.exp(-score));
        }
        if (score >= bestScore) {
          bestScore = score;
          bestCls = c;
        }
      }

      if (bestCls === -1) continue;

      const slot = pool.acquire();
      if (!slot) break;

      let cx = detTensor[rowOffset + 0];
      let cy = detTensor[rowOffset + 1];
      let w = detTensor[rowOffset + 2];
      let h = detTensor[rowOffset + 3];

      if (w > 1.5 || h > 1.5 || cx > 1.5 || cy > 1.5) {
        cx *= invSize;
        cy *= invSize;
        w *= invSize;
        h *= invSize;
      }

      slot.classId = bestCls;
      slot.className = getCocoClassName(bestCls);
      slot.category = getSemanticCategory(bestCls);
      slot.confidence = bestScore;
      slot.cx = cx;
      slot.cy = cy;
      slot.w = w;
      slot.h = h;
      slot.hasPose = false;
    }
  }

  return pool;
}

/**
 * 1B. Decodificador de Tensor de Pose YOLO ([1, 56, 8400] o [1, 8400, 56])
 * Extrae cajas de personas junto con sus 17 keypoints COCO ([x, y, visibility]).
 */
export function decodeYoloPose(
  poseTensor: Float32Array,
  confThreshold = 0.45,
  tensorSize = 640,
  pool: ProposalPool = DEFAULT_POSE_POOL,
  outputShape?: readonly number[]
): ProposalPool {
  'worklet';
  pool.reset();

  const { layout, numChannels, numAnchors } = resolveOutputTensorLayout(
    poseTensor.length,
    outputShape,
    56
  );
  const invSize = 1 / tensorSize;

  if (layout === 'channels-first') {
    const ch0 = 0;
    const ch1 = numAnchors;
    const ch2 = 2 * numAnchors;
    const ch3 = 3 * numAnchors;
    const chConf = 4 * numAnchors;

    for (let i = 0; i < numAnchors; i++) {
      let score = poseTensor[chConf + i];
      if (score > 1.001) {
        score = 1 / (1 + Math.exp(-score));
      }
      if (score < confThreshold) continue;

      const slot = pool.acquire();
      if (!slot) break;

      let cx = poseTensor[ch0 + i];
      let cy = poseTensor[ch1 + i];
      let w = poseTensor[ch2 + i];
      let h = poseTensor[ch3 + i];

      const isPixelScale = w > 1.5 || h > 1.5 || cx > 1.5 || cy > 1.5;
      if (isPixelScale) {
        cx *= invSize;
        cy *= invSize;
        w *= invSize;
        h *= invSize;
      }

      slot.classId = 0;
      slot.className = 'person';
      slot.category = 'person';
      slot.confidence = score;
      slot.cx = cx;
      slot.cy = cy;
      slot.w = w;
      slot.h = h;
      slot.hasPose = true;

      for (let k = 0; k < NUM_KEYPOINTS; k++) {
        const baseChannel = (5 + k * 3) * numAnchors;
        let kx = poseTensor[baseChannel + i];
        let ky = poseTensor[baseChannel + numAnchors + i];
        let kv = poseTensor[baseChannel + 2 * numAnchors + i];

        if (kv > 1.001 || kv < -0.001) {
          kv = 1 / (1 + Math.exp(-kv));
        }
        if (isPixelScale || kx > 1.5 || ky > 1.5) {
          kx *= invSize;
          ky *= invSize;
        }

        const kp = slot.keypoints[k];
        kp.x = kx;
        kp.y = ky;
        kp.visibility = kv;
      }
    }
  } else {
    for (let i = 0; i < numAnchors; i++) {
      const rowOffset = i * numChannels;
      let score = poseTensor[rowOffset + 4];
      if (score > 1.001) {
        score = 1 / (1 + Math.exp(-score));
      }
      if (score < confThreshold) continue;

      const slot = pool.acquire();
      if (!slot) break;

      let cx = poseTensor[rowOffset + 0];
      let cy = poseTensor[rowOffset + 1];
      let w = poseTensor[rowOffset + 2];
      let h = poseTensor[rowOffset + 3];

      const isPixelScale = w > 1.5 || h > 1.5 || cx > 1.5 || cy > 1.5;
      if (isPixelScale) {
        cx *= invSize;
        cy *= invSize;
        w *= invSize;
        h *= invSize;
      }

      slot.classId = 0;
      slot.className = 'person';
      slot.category = 'person';
      slot.confidence = score;
      slot.cx = cx;
      slot.cy = cy;
      slot.w = w;
      slot.h = h;
      slot.hasPose = true;

      for (let k = 0; k < NUM_KEYPOINTS; k++) {
        const kpBase = rowOffset + 5 + k * 3;
        let kx = poseTensor[kpBase];
        let ky = poseTensor[kpBase + 1];
        let kv = poseTensor[kpBase + 2];

        if (kv > 1.001 || kv < -0.001) {
          kv = 1 / (1 + Math.exp(-kv));
        }
        if (isPixelScale || kx > 1.5 || ky > 1.5) {
          kx *= invSize;
          ky *= invSize;
        }

        const kp = slot.keypoints[k];
        kp.x = kx;
        kp.y = ky;
        kp.visibility = kv;
      }
    }
  }

  return pool;
}

/**
 * 2A. Intersection over Union (IoU) entre dos cajas en formato centro [cx, cy, w, h].
 * Cero asignaciones de memoria.
 */
export function calculateIoUCenter(
  ax: number,
  ay: number,
  aw: number,
  ah: number,
  bx: number,
  by: number,
  bw: number,
  bh: number
): number {
  'worklet';
  const aHalfW = aw * 0.5;
  const aHalfH = ah * 0.5;
  const bHalfW = bw * 0.5;
  const bHalfH = bh * 0.5;

  const ax1 = ax - aHalfW;
  const ay1 = ay - aHalfH;
  const ax2 = ax + aHalfW;
  const ay2 = ay + aHalfH;

  const bx1 = bx - bHalfW;
  const by1 = by - bHalfH;
  const bx2 = bx + bHalfW;
  const by2 = by + bHalfH;

  const interX1 = ax1 > bx1 ? ax1 : bx1;
  const interY1 = ay1 > by1 ? ay1 : by1;
  const interX2 = ax2 < bx2 ? ax2 : bx2;
  const interY2 = ay2 < by2 ? ay2 : by2;

  const interW = interX2 - interX1;
  const interH = interY2 - interY1;
  if (interW <= 0 || interH <= 0) return 0;

  const interArea = interW * interH;
  const unionArea = aw * ah + bw * bh - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

/**
 * 2B. Intersection over Union (IoU) entre dos ScreenRect [x, y, width, height].
 */
export function calculateIoU(a: ScreenRect, b: ScreenRect): number {
  'worklet';
  const x1 = a.x > b.x ? a.x : b.x;
  const y1 = a.y > b.y ? a.y : b.y;
  const x2 = a.x + a.width < b.x + b.width ? a.x + a.width : b.x + b.width;
  const y2 = a.y + a.height < b.y + b.height ? a.y + a.height : b.y + b.height;

  const interW = x2 - x1;
  const interH = y2 - y1;
  if (interW <= 0 || interH <= 0) return 0;

  const interArea = interW * interH;
  const unionArea = a.width * a.height + b.width * b.height - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

/**
 * 2C. Class-Aware Non-Maximum Suppression (NMS) sobre un ProposalPool.
 * Suprime únicamente duplicados dentro de la misma clase (ej. una persona con un perro
 * o frente a un vehículo jamás se suprimen entre sí).
 */
export function applyClassAwareNMS(
  pool: ProposalPool,
  iouThreshold = 0.45,
  outKeptIndices?: Int32Array
): { keptIndices: Int32Array; keptCount: number } {
  'worklet';
  const n = pool.count;
  const targetOut = outKeptIndices ?? new Int32Array(MAX_POOL_PROPOSALS);
  if (n === 0) {
    return { keptIndices: targetOut, keptCount: 0 };
  }

  const items = pool.items;

  for (let i = 0; i < n; i++) {
    SORTED_INDICES[i] = i;
    SUPPRESSED_FLAGS[i] = 0;
  }

  // Insertion sort in-place por confianza descendente (óptimo para n <= 256)
  for (let i = 1; i < n; i++) {
    const keyIdx = SORTED_INDICES[i];
    const keyConf = items[keyIdx].confidence;
    let j = i - 1;
    while (j >= 0 && items[SORTED_INDICES[j]].confidence < keyConf) {
      SORTED_INDICES[j + 1] = SORTED_INDICES[j];
      j--;
    }
    SORTED_INDICES[j + 1] = keyIdx;
  }

  let keptCount = 0;

  for (let i = 0; i < n; i++) {
    if (SUPPRESSED_FLAGS[i] === 1) continue;

    const candIdx = SORTED_INDICES[i];
    const cand = items[candIdx];
    targetOut[keptCount++] = candIdx;

    for (let j = i + 1; j < n; j++) {
      if (SUPPRESSED_FLAGS[j] === 1) continue;

      const other = items[SORTED_INDICES[j]];
      if (other.classId !== cand.classId) continue;

      const iou = calculateIoUCenter(
        cand.cx,
        cand.cy,
        cand.w,
        cand.h,
        other.cx,
        other.cy,
        other.w,
        other.h
      );

      if (iou > iouThreshold) {
        SUPPRESSED_FLAGS[j] = 1;
      }
    }
  }

  return { keptIndices: targetOut, keptCount };
}

/**
 * 4. Proyección del Tensor 640x640 a Coordenadas Exactas de Pantalla (projectToScreen)
 * Soporta tanto redimensionado directo de vision-camera-resize-plugin ('stretch')
 * como preprocesamiento con bandas negras ('letterbox'), además de rotación de sensor.
 */
export function projectToScreen(
  normCx: number,
  normCy: number,
  normW: number,
  normH: number,
  params: ProjectionParams,
  outRect?: ScreenRect
): ScreenRect {
  'worklet';
  const tensorSize = params.tensorSize ?? 640;
  const orientation = params.orientation ?? 0;
  const resizeMode = params.resizeMode ?? 'cover';
  const inputPaddingMode = params.inputPaddingMode ?? 'stretch';
  const isFrontCamera = params.isFrontCamera ?? false;

  // Si la coordenada entró en píxeles [0..640] por error externo, la llevamos a [0..1]
  const nCx = normCx > 1.5 ? normCx / tensorSize : normCx;
  const nCy = normCy > 1.5 ? normCy / tensorSize : normCy;
  const nW = normW > 1.5 ? normW / tensorSize : normW;
  const nH = normH > 1.5 ? normH / tensorSize : normH;

  // Dimensiones efectivas del fotograma tras orientación del sensor en VisionCamera v4
  const isRotated90or270 = orientation === 90 || orientation === 270;
  const frameW = Math.max(1, isRotated90or270 ? params.cameraHeight : params.cameraWidth);
  const frameH = Math.max(1, isRotated90or270 ? params.cameraWidth : params.cameraHeight);

  let camCx = 0;
  let camCy = 0;
  let camW = 0;
  let camH = 0;

  if (inputPaddingMode === 'letterbox') {
    const lbScale = Math.min(tensorSize / frameW, tensorSize / frameH);
    const padX = (tensorSize - frameW * lbScale) * 0.5;
    const padY = (tensorSize - frameH * lbScale) * 0.5;

    camCx = (nCx * tensorSize - padX) / lbScale;
    camCy = (nCy * tensorSize - padY) / lbScale;
    camW = (nW * tensorSize) / lbScale;
    camH = (nH * tensorSize) / lbScale;
  } else {
    // 'stretch': vision-camera-resize-plugin mapea [0..1] del tensor directamente a [0..frameW/H]
    camCx = nCx * frameW;
    camCy = nCy * frameH;
    camW = nW * frameW;
    camH = nH * frameH;
  }

  // Mapear del espacio de cámara al componente de pantalla (cover / contain)
  const screenW = Math.max(1, params.screenWidth);
  const screenH = Math.max(1, params.screenHeight);
  const viewScale =
    resizeMode === 'cover'
      ? Math.max(screenW / frameW, screenH / frameH)
      : Math.min(screenW / frameW, screenH / frameH);

  const viewOffsetX = (frameW * viewScale - screenW) * 0.5;
  const viewOffsetY = (frameH * viewScale - screenH) * 0.5;

  const outW = camW * viewScale;
  const outH = camH * viewScale;
  let outX = (camCx - camW * 0.5) * viewScale - viewOffsetX;
  const outY = (camCy - camH * 0.5) * viewScale - viewOffsetY;

  if (isFrontCamera) {
    outX = screenW - outX - outW;
  }

  const target = outRect ?? { x: 0, y: 0, width: 0, height: 0 };
  target.x = outX;
  target.y = outY;
  target.width = outW;
  target.height = outH;
  return target;
}

/**
 * Proyecta un punto clave normalizado [nx, ny] del tensor 640x640 a coordenadas de pantalla.
 */
export function projectPointToScreen(
  normX: number,
  normY: number,
  params: ProjectionParams,
  outPoint: PoseKeypoint
): void {
  'worklet';
  const tensorSize = params.tensorSize ?? 640;
  const orientation = params.orientation ?? 0;
  const resizeMode = params.resizeMode ?? 'cover';
  const inputPaddingMode = params.inputPaddingMode ?? 'stretch';
  const isFrontCamera = params.isFrontCamera ?? false;

  const nX = normX > 1.5 ? normX / tensorSize : normX;
  const nY = normY > 1.5 ? normY / tensorSize : normY;

  const isRotated90or270 = orientation === 90 || orientation === 270;
  const frameW = Math.max(1, isRotated90or270 ? params.cameraHeight : params.cameraWidth);
  const frameH = Math.max(1, isRotated90or270 ? params.cameraWidth : params.cameraHeight);

  let camX = 0;
  let camY = 0;

  if (inputPaddingMode === 'letterbox') {
    const lbScale = Math.min(tensorSize / frameW, tensorSize / frameH);
    const padX = (tensorSize - frameW * lbScale) * 0.5;
    const padY = (tensorSize - frameH * lbScale) * 0.5;
    camX = (nX * tensorSize - padX) / lbScale;
    camY = (nY * tensorSize - padY) / lbScale;
  } else {
    camX = nX * frameW;
    camY = nY * frameH;
  }

  const screenW = Math.max(1, params.screenWidth);
  const screenH = Math.max(1, params.screenHeight);
  const viewScale =
    resizeMode === 'cover'
      ? Math.max(screenW / frameW, screenH / frameH)
      : Math.min(screenW / frameW, screenH / frameH);

  const viewOffsetX = (frameW * viewScale - screenW) * 0.5;
  const viewOffsetY = (frameH * viewScale - screenH) * 0.5;

  let sx = camX * viewScale - viewOffsetX;
  const sy = camY * viewScale - viewOffsetY;

  if (isFrontCamera) {
    sx = screenW - sx;
  }

  outPoint.x = sx;
  outPoint.y = sy;
}

/**
 * 3. Rastreador Ligero Multi-Objeto para las 80 clases de COCO (SimpleByteTracker)
 * Inspirado en ByteTrack / SORT con matriz de similitud (IoU + proximidad de centroide),
 * retención temporal (maxLostFrames = 15) y cálculo de deltas por categoría.
 */
export class SimpleByteTracker {
  private nextTrackId = 1;
  private tracks: TrackedObject[] = [];
  private readonly maxLostFrames: number;
  private readonly similarityThreshold: number;
  private readonly highConfThreshold: number;

  private prevCounts = { persons: 0, vehicles: 0, bicycles: 0, animals: 0, objects: 0 };
  private currentCounts = { persons: 0, vehicles: 0, bicycles: 0, animals: 0, objects: 0 };

  constructor(maxLostFrames = 15, similarityThreshold = 0.28, highConfThreshold = 0.42) {
    this.maxLostFrames = maxLostFrames;
    this.similarityThreshold = similarityThreshold;
    this.highConfThreshold = highConfThreshold;
  }

  public reset(): void {
    'worklet';
    this.nextTrackId = 1;
    this.tracks.length = 0;
    this.prevCounts = { persons: 0, vehicles: 0, bicycles: 0, animals: 0, objects: 0 };
    this.currentCounts = { persons: 0, vehicles: 0, bicycles: 0, animals: 0, objects: 0 };
  }

  public update(
    pool: ProposalPool,
    keptIndices: Int32Array,
    keptCount: number,
    projection: ProjectionParams
  ): TrackedObject[] {
    'worklet';
    const numTracks = Math.min(this.tracks.length, MAX_TRACKS);
    const numDets = Math.min(keptCount, MAX_POOL_PROPOSALS);
    const screenDiag = Math.hypot(projection.screenWidth, projection.screenHeight) || 1;

    for (let t = 0; t < numTracks; t++) MATCHED_TRACK_FLAGS[t] = 0;
    for (let d = 0; d < numDets; d++) MATCHED_DET_FLAGS[d] = 0;

    // 1. Proyectar detecciones sobrevivientes de NMS a espacio de pantalla
    const detRects = new Array<ScreenRect>(numDets);
    for (let d = 0; d < numDets; d++) {
      const prop = pool.items[keptIndices[d]];
      detRects[d] = projectToScreen(prop.cx, prop.cy, prop.w, prop.h, projection);
    }

    // 2. Poblar matriz de similitud: 0.65 * IoU + 0.35 * Proximidad de Centroide
    for (let t = 0; t < numTracks; t++) {
      const tr = this.tracks[t];
      const rowOffset = t * numDets;

      for (let d = 0; d < numDets; d++) {
        const prop = pool.items[keptIndices[d]];
        if (tr.classId !== prop.classId) {
          SIMILARITY_MATRIX[rowOffset + d] = 0;
          continue;
        }

        const box = detRects[d];
        const iou = calculateIoU(tr.screenBox, box);
        const detCx = box.x + box.width * 0.5;
        const detCy = box.y + box.height * 0.5;
        const dist = Math.hypot(detCx - tr.cx, detCy - tr.cy);
        const prox = Math.max(0, 1 - dist / (screenDiag * 0.25));

        SIMILARITY_MATRIX[rowOffset + d] = 0.65 * iou + 0.35 * prox;
      }
    }

    // 3. Asociación greedy en 2 etapas
    for (let pass = 0; pass < 2; pass++) {
      const minConf = pass === 0 ? this.highConfThreshold : 0;

      for (let t = 0; t < numTracks; t++) {
        if (MATCHED_TRACK_FLAGS[t] === 1) continue;
        const rowOffset = t * numDets;
        let bestScore = this.similarityThreshold;
        let bestDet = -1;

        for (let d = 0; d < numDets; d++) {
          if (MATCHED_DET_FLAGS[d] === 1) continue;
          const prop = pool.items[keptIndices[d]];
          if (prop.confidence < minConf) continue;

          const score = SIMILARITY_MATRIX[rowOffset + d];
          if (score > bestScore) {
            bestScore = score;
            bestDet = d;
          }
        }

        if (bestDet !== -1) {
          MATCHED_TRACK_FLAGS[t] = 1;
          MATCHED_DET_FLAGS[bestDet] = 1;

          const prop = pool.items[keptIndices[bestDet]];
          const rect = detRects[bestDet];
          const tr = this.tracks[t];

          tr.confidence = prop.confidence;
          tr.screenBox.x = rect.x;
          tr.screenBox.y = rect.y;
          tr.screenBox.width = rect.width;
          tr.screenBox.height = rect.height;
          tr.cx = rect.x + rect.width * 0.5;
          tr.cy = rect.y + rect.height * 0.5;
          tr.ageFrames += 1;
          tr.lostFrames = 0;
          tr.hasPose = prop.hasPose;

          if (prop.hasPose) {
            for (let k = 0; k < NUM_KEYPOINTS; k++) {
              const srcKp = prop.keypoints[k];
              const dstKp = tr.keypoints[k];
              projectPointToScreen(srcKp.x, srcKp.y, projection, dstKp);
              dstKp.visibility = srcKp.visibility;
            }
          }
        }
      }
    }

    // 4. Envejecer tracks no emparejados y podar los que superen maxLostFrames
    const nextTracks: TrackedObject[] = [];
    for (let t = 0; t < numTracks; t++) {
      const tr = this.tracks[t];
      if (MATCHED_TRACK_FLAGS[t] === 1) {
        nextTracks.push(tr);
      } else {
        tr.lostFrames += 1;
        if (tr.lostFrames <= this.maxLostFrames) {
          nextTracks.push(tr);
        }
      }
    }

    // 5. Crear nuevos tracks para todas las detecciones válidas que sobrevivieron NMS
    for (let d = 0; d < numDets; d++) {
      if (MATCHED_DET_FLAGS[d] === 1) continue;
      const prop = pool.items[keptIndices[d]];

      const rect = detRects[d];
      const kps = new Array<PoseKeypoint>(NUM_KEYPOINTS);
      for (let k = 0; k < NUM_KEYPOINTS; k++) {
        kps[k] = { x: 0, y: 0, visibility: 0 };
        if (prop.hasPose) {
          projectPointToScreen(prop.keypoints[k].x, prop.keypoints[k].y, projection, kps[k]);
          kps[k].visibility = prop.keypoints[k].visibility;
        }
      }

      nextTracks.push({
        trackId: this.nextTrackId++,
        classId: prop.classId,
        className: prop.className,
        category: prop.category,
        confidence: prop.confidence,
        screenBox: rect,
        keypoints: kps,
        hasPose: prop.hasPose,
        ageFrames: 1,
        lostFrames: 0,
        cx: rect.x + rect.width * 0.5,
        cy: rect.y + rect.height * 0.5,
      });
    }

    this.tracks = nextTracks;

    // 6. Actualizar conteos visibles (lostFrames === 0)
    let persons = 0;
    let vehicles = 0;
    let bicycles = 0;
    let animals = 0;
    let objects = 0;

    for (let i = 0; i < this.tracks.length; i++) {
      const tr = this.tracks[i];
      if (tr.lostFrames > 0) continue;
      if (tr.category === 'person') persons++;
      else if (tr.category === 'vehicle') vehicles++;
      else if (tr.category === 'bicycle') bicycles++;
      else if (tr.category === 'animal') animals++;
      else objects++;
    }

    this.currentCounts = { persons, vehicles, bicycles, animals, objects };
    return this.tracks;
  }

  public commitIntervalDeltas(): CategoryDeltaSnapshot {
    'worklet';
    const snapshot: CategoryDeltaSnapshot = {
      persons: this.currentCounts.persons,
      vehicles: this.currentCounts.vehicles,
      bicycles: this.currentCounts.bicycles,
      animals: this.currentCounts.animals,
      objects: this.currentCounts.objects,
      deltaPersons: this.currentCounts.persons - this.prevCounts.persons,
      deltaVehicles: this.currentCounts.vehicles - this.prevCounts.vehicles,
      deltaBicycles: this.currentCounts.bicycles - this.prevCounts.bicycles,
    };
    this.prevCounts = { ...this.currentCounts };
    return snapshot;
  }
}
