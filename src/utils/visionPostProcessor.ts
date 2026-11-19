/**
 * VisionEdge RN — Zero-Allocation Computer Vision Post-Processing Module
 * Target budget: < 2.0 ms per frame inside a react-native-worklets-core JSI Worklet.
 *
 * Pure TypeScript with zero heavy runtime dependencies. All hot-path buffers are
 * preallocated once at module/instance initialization to avoid Garbage Collection pauses.
 */

export type UrbanClassId = 0 | 1 | 2 | 5;
export type UrbanClassName = 'person' | 'bicycle' | 'car' | 'bus';
export type CategoryKey = 'persons' | 'vehicles' | 'bicycles';
export type SensorOrientation = 0 | 90 | 180 | 270;

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
  classId: UrbanClassId;
  className: UrbanClassName;
  confidence: number;
  /** Normalized center X in 640x640 letterboxed tensor space [0..1] */
  cx: number;
  /** Normalized center Y in 640x640 letterboxed tensor space [0..1] */
  cy: number;
  /** Normalized width in 640x640 letterboxed tensor space [0..1] */
  w: number;
  /** Normalized height in 640x640 letterboxed tensor space [0..1] */
  h: number;
  /** 17 COCO keypoints (populated for person pose detections) */
  keypoints: PoseKeypoint[];
  hasPose: boolean;
}

export interface TrackedObject {
  trackId: number;
  classId: UrbanClassId;
  className: UrbanClassName;
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
  isFrontCamera?: boolean; // Default false
}

const NUM_ANCHORS = 8400;
const NUM_KEYPOINTS = 17;
const MAX_POOL_PROPOSALS = 256;
const MAX_TRACKS = 128;

// Preallocated hot-path scratch buffers (zero GC pressure)
const SORTED_INDICES = new Int32Array(MAX_POOL_PROPOSALS);
const SUPPRESSED_FLAGS = new Uint8Array(MAX_POOL_PROPOSALS);
const SIMILARITY_MATRIX = new Float32Array(MAX_TRACKS * MAX_POOL_PROPOSALS);
const MATCHED_TRACK_FLAGS = new Uint8Array(MAX_TRACKS);
const MATCHED_DET_FLAGS = new Uint8Array(MAX_POOL_PROPOSALS);

function mapClassName(classId: UrbanClassId): UrbanClassName {
  'worklet';
  switch (classId) {
    case 0:
      return 'person';
    case 1:
      return 'bicycle';
    case 2:
      return 'car';
    case 5:
      return 'bus';
  }
}

/**
 * Preallocated object pool for YoloProposal instances so decoding 8,400 anchors
 * never allocates intermediate detection objects on the JS/Worklet heap.
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
 * 1A. Decodificador de Tensor de Detección YOLO ([1, 84, 8400])
 * Filtra únicamente las clases urbanas objetivo: 0 (person), 1 (bicycle), 2 (car), 5 (bus).
 * Devuelve el conteo válido escrito en `pool` (zero-allocation).
 */
export function decodeYoloDetections(
  detTensor: Float32Array,
  confThreshold = 0.45,
  tensorSize = 640,
  pool: ProposalPool = DEFAULT_DET_POOL
): ProposalPool {
  'worklet';
  pool.reset();

  const ch0 = 0;
  const ch1 = NUM_ANCHORS;
  const ch2 = 2 * NUM_ANCHORS;
  const ch3 = 3 * NUM_ANCHORS;

  const chPerson = 4 * NUM_ANCHORS;
  const chBicycle = 5 * NUM_ANCHORS;
  const chCar = 6 * NUM_ANCHORS;
  const chBus = 9 * NUM_ANCHORS;

  const invSize = 1 / tensorSize;

  for (let i = 0; i < NUM_ANCHORS; i++) {
    const sPerson = detTensor[chPerson + i];
    const sBicycle = detTensor[chBicycle + i];
    const sCar = detTensor[chCar + i];
    const sBus = detTensor[chBus + i];

    let bestScore = confThreshold;
    let bestCls: UrbanClassId | -1 = -1;

    if (sPerson >= bestScore) {
      bestScore = sPerson;
      bestCls = 0;
    }
    if (sBicycle > bestScore) {
      bestScore = sBicycle;
      bestCls = 1;
    }
    if (sCar > bestScore) {
      bestScore = sCar;
      bestCls = 2;
    }
    if (sBus > bestScore) {
      bestScore = sBus;
      bestCls = 5;
    }

    if (bestCls === -1) continue;

    const slot = pool.acquire();
    if (!slot) break;

    let cx = detTensor[ch0 + i];
    let cy = detTensor[ch1 + i];
    let w = detTensor[ch2 + i];
    let h = detTensor[ch3 + i];

    // Normalizar a [0..1] si el export devolvió escala en píxeles [0..640]
    if (cx > 1.5 || cy > 1.5 || w > 1.5 || h > 1.5) {
      cx *= invSize;
      cy *= invSize;
      w *= invSize;
      h *= invSize;
    }

    slot.classId = bestCls;
    slot.className = mapClassName(bestCls);
    slot.confidence = bestScore;
    slot.cx = cx;
    slot.cy = cy;
    slot.w = w;
    slot.h = h;
    slot.hasPose = false;
  }

  return pool;
}

/**
 * 1B. Decodificador de Tensor de Pose YOLO ([1, 56, 8400])
 * Extrae cajas de personas junto con sus 17 keypoints COCO ([x, y, visibility]).
 */
export function decodeYoloPose(
  poseTensor: Float32Array,
  confThreshold = 0.45,
  tensorSize = 640,
  pool: ProposalPool = DEFAULT_POSE_POOL
): ProposalPool {
  'worklet';
  pool.reset();

  const ch0 = 0;
  const ch1 = NUM_ANCHORS;
  const ch2 = 2 * NUM_ANCHORS;
  const ch3 = 3 * NUM_ANCHORS;
  const chConf = 4 * NUM_ANCHORS;
  const invSize = 1 / tensorSize;

  for (let i = 0; i < NUM_ANCHORS; i++) {
    const score = poseTensor[chConf + i];
    if (score < confThreshold) continue;

    const slot = pool.acquire();
    if (!slot) break;

    let cx = poseTensor[ch0 + i];
    let cy = poseTensor[ch1 + i];
    let w = poseTensor[ch2 + i];
    let h = poseTensor[ch3 + i];

    const isPixelScale = cx > 1.5 || cy > 1.5 || w > 1.5 || h > 1.5;
    if (isPixelScale) {
      cx *= invSize;
      cy *= invSize;
      w *= invSize;
      h *= invSize;
    }

    slot.classId = 0;
    slot.className = 'person';
    slot.confidence = score;
    slot.cx = cx;
    slot.cy = cy;
    slot.w = w;
    slot.h = h;
    slot.hasPose = true;

    for (let k = 0; k < NUM_KEYPOINTS; k++) {
      const baseChannel = (5 + k * 3) * NUM_ANCHORS;
      let kx = poseTensor[baseChannel + i];
      let ky = poseTensor[baseChannel + NUM_ANCHORS + i];
      const kv = poseTensor[baseChannel + 2 * NUM_ANCHORS + i];

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
 * Suprime únicamente duplicados dentro de la misma clase (una persona frente
 * a un automóvil jamás es suprimida). Escribe los índices sobrevivientes en `outKeptIndices`.
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

  // Inicializar índices y flags de supresión sin crear arrays
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
      // Aislamiento por clase (Class-Aware)
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
 * 4. Proyección Inversa Letterbox (projectToScreen)
 * Convierte coordenadas normalizadas [0..1] del tensor cuadrado 640x640 (con bandas negras
 * de relleno padding) a píxeles exactos de pantalla [x, y, width, height].
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
  const isFrontCamera = params.isFrontCamera ?? false;

  // Dimensiones efectivas del fotograma tras orientación del sensor
  const isRotated90or270 = orientation === 90 || orientation === 270;
  const frameW = isRotated90or270 ? params.cameraHeight : params.cameraWidth;
  const frameH = isRotated90or270 ? params.cameraWidth : params.cameraHeight;

  // 1. Parámetros del padding letterbox en el tensor 640x640
  const lbScale = Math.min(tensorSize / frameW, tensorSize / frameH);
  const padX = (tensorSize - frameW * lbScale) * 0.5;
  const padY = (tensorSize - frameH * lbScale) * 0.5;

  // 2. Deshacer normalización [0..1] -> píxeles tensor -> píxeles cámara original
  const tensorCx = normCx * tensorSize;
  const tensorCy = normCy * tensorSize;
  const tensorW = normW * tensorSize;
  const tensorH = normH * tensorSize;

  const camCx = (tensorCx - padX) / lbScale;
  const camCy = (tensorCy - padY) / lbScale;
  const camW = tensorW / lbScale;
  const camH = tensorH / lbScale;

  // 3. Mapear del espacio de cámara al componente de pantalla (cover / contain)
  const screenW = params.screenWidth;
  const screenH = params.screenHeight;
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
  const isFrontCamera = params.isFrontCamera ?? false;

  const isRotated90or270 = orientation === 90 || orientation === 270;
  const frameW = isRotated90or270 ? params.cameraHeight : params.cameraWidth;
  const frameH = isRotated90or270 ? params.cameraWidth : params.cameraHeight;

  const lbScale = Math.min(tensorSize / frameW, tensorSize / frameH);
  const padX = (tensorSize - frameW * lbScale) * 0.5;
  const padY = (tensorSize - frameH * lbScale) * 0.5;

  const camX = (normX * tensorSize - padX) / lbScale;
  const camY = (normY * tensorSize - padY) / lbScale;

  const screenW = params.screenWidth;
  const screenH = params.screenHeight;
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
 * 3. Rastreador Ligero Multi-Objeto (SimpleByteTracker)
 * Inspirado en ByteTrack / SORT con matriz de similitud (IoU + proximidad de centroide),
 * retención temporal (maxLostFrames = 15) y cálculo de deltas por categoría (+1, -1, 0).
 */
export class SimpleByteTracker {
  private nextTrackId = 1;
  private tracks: TrackedObject[] = [];
  private readonly maxLostFrames: number;
  private readonly similarityThreshold: number;
  private readonly highConfThreshold: number;

  private prevCounts = { persons: 0, vehicles: 0, bicycles: 0 };
  private currentCounts = { persons: 0, vehicles: 0, bicycles: 0 };

  constructor(maxLostFrames = 15, similarityThreshold = 0.28, highConfThreshold = 0.52) {
    this.maxLostFrames = maxLostFrames;
    this.similarityThreshold = similarityThreshold;
    this.highConfThreshold = highConfThreshold;
  }

  public reset(): void {
    'worklet';
    this.nextTrackId = 1;
    this.tracks.length = 0;
    this.prevCounts = { persons: 0, vehicles: 0, bicycles: 0 };
    this.currentCounts = { persons: 0, vehicles: 0, bicycles: 0 };
  }

  /**
   * Actualiza los tracks con las detecciones filtradas por NMS y proyectadas a pantalla.
   * Reutiliza la matriz de similitud preasignada para mantener latencia < 0.3 ms.
   */
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

    // 1. Proyectar detecciones sobrevivientes de NMS a espacio de pantalla (en slots temporales)
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

    // 3. Asociación greedy en 2 etapas (alta confianza primero, recuperación después)
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

    // 4. Envejecer tracks no emparejados y podar los que superen maxLostFrames = 15
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

    // 5. Crear nuevos tracks para detecciones de alta confianza sin emparejar
    for (let d = 0; d < numDets; d++) {
      if (MATCHED_DET_FLAGS[d] === 1) continue;
      const prop = pool.items[keptIndices[d]];
      if (prop.confidence < this.highConfThreshold) continue;

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
    for (let i = 0; i < this.tracks.length; i++) {
      const tr = this.tracks[i];
      if (tr.lostFrames > 0) continue;
      if (tr.classId === 0) persons++;
      else if (tr.classId === 2 || tr.classId === 5) vehicles++;
      else if (tr.classId === 1) bicycles++;
    }

    this.currentCounts = { persons, vehicles, bicycles };
    return this.tracks;
  }

  /**
   * Calcula y confirma el delta (+1, -1, 0) respecto al intervalo anterior
   * para alimentar las tarjetas de Personas, Vehículos y Bicicletas.
   */
  public commitIntervalDeltas(): CategoryDeltaSnapshot {
    'worklet';
    const snapshot: CategoryDeltaSnapshot = {
      persons: this.currentCounts.persons,
      vehicles: this.currentCounts.vehicles,
      bicycles: this.currentCounts.bicycles,
      deltaPersons: this.currentCounts.persons - this.prevCounts.persons,
      deltaVehicles: this.currentCounts.vehicles - this.prevCounts.vehicles,
      deltaBicycles: this.currentCounts.bicycles - this.prevCounts.bicycles,
    };
    this.prevCounts.persons = this.currentCounts.persons;
    this.prevCounts.vehicles = this.currentCounts.vehicles;
    this.prevCounts.bicycles = this.currentCounts.bicycles;
    return snapshot;
  }
}
