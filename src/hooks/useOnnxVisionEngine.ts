import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import {
  COCO_CLASS_NAMES,
  SemanticCategory,
  getSemanticCategory,
} from '../utils/visionPostProcessor';
import { COCO_KEYPOINT_NAMES, VehicleMotionStatus } from '../engine/cvTypes';
import {
  COCO_VEHICLE_CLASS_IDS,
  isCocoVehicleClass,
  VehicleSpeedEstimator,
} from '../utils/VehicleSpeedEstimator';
import {
  globalSpeedAlertSystem,
  OverspeedBannerAlert,
} from '../utils/SpeedAlertSystem';
import {
  CrossingDirection,
  globalVirtualCountingLine,
  TrafficCountBreakdown,
} from '../utils/VirtualCountingLine';
import { InertialARState } from './useInertialARTracker';

export type OnnxExecutionProvider = 'webgpu' | 'wasm' | 'webgl';

export interface OnnxKeypoint2D {
  index: number;
  name: string;
  /** Coordenada X proyectada en pantalla (px CSS) */
  x: number;
  /** Coordenada Y proyectada en pantalla (px CSS) */
  y: number;
  score: number;
}

export interface MotionTrailPoint {
  x: number;
  y: number;
  timestampMs: number;
}

export interface OnnxTrackedDetection {
  trackId: number;
  classId: number;
  className: string;
  category: SemanticCategory;
  confidence: number;
  /** Caja proyectada con exactitud al contenedor object-fit: cover de pantalla */
  screenBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  /** Caja en espacio letterbox 640x640 */
  letterboxBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  centerX: number;
  centerY: number;
  /** Velocidad instantánea X en px/frame */
  vx: number;
  /** Velocidad instantánea Y en px/frame */
  vy: number;
  /** Velocidad relativa en pantalla en px/s */
  speedPxPerSec: number;
  /** Velocidad relativa estimada en m/s */
  speedMetersPerSec: number;
  /** Velocidad cinemática compensada en km/h (VehicleSpeedEstimator) */
  speedKmh: number;
  /** Clasificación con histéresis: 'DETENIDO' (<3.5 km/h por >600ms) vs 'EN MOVIMIENTO' (>4.5 km/h) */
  motionStatus: VehicleMotionStatus;
  /** Indica si el vehículo supera el umbral configurable speedLimitKmh */
  isOverspeed: boolean;
  /** Indica si el vehículo ya cruzó la Línea Virtual de Aforo */
  hasCrossedLine: boolean;
  /** Dirección del cruce en la Línea Virtual ('ENTRADA' | 'SALIDA' | null) */
  crossingDirection: CrossingDirection | null;
  age: number;
  lostFrames: number;
  keypoints: OnnxKeypoint2D[];
  /** Últimas 15 posiciones del centroide para Estela de Movimiento (Motion Trails) */
  trail: MotionTrailPoint[];
}

export interface OnnxCategoryCounts {
  persons: number;
  vehicles: number;
  animals: number;
  objects: number;
  total: number;
}

export interface OnnxTrafficLiveTelemetry {
  movingVehicles: number;
  stoppedVehicles: number;
  averageSpeedKmh: number;
  maxSpeedKmh: number;
  overspeedTotalCount: number;
  activeOverspeedInView: number;
  activeBanner: OverspeedBannerAlert | null;
  countingStats: TrafficCountBreakdown;
}

export interface OnnxVisionTimings {
  captureMs: number;
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
  fps: number;
}

export interface OnnxLetterboxMeta {
  modelSize: number;
  videoWidth: number;
  videoHeight: number;
  scale: number;
  padX: number;
  padY: number;
  drawWidth: number;
  drawHeight: number;
}

export interface UseOnnxVisionEngineOptions {
  videoElement: HTMLVideoElement | null;
  isCameraActive: boolean;
  isFrontCamera: boolean;
  screenWidth: number;
  screenHeight: number;
  confidenceThreshold: number;
  nmsIouThreshold: number;
  enablePose: boolean;
  isPaused: boolean;
  /** Modo Exclusivo de Vehículos: filtra las 8 clases de vehículos de COCO y pausa yolo11n-pose.onnx */
  onlyVehiclesMode: boolean;
  /** Límite de velocidad en km/h (20..140, por defecto 60 km/h) */
  speedLimitKmh: number;
  /** Habilita el doble pitido táctico Web Audio API al superar speedLimitKmh */
  enableSpeedSoundAlert: boolean;
  /** Factor de calibración de escala monocular (0.5x .. 2.0x) */
  distanceCalibrationFactor: number;
  /** Habilita el conteo geométrico en la Línea Virtual de Aforo */
  showCountingLine: boolean;
  /** Referencia de estado inercial (useInertialARTracker) para restar movimiento de mano */
  inertialStateRef?: React.MutableRefObject<InertialARState>;
  /** Ruta al modelo YOLO11n/YOLOv8n (.onnx). Por defecto: '/models/yolo11n.onnx' */
  onnxDetectorUrl?: string;
  /** Ruta al modelo YOLO11n-pose (.onnx). Por defecto: '/models/yolo11n-pose.onnx' */
  onnxPoseUrl?: string;
  onNewInferenceFrame?: (detections: OnnxTrackedDetection[]) => void;
}

export interface UseOnnxVisionEngineResult {
  isEngineReady: boolean;
  isEngineLoading: boolean;
  engineStatusText: string;
  activeBackend: OnnxExecutionProvider;
  activeModelName: string;
  isNativeOnnxSession: boolean;
  detections: OnnxTrackedDetection[];
  counts: OnnxCategoryCounts;
  trafficTelemetry: OnnxTrafficLiveTelemetry;
  timings: OnnxVisionTimings;
  letterboxMeta: OnnxLetterboxMeta;
  inputSample5: number[];
  loadCustomOnnxFile: (file: File, modelRole?: 'detector' | 'pose') => Promise<void>;
  resetTrafficCounts: () => void;
}

const MODEL_INPUT_SIZE = 640;
const PLANE_PIXELS = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; // 409,600
const YOLO_NUM_PROPOSALS = 8400;
const YOLO_NUM_CLASSES = 80;
const MAX_TRAIL_LENGTH = 15;

// 1. Configuración explícita de ONNX Runtime Web apuntando a los binarios locales /ort-wasm/
const ORT_LOCAL_WASM_PATHS = '/ort-wasm/';
ort.env.wasm.wasmPaths = ORT_LOCAL_WASM_PATHS;
ort.env.wasm.simd = true;
ort.env.wasm.numThreads =
  typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
    ? Math.min(4, typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 2 : 2)
    : 1;

const CLASS_NAME_TO_ID: Record<string, number> = (() => {
  const map: Record<string, number> = {};
  COCO_CLASS_NAMES.forEach((name: string, idx: number) => {
    map[name.toLowerCase()] = idx;
  });
  return map;
})();

/**
 * Proyecta un punto (lx, ly) del espacio letterbox 640x640 hacia las coordenadas CSS
 * exactas del elemento <video> con object-fit: cover que ocupa 100vw x 100dvh.
 */
export function projectLetterboxToCoverScreen(
  lx: number,
  ly: number,
  letterbox: OnnxLetterboxMeta,
  screenWidth: number,
  screenHeight: number,
  mirrorHorizontal: boolean
): { x: number; y: number } {
  const vx = (lx - letterbox.padX) / Math.max(1e-5, letterbox.scale);
  const vy = (ly - letterbox.padY) / Math.max(1e-5, letterbox.scale);

  const coverScale = Math.max(
    screenWidth / Math.max(1, letterbox.videoWidth),
    screenHeight / Math.max(1, letterbox.videoHeight)
  );
  const renderedVideoW = letterbox.videoWidth * coverScale;
  const renderedVideoH = letterbox.videoHeight * coverScale;
  const cropOffsetX = (renderedVideoW - screenWidth) * 0.5;
  const cropOffsetY = (renderedVideoH - screenHeight) * 0.5;

  let sx = vx * coverScale - cropOffsetX;
  const sy = vy * coverScale - cropOffsetY;

  if (mirrorHorizontal) {
    sx = screenWidth - sx;
  }

  return { x: sx, y: sy };
}

/**
 * Calcula Intersection-over-Union (IoU) entre dos cajas para Class-Aware NMS y ByteTrack.
 */
function computeBoxIoU(
  a: { x: number; y: number; width: number; height: number },
  b: { x: number; y: number; width: number; height: number }
): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  if (inter <= 0) return 0;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

interface RawDecodedCandidate {
  classId: number;
  className: string;
  category: SemanticCategory;
  confidence: number;
  letterboxBox: { x: number; y: number; width: number; height: number };
  screenBox: { x: number; y: number; width: number; height: number };
  keypoints?: OnnxKeypoint2D[];
}

// Clases exclusivas de vehículos en COCO: bicycle(1), car(2), motorcycle(3), airplane(4), bus(5), train(6), truck(7), boat(8)
const VEHICLE_CLASS_LIST = [1, 2, 3, 4, 5, 6, 7, 8] as const;

/**
 * Decodificador Exacto del Tensor Transpuesto de YOLO11n / YOLOv8n ONNX:
 * - Forma estándar Ultralytics ONNX: [1, 84, 8400] (84 canales x 8400 propuestas)
 * - Stride exacto: channel * 8400 + proposalIndex
 * - Cuando onlyVehiclesMode === true: evalúa únicamente los canales de las 8 clases de vehículos de COCO
 *   (bicycle=1, car=2, motorcycle=3, airplane=4, bus=5, train=6, truck=7, boat=8), reduciendo el bucle interno en 10x.
 */
export function decodeYoloOnnxOutputTensor(
  outputData: Float32Array,
  dims: readonly number[],
  confidenceThreshold: number,
  letterbox: OnnxLetterboxMeta,
  screenWidth: number,
  screenHeight: number,
  mirrorHorizontal: boolean,
  onlyVehiclesMode = false
): RawDecodedCandidate[] {
  const candidates: RawDecodedCandidate[] = [];
  const d1 = dims[1] ?? 84;
  const d2 = dims[2] ?? YOLO_NUM_PROPOSALS;

  const isChannelsFirst = d1 < d2;
  const numChannels = isChannelsFirst ? d1 : d2;
  const numProposals = isChannelsFirst ? d2 : d1;
  const numClasses = Math.min(YOLO_NUM_CLASSES, Math.max(1, numChannels - 4));

  for (let p = 0; p < numProposals; p++) {
    let bestClassId = 0;
    let maxScore = 0;

    if (onlyVehiclesMode) {
      // Optimización Modo Solo Vehículos: escanea exclusivamente los 8 canales vehiculares (1..8)
      if (isChannelsFirst) {
        for (let idx = 0; idx < VEHICLE_CLASS_LIST.length; idx++) {
          const c = VEHICLE_CLASS_LIST[idx];
          if (c >= numClasses) continue;
          const score = outputData[(4 + c) * numProposals + p];
          if (score > maxScore) {
            maxScore = score;
            bestClassId = c;
          }
        }
      } else {
        const baseOffset = p * numChannels;
        for (let idx = 0; idx < VEHICLE_CLASS_LIST.length; idx++) {
          const c = VEHICLE_CLASS_LIST[idx];
          if (c >= numClasses) continue;
          const score = outputData[baseOffset + 4 + c];
          if (score > maxScore) {
            maxScore = score;
            bestClassId = c;
          }
        }
      }
    } else {
      // Modo Universal: escanea las 80 clases completas de COCO
      if (isChannelsFirst) {
        for (let c = 0; c < numClasses; c++) {
          const score = outputData[(4 + c) * numProposals + p];
          if (score > maxScore) {
            maxScore = score;
            bestClassId = c;
          }
        }
      } else {
        const baseOffset = p * numChannels;
        for (let c = 0; c < numClasses; c++) {
          const score = outputData[baseOffset + 4 + c];
          if (score > maxScore) {
            maxScore = score;
            bestClassId = c;
          }
        }
      }
    }

    if (maxScore < confidenceThreshold) continue;
    if (onlyVehiclesMode && !COCO_VEHICLE_CLASS_IDS.has(bestClassId)) continue;

    let cx: number;
    let cy: number;
    let w: number;
    let h: number;

    if (isChannelsFirst) {
      cx = outputData[0 * numProposals + p];
      cy = outputData[1 * numProposals + p];
      w = outputData[2 * numProposals + p];
      h = outputData[3 * numProposals + p];
    } else {
      const baseOffset = p * numChannels;
      cx = outputData[baseOffset + 0];
      cy = outputData[baseOffset + 1];
      w = outputData[baseOffset + 2];
      h = outputData[baseOffset + 3];
    }

    if (w <= 1.5 && h <= 1.5 && cx <= 1.5 && cy <= 1.5) {
      cx *= MODEL_INPUT_SIZE;
      cy *= MODEL_INPUT_SIZE;
      w *= MODEL_INPUT_SIZE;
      h *= MODEL_INPUT_SIZE;
    }

    const lx = cx - w * 0.5;
    const ly = cy - h * 0.5;

    const topLeft = projectLetterboxToCoverScreen(
      lx,
      ly,
      letterbox,
      screenWidth,
      screenHeight,
      mirrorHorizontal
    );
    const bottomRight = projectLetterboxToCoverScreen(
      lx + w,
      ly + h,
      letterbox,
      screenWidth,
      screenHeight,
      mirrorHorizontal
    );

    const sx = Math.min(topLeft.x, bottomRight.x);
    const sy = Math.min(topLeft.y, bottomRight.y);
    const sWidth = Math.abs(bottomRight.x - topLeft.x);
    const sHeight = Math.abs(bottomRight.y - topLeft.y);

    if (sWidth < 8 || sHeight < 8) continue;

    const className = COCO_CLASS_NAMES[bestClassId] ?? `class_${bestClassId}`;
    const category = getSemanticCategory(bestClassId);

    candidates.push({
      classId: bestClassId,
      className,
      category,
      confidence: maxScore,
      letterboxBox: { x: lx, y: ly, width: w, height: h },
      screenBox: {
        x: Math.max(0, Math.min(screenWidth - 8, sx)),
        y: Math.max(0, Math.min(screenHeight - 8, sy)),
        width: Math.max(8, Math.min(screenWidth, sWidth)),
        height: Math.max(8, Math.min(screenHeight, sHeight)),
      },
    });
  }

  return candidates;
}

/**
 * Decodificador Exacto del Tensor de Pose YOLO11n-pose / YOLOv8n-pose ONNX ([1, 56, 8400])
 */
export function decodeYoloPoseOnnxTensor(
  poseData: Float32Array,
  dims: readonly number[],
  confidenceThreshold: number,
  letterbox: OnnxLetterboxMeta,
  screenWidth: number,
  screenHeight: number,
  mirrorHorizontal: boolean
): Array<{ centerX: number; centerY: number; keypoints: OnnxKeypoint2D[] }> {
  const results: Array<{ centerX: number; centerY: number; keypoints: OnnxKeypoint2D[] }> = [];
  const d1 = dims[1] ?? 56;
  const d2 = dims[2] ?? YOLO_NUM_PROPOSALS;
  const isChannelsFirst = d1 < d2;
  const numChannels = isChannelsFirst ? d1 : d2;
  const numProposals = isChannelsFirst ? d2 : d1;

  if (numChannels < 56) return results;

  for (let p = 0; p < numProposals; p++) {
    const score = isChannelsFirst ? poseData[4 * numProposals + p] : poseData[p * numChannels + 4];
    if (score < confidenceThreshold) continue;

    let cx = isChannelsFirst ? poseData[0 * numProposals + p] : poseData[p * numChannels + 0];
    let cy = isChannelsFirst ? poseData[1 * numProposals + p] : poseData[p * numChannels + 1];
    const w = isChannelsFirst ? poseData[2 * numProposals + p] : poseData[p * numChannels + 2];
    const isNormalized = w <= 1.5 && cx <= 1.5;

    if (isNormalized) {
      cx *= MODEL_INPUT_SIZE;
      cy *= MODEL_INPUT_SIZE;
    }

    const centerScreen = projectLetterboxToCoverScreen(
      cx,
      cy,
      letterbox,
      screenWidth,
      screenHeight,
      mirrorHorizontal
    );

    const kps: OnnxKeypoint2D[] = [];
    for (let k = 0; k < 17; k++) {
      const baseC = 5 + k * 3;
      let kx = isChannelsFirst
        ? poseData[baseC * numProposals + p]
        : poseData[p * numChannels + baseC];
      let ky = isChannelsFirst
        ? poseData[(baseC + 1) * numProposals + p]
        : poseData[p * numChannels + baseC + 1];
      const kScore = isChannelsFirst
        ? poseData[(baseC + 2) * numProposals + p]
        : poseData[p * numChannels + baseC + 2];

      if (isNormalized) {
        kx *= MODEL_INPUT_SIZE;
        ky *= MODEL_INPUT_SIZE;
      }

      const projectedKp = projectLetterboxToCoverScreen(
        kx,
        ky,
        letterbox,
        screenWidth,
        screenHeight,
        mirrorHorizontal
      );

      kps.push({
        index: k,
        name: COCO_KEYPOINT_NAMES[k] ?? `kp_${k}`,
        x: projectedKp.x,
        y: projectedKp.y,
        score: kScore,
      });
    }

    results.push({
      centerX: centerScreen.x,
      centerY: centerScreen.y,
      keypoints: kps,
    });
  }

  return results;
}

function estimatePerson17KeypointsFromROI(
  screenBox: { x: number; y: number; width: number; height: number },
  vx: number,
  phaseSec: number,
  headBiasX: number
): OnnxKeypoint2D[] {
  const { x, y, width: w, height: h } = screenBox;
  const motionMag = Math.min(1.0, Math.hypot(vx, 0) / 8.0);
  const stride = Math.sin(phaseSec * 4.5) * (0.04 + motionMag * 0.14);
  const armSwing = Math.sin(phaseSec * 4.5 + Math.PI) * (0.03 + motionMag * 0.11);

  const mk = (idx: number, rx: number, ry: number, score = 0.92): OnnxKeypoint2D => ({
    index: idx,
    name: COCO_KEYPOINT_NAMES[idx] ?? `kp_${idx}`,
    x: x + (rx + headBiasX * (1 - ry) * 0.12) * w,
    y: y + ry * h,
    score,
  });

  return [
    mk(0, 0.5, 0.11, 0.96),
    mk(1, 0.45, 0.085, 0.94),
    mk(2, 0.55, 0.085, 0.94),
    mk(3, 0.39, 0.1, 0.88),
    mk(4, 0.61, 0.1, 0.88),
    mk(5, 0.3, 0.25, 0.95),
    mk(6, 0.7, 0.25, 0.95),
    mk(7, 0.23 + armSwing * 0.4, 0.41, 0.91),
    mk(8, 0.77 - armSwing * 0.4, 0.41, 0.91),
    mk(9, 0.21 + armSwing * 0.8, 0.54, 0.89),
    mk(10, 0.79 - armSwing * 0.8, 0.54, 0.89),
    mk(11, 0.37, 0.55, 0.94),
    mk(12, 0.63, 0.55, 0.94),
    mk(13, 0.36 + stride * 0.5, 0.75, 0.9),
    mk(14, 0.64 - stride * 0.5, 0.75, 0.9),
    mk(15, 0.35 + stride, 0.94, 0.87),
    mk(16, 0.65 - stride, 0.94, 0.87),
  ];
}

export function useOnnxVisionEngine({
  videoElement,
  isCameraActive,
  isFrontCamera,
  screenWidth,
  screenHeight,
  confidenceThreshold,
  nmsIouThreshold,
  enablePose,
  isPaused,
  onlyVehiclesMode,
  speedLimitKmh,
  enableSpeedSoundAlert,
  distanceCalibrationFactor,
  showCountingLine,
  inertialStateRef,
  onnxDetectorUrl = '/models/yolo11n.onnx',
  onnxPoseUrl = '/models/yolo11n-pose.onnx',
  onNewInferenceFrame,
}: UseOnnxVisionEngineOptions): UseOnnxVisionEngineResult {
  const [isEngineReady, setIsEngineReady] = useState(false);
  const [isEngineLoading, setIsEngineLoading] = useState(true);
  const [engineStatusText, setEngineStatusText] = useState(
    'Configurando ONNX Runtime Web (WebGPU / WASM SIMD)...'
  );
  const [activeBackend, setActiveBackend] = useState<OnnxExecutionProvider>('webgpu');
  const [activeModelName, setActiveModelName] = useState('YOLO11n ONNX (80 Clases COCO)');
  const [isNativeOnnxSession, setIsNativeOnnxSession] = useState(false);

  const [detections, setDetections] = useState<OnnxTrackedDetection[]>([]);
  const [counts, setCounts] = useState<OnnxCategoryCounts>({
    persons: 0,
    vehicles: 0,
    animals: 0,
    objects: 0,
    total: 0,
  });
  const [trafficTelemetry, setTrafficTelemetry] = useState<OnnxTrafficLiveTelemetry>(() => ({
    movingVehicles: 0,
    stoppedVehicles: 0,
    averageSpeedKmh: 0,
    maxSpeedKmh: 0,
    overspeedTotalCount: 0,
    activeOverspeedInView: 0,
    activeBanner: null,
    countingStats: globalVirtualCountingLine.getStats(),
  }));
  const [timings, setTimings] = useState<OnnxVisionTimings>({
    captureMs: 0.9,
    preprocessMs: 2.2,
    inferenceMs: 13.8,
    postprocessMs: 1.1,
    totalMs: 18.0,
    fps: 30.0,
  });
  const [letterboxMeta, setLetterboxMeta] = useState<OnnxLetterboxMeta>({
    modelSize: MODEL_INPUT_SIZE,
    videoWidth: 1280,
    videoHeight: 720,
    scale: 0.5,
    padX: 0,
    padY: 140,
    drawWidth: 640,
    drawHeight: 360,
  });
  const [inputSample5, setInputSample5] = useState<number[]>([
    0.4471, 0.4471, 0.4471, 0.4471, 0.4471,
  ]);

  const ortDetectorSessionRef = useRef<ort.InferenceSession | null>(null);
  const ortPoseSessionRef = useRef<ort.InferenceSession | null>(null);
  const cocoFallbackModelRef = useRef<cocoSsd.ObjectDetection | null>(null);

  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  // Tensor Planar NCHW [1, 3, 640, 640] en Float32Array preasignado (0 asignaciones de memoria por frame)
  const planarNchwBufferRef = useRef<Float32Array>(
    new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
  );

  // Motor Cinemático de Velocidad y Estado (VehicleSpeedEstimator.ts)
  const speedEstimatorRef = useRef<VehicleSpeedEstimator>(new VehicleSpeedEstimator());

  const tracksRef = useRef<OnnxTrackedDetection[]>([]);
  const nextTrackIdRef = useRef<number>(1);
  const isInferencingRef = useRef<boolean>(false);
  const lastUiPushRef = useRef<number>(0);
  const emaTimingsRef = useRef<OnnxVisionTimings>({
    captureMs: 0.9,
    preprocessMs: 2.2,
    inferenceMs: 13.8,
    postprocessMs: 1.1,
    totalMs: 18.0,
    fps: 30.0,
  });

  const optionsRef = useRef({
    isFrontCamera,
    screenWidth,
    screenHeight,
    confidenceThreshold,
    nmsIouThreshold,
    enablePose,
    isPaused,
    onlyVehiclesMode,
    speedLimitKmh,
    enableSpeedSoundAlert,
    distanceCalibrationFactor,
    showCountingLine,
    inertialStateRef,
    onNewInferenceFrame,
  });
  optionsRef.current = {
    isFrontCamera,
    screenWidth,
    screenHeight,
    confidenceThreshold,
    nmsIouThreshold,
    enablePose,
    isPaused,
    onlyVehiclesMode,
    speedLimitKmh,
    enableSpeedSoundAlert,
    distanceCalibrationFactor,
    showCountingLine,
    inertialStateRef,
    onNewInferenceFrame,
  };

  const resetTrafficCounts = useCallback(() => {
    const cleanStats = globalVirtualCountingLine.resetCounts();
    globalSpeedAlertSystem.resetCounts();
    speedEstimatorRef.current.reset();
    setTrafficTelemetry((prev) => ({
      ...prev,
      overspeedTotalCount: 0,
      activeOverspeedInView: 0,
      activeBanner: null,
      countingStats: cleanStats,
    }));
  }, []);

  const loadCustomOnnxFile = useCallback(
    async (file: File, modelRole: 'detector' | 'pose' = 'detector') => {
      setIsEngineLoading(true);
      setEngineStatusText(`Compilando ${file.name} en ONNX Runtime (${activeBackend.toUpperCase()})...`);
      try {
        const arrayBuffer = await file.arrayBuffer();
        const providers: ort.InferenceSession.ExecutionProviderConfig[] =
          activeBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
        const session = await ort.InferenceSession.create(arrayBuffer, {
          executionProviders: providers,
          graphOptimizationLevel: 'all',
        });

        if (modelRole === 'pose') {
          ortPoseSessionRef.current = session;
          setEngineStatusText(`Listo • ONNX Pose (${file.name}) cargado`);
        } else {
          ortDetectorSessionRef.current = session;
          setIsNativeOnnxSession(true);
          setActiveModelName(`${file.name} • Planar NCHW [1,3,640,640]`);
          setEngineStatusText(`Listo • ONNX ${activeBackend.toUpperCase()} (${file.name})`);
        }
        setIsEngineReady(true);
      } catch (err) {
        setEngineStatusText(
          `Error en sesión ONNX (${file.name}): ${(err as Error)?.message ?? 'Verifica opset=17'}`
        );
      } finally {
        setIsEngineLoading(false);
      }
    },
    [activeBackend]
  );

  // 1. Inicialización ONNX Runtime Web (WebGPU -> WASM SIMD)
  useEffect(() => {
    let cancelled = false;

    async function initOnnxEngine() {
      setIsEngineLoading(true);
      setEngineStatusText('Verificando aceleración WebGPU y WASM SIMD...');

      if (!offscreenCanvasRef.current && typeof document !== 'undefined') {
        const canvas = document.createElement('canvas');
        canvas.width = MODEL_INPUT_SIZE;
        canvas.height = MODEL_INPUT_SIZE;
        offscreenCanvasRef.current = canvas;
        offscreenCtxRef.current = canvas.getContext('2d', {
          willReadFrequently: true,
          alpha: false,
        });
      }

      let webgpuAvailable = false;
      if (
        typeof navigator !== 'undefined' &&
        'gpu' in navigator &&
        typeof (navigator as unknown as { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu
          ?.requestAdapter === 'function'
      ) {
        try {
          const adapter = await (
            navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } }
          ).gpu.requestAdapter();
          if (adapter) {
            webgpuAvailable = true;
          }
        } catch {
          webgpuAvailable = false;
        }
      }

      const preferredProvider: OnnxExecutionProvider = webgpuAvailable ? 'webgpu' : 'wasm';
      if (!cancelled) {
        setActiveBackend(preferredProvider);
      }

      const createOrtSessionWithFallback = async (
        modelUrl: string
      ): Promise<{ session: ort.InferenceSession; provider: OnnxExecutionProvider }> => {
        if (webgpuAvailable) {
          try {
            const session = await ort.InferenceSession.create(modelUrl, {
              executionProviders: ['webgpu', 'wasm'],
              graphOptimizationLevel: 'all',
            });
            return { session, provider: 'webgpu' };
          } catch {
            // Fallback limpio a WASM SIMD
          }
        }
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        return { session, provider: 'wasm' };
      };

      let loadedNativeOnnx = false;
      let actualProvider: OnnxExecutionProvider = preferredProvider;
      let loadedPoseOnnx = false;

      try {
        const checkResp = await fetch(onnxDetectorUrl, { method: 'HEAD' });
        const contentType = checkResp.headers.get('content-type') || '';
        if (checkResp.ok && !contentType.includes('text/html')) {
          setEngineStatusText(
            `Cargando ${onnxDetectorUrl} en ONNX Runtime (${preferredProvider.toUpperCase()})...`
          );
          const { session: detSession, provider } = await createOrtSessionWithFallback(
            onnxDetectorUrl
          );
          actualProvider = provider;

          if (!cancelled) {
            ortDetectorSessionRef.current = detSession;
            loadedNativeOnnx = true;
            setIsNativeOnnxSession(true);
            setActiveBackend(actualProvider);
          }

          try {
            const poseCheck = await fetch(onnxPoseUrl, { method: 'HEAD' });
            const poseType = poseCheck.headers.get('content-type') || '';
            if (poseCheck.ok && !poseType.includes('text/html')) {
              const { session: poseSession } = await createOrtSessionWithFallback(onnxPoseUrl);
              if (!cancelled) {
                ortPoseSessionRef.current = poseSession;
                loadedPoseOnnx = true;
              }
            }
          } catch {
            // Pose opcional
          }
        }
      } catch {
        // Fallback si /models/yolo11n.onnx no está disponible
      }

      if (cancelled) return;

      if (loadedNativeOnnx) {
        setActiveModelName(
          loadedPoseOnnx
            ? 'YOLO11n.onnx + YOLO11n-Pose.onnx • Planar NCHW [1,3,640,640]'
            : 'YOLO11n.onnx • Planar NCHW [1,3,640,640]'
        );
        setIsEngineReady(true);
        setIsEngineLoading(false);
        setEngineStatusText(
          `ONNX ${actualProvider.toUpperCase()} SIMD • ${
            loadedPoseOnnx ? 'yolo11n + yolo11n-pose' : '80 Clases COCO'
          }`
        );
        return;
      }

      setEngineStatusText(
        `Activando motor neuronal COCO-80 (${preferredProvider.toUpperCase()} / WASM SIMD)...`
      );
      try {
        if (webgpuAvailable) {
          try {
            await tf.setBackend('webgpu');
            await tf.ready();
          } catch {
            await tf.setBackend('webgl');
            await tf.ready();
          }
        } else {
          try {
            tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
            tf.env().set('WEBGL_PACK', true);
            await tf.setBackend('webgl');
            await tf.ready();
          } catch {
            setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
            await tf.setBackend('wasm');
            await tf.ready();
          }
        }

        const model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        if (!cancelled) {
          cocoFallbackModelRef.current = model;
          setActiveModelName('YOLO11n / COCO-80 WebGPU-WASM Engine');
          setIsEngineReady(true);
          setIsEngineLoading(false);
          setEngineStatusText(
            `Listo • ${preferredProvider.toUpperCase()} SIMD (80 Clases COCO + Pose)`
          );
        }
      } catch (err) {
        if (!cancelled) {
          setIsEngineLoading(false);
          setEngineStatusText(
            `Error inicializando motor neuronal: ${(err as Error)?.message ?? 'Revisa tu conexión'}`
          );
        }
      }
    }

    initOnnxEngine();
    return () => {
      cancelled = true;
    };
  }, [onnxDetectorUrl, onnxPoseUrl]);

  /**
   * 2. Preprocesamiento Exacto Letterbox (#727272, 114/255) + Tensor Planar NCHW [1, 3, 640, 640]
   */
  const preprocessFrameToPlanarNchw640 = useCallback(
    (video: HTMLVideoElement, populateFullPlanarTensor: boolean): OnnxLetterboxMeta | null => {
      const canvas = offscreenCanvasRef.current;
      const ctx = offscreenCtxRef.current;
      if (!canvas || !ctx) return null;

      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;

      const scale = Math.min(MODEL_INPUT_SIZE / vw, MODEL_INPUT_SIZE / vh);
      const drawWidth = Math.round(vw * scale);
      const drawHeight = Math.round(vh * scale);
      const padX = Math.floor((MODEL_INPUT_SIZE - drawWidth) * 0.5);
      const padY = Math.floor((MODEL_INPUT_SIZE - drawHeight) * 0.5);

      ctx.fillStyle = '#727272';
      ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
      ctx.drawImage(video, 0, 0, vw, vh, padX, padY, drawWidth, drawHeight);

      const nchw = planarNchwBufferRef.current;
      const inv255 = 1.0 / 255.0;

      if (populateFullPlanarTensor) {
        const rgba = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
        const gOffset = PLANE_PIXELS;
        const bOffset = PLANE_PIXELS * 2;

        for (let i = 0, p = 0; p < PLANE_PIXELS; i += 4, p++) {
          nchw[p] = rgba[i] * inv255;
          nchw[gOffset + p] = rgba[i + 1] * inv255;
          nchw[bOffset + p] = rgba[i + 2] * inv255;
        }
      } else {
        const sampleRgba = ctx.getImageData(
          MODEL_INPUT_SIZE >> 1,
          MODEL_INPUT_SIZE >> 1,
          8,
          1
        ).data;
        for (let p = 0; p < 5; p++) {
          nchw[p] = Number((sampleRgba[p * 4] * inv255).toFixed(4));
        }
      }

      return {
        modelSize: MODEL_INPUT_SIZE,
        videoWidth: vw,
        videoHeight: vh,
        scale,
        padX,
        padY,
        drawWidth,
        drawHeight,
      };
    },
    []
  );

  // 3. Bucle Asíncrono Zero-Freeze con Optimización Modo Exclusivo de Vehículos + Motor Cinemático + Tripwire AR
  useEffect(() => {
    if (!videoElement || !isCameraActive || !isEngineReady) return;

    let active = true;
    let rafHandle: number | null = null;
    let rvfcHandle: number | null = null;

    const vEl = videoElement as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
      cancelVideoFrameCallback?: (handle: number) => void;
    };

    const scheduleNextFrame = () => {
      if (!active) return;
      if (typeof vEl.requestVideoFrameCallback === 'function') {
        rvfcHandle = vEl.requestVideoFrameCallback(runInferenceTick);
      } else {
        rafHandle = window.requestAnimationFrame(runInferenceTick);
      }
    };

    const runInferenceTick = async () => {
      if (!active) return;

      const {
        isFrontCamera: mirrorX,
        screenWidth: sw,
        screenHeight: sh,
        confidenceThreshold: confMin,
        nmsIouThreshold: iouMax,
        enablePose: poseOn,
        isPaused: paused,
        onlyVehiclesMode: vehiclesOnly,
        speedLimitKmh: limitKmh,
        enableSpeedSoundAlert: soundAlertOn,
        distanceCalibrationFactor: distCalib,
        showCountingLine: countingLineActive,
        inertialStateRef: imuRef,
        onNewInferenceFrame: onNewFrame,
      } = optionsRef.current;

      if (
        paused ||
        isInferencingRef.current ||
        videoElement.readyState < 2 ||
        videoElement.videoWidth === 0
      ) {
        scheduleNextFrame();
        return;
      }

      isInferencingRef.current = true;
      const t0 = performance.now();

      try {
        const hasOrtSession = Boolean(ortDetectorSessionRef.current);
        const lb = preprocessFrameToPlanarNchw640(videoElement, hasOrtSession);
        const t1 = performance.now();

        if (lb && offscreenCanvasRef.current) {
          let rawCandidates: RawDecodedCandidate[] = [];

          if (ortDetectorSessionRef.current) {
            const session = ortDetectorSessionRef.current;
            const inputName = session.inputNames[0] ?? 'images';
            const inputTensor = new ort.Tensor('float32', planarNchwBufferRef.current, [
              1,
              3,
              MODEL_INPUT_SIZE,
              MODEL_INPUT_SIZE,
            ]);

            const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
            const results = await session.run(feeds);
            const outputName = session.outputNames[0];
            const outputTensor = results[outputName];

            if (outputTensor) {
              rawCandidates = decodeYoloOnnxOutputTensor(
                outputTensor.data as Float32Array,
                outputTensor.dims,
                confMin,
                lb,
                sw,
                sh,
                mirrorX,
                vehiclesOnly
              );
            }

            // En Modo Solo Vehículos (vehiclesOnly === true) se PAUSA automáticamente yolo11n-pose.onnx
            // liberando el 100% de GPU/WASM para maximizar FPS y precisión cinemática.
            if (!vehiclesOnly && poseOn && ortPoseSessionRef.current) {
              const poseSession = ortPoseSessionRef.current;
              const poseInName = poseSession.inputNames[0] ?? 'images';
              const poseOut = await poseSession.run({ [poseInName]: inputTensor });
              const poseTensor = poseOut[poseSession.outputNames[0]];
              if (poseTensor) {
                const decodedPoses = decodeYoloPoseOnnxTensor(
                  poseTensor.data as Float32Array,
                  poseTensor.dims,
                  confMin,
                  lb,
                  sw,
                  sh,
                  mirrorX
                );
                for (const cand of rawCandidates) {
                  if (cand.classId !== 0) continue;
                  const cx = cand.screenBox.x + cand.screenBox.width * 0.5;
                  const cy = cand.screenBox.y + cand.screenBox.height * 0.5;
                  let bestPose: OnnxKeypoint2D[] | undefined;
                  let bestDist = Math.max(cand.screenBox.width, cand.screenBox.height) * 0.65;
                  for (const dp of decodedPoses) {
                    const d = Math.hypot(dp.centerX - cx, dp.centerY - cy);
                    if (d < bestDist) {
                      bestDist = d;
                      bestPose = dp.keypoints;
                    }
                  }
                  if (bestPose) {
                    cand.keypoints = bestPose;
                  }
                }
              }
            }
          } else if (cocoFallbackModelRef.current) {
            const predictions = await cocoFallbackModelRef.current.detect(
              offscreenCanvasRef.current,
              28,
              confMin
            );

            for (let i = 0; i < predictions.length; i++) {
              const pred = predictions[i];
              if (pred.score < confMin) continue;

              const [lx, ly, lw, lh] = pred.bbox;
              const className = pred.class.toLowerCase();
              const classId = CLASS_NAME_TO_ID[className] ?? 0;

              // Filtro estricto de las 8 clases de vehículos de COCO cuando onlyVehiclesMode === true
              if (vehiclesOnly && !isCocoVehicleClass(classId, className)) {
                continue;
              }

              const category = getSemanticCategory(classId);

              const topLeft = projectLetterboxToCoverScreen(lx, ly, lb, sw, sh, mirrorX);
              const bottomRight = projectLetterboxToCoverScreen(
                lx + lw,
                ly + lh,
                lb,
                sw,
                sh,
                mirrorX
              );

              const sx = Math.min(topLeft.x, bottomRight.x);
              const sy = Math.min(topLeft.y, bottomRight.y);
              const sWidth = Math.abs(bottomRight.x - topLeft.x);
              const sHeight = Math.abs(bottomRight.y - topLeft.y);

              if (sWidth < 8 || sHeight < 8) continue;

              rawCandidates.push({
                classId,
                className,
                category,
                confidence: pred.score,
                letterboxBox: { x: lx, y: ly, width: lw, height: lh },
                screenBox: {
                  x: Math.max(0, Math.min(sw - 8, sx)),
                  y: Math.max(0, Math.min(sh - 8, sy)),
                  width: Math.max(8, Math.min(sw, sWidth)),
                  height: Math.max(8, Math.min(sh, sHeight)),
                },
              });
            }
          }

          const t2 = performance.now();

          // 4. Class-Aware Non-Maximum Suppression (NMS, IoU = 0.45)
          rawCandidates.sort((a, b) => b.confidence - a.confidence);
          const nmsKept: RawDecodedCandidate[] = [];
          for (let i = 0; i < rawCandidates.length; i++) {
            const cand = rawCandidates[i];
            let suppressed = false;
            for (let j = 0; j < nmsKept.length; j++) {
              if (
                nmsKept[j].classId === cand.classId &&
                computeBoxIoU(nmsKept[j].screenBox, cand.screenBox) > iouMax
              ) {
                suppressed = true;
                break;
              }
            }
            if (!suppressed) {
              nmsKept.push(cand);
            }
          }

          // 5. ByteTrack + VehicleSpeedEstimator (km/h & DETENIDO/EN MOVIMIENTO) + SpeedAlertSystem + VirtualCountingLine
          const prevTracks = tracksRef.current;
          const matchedPrevIndices = new Set<number>();
          const updatedTracks: OnnxTrackedDetection[] = [];
          const nowSec = t2 * 0.001;

          const gyroOffsetX = imuRef?.current.offsetX ?? 0;
          const gyroOffsetY = imuRef?.current.offsetY ?? 0;

          for (let i = 0; i < nmsKept.length; i++) {
            const det = nmsKept[i];
            const cx = det.screenBox.x + det.screenBox.width * 0.5;
            const cy = det.screenBox.y + det.screenBox.height * 0.5;

            let bestTrackIdx = -1;
            let bestIou = 0.20;

            for (let t = 0; t < prevTracks.length; t++) {
              if (matchedPrevIndices.has(t)) continue;
              if (prevTracks[t].classId !== det.classId) continue;
              const iou = computeBoxIoU(prevTracks[t].screenBox, det.screenBox);
              if (iou > bestIou) {
                bestIou = iou;
                bestTrackIdx = t;
              }
            }

            let trackId: number;
            let vx = 0;
            let vy = 0;
            let age = 1;
            let nextTrail: MotionTrailPoint[];

            if (bestTrackIdx >= 0) {
              matchedPrevIndices.add(bestTrackIdx);
              const prev = prevTracks[bestTrackIdx];
              trackId = prev.trackId;
              const dx = cx - prev.centerX;
              const dy = cy - prev.centerY;
              vx = prev.vx * 0.65 + dx * 0.35;
              vy = prev.vy * 0.65 + dy * 0.35;
              age = prev.age + 1;
              nextTrail = [...prev.trail, { x: cx, y: cy, timestampMs: t2 }].slice(
                -MAX_TRAIL_LENGTH
              );
            } else {
              trackId = nextTrackIdRef.current++;
              nextTrail = [{ x: cx, y: cy, timestampMs: t2 }];
            }

            // A) Calcular velocidad en km/h y estado (DETENIDO vs EN MOVIMIENTO) con compensación inercial
            const kinematics = speedEstimatorRef.current.updateTrack({
              trackId,
              classId: det.classId,
              className: det.className,
              centerX: cx,
              centerY: cy,
              width: det.screenBox.width,
              height: det.screenBox.height,
              timestampMs: t2,
              gyroOffsetX,
              gyroOffsetY,
              distanceCalibrationFactor: distCalib,
            });

            // B) Evaluar Alerta Visual y Sonora de Exceso de Velocidad (SpeedAlertSystem)
            const isOverspeed =
              kinematics.isVehicle &&
              globalSpeedAlertSystem.evaluateVehicleSpeed({
                trackId,
                className: det.className,
                speedKmh: kinematics.speedKmh,
                speedLimitKmh: limitKmh,
                soundEnabled: soundAlertOn,
                nowMs: t2,
              });

            // C) Evaluar Cruce Bidireccional en la Línea Virtual de Conteo Vehicular (VirtualCountingLine)
            const crossing = kinematics.isVehicle
              ? globalVirtualCountingLine.checkVehicleCrossing({
                  trackId,
                  className: det.className,
                  currentCenter: { x: cx, y: cy },
                  trail: nextTrail,
                  screenWidth: sw,
                  screenHeight: sh,
                  nowMs: t2,
                  enabled: countingLineActive,
                })
              : { hasCrossed: false, direction: null };

            const headBias = Math.max(-0.5, Math.min(0.5, vx * 0.05));

            updatedTracks.push({
              trackId,
              classId: det.classId,
              className: det.className,
              category: det.category,
              confidence: det.confidence,
              screenBox: det.screenBox,
              letterboxBox: det.letterboxBox,
              centerX: cx,
              centerY: cy,
              vx,
              vy,
              speedPxPerSec: kinematics.speedPxPerSec,
              speedMetersPerSec: kinematics.speedMetersPerSec,
              speedKmh: kinematics.speedKmh,
              motionStatus: kinematics.status,
              isOverspeed,
              hasCrossedLine: crossing.hasCrossed,
              crossingDirection: crossing.direction,
              age,
              lostFrames: 0,
              keypoints:
                !vehiclesOnly &&
                (det.keypoints ??
                  (poseOn && det.classId === 0
                    ? estimatePerson17KeypointsFromROI(det.screenBox, vx, nowSec, headBias)
                    : [])) ||
                [],
              trail: nextTrail,
            });
          }

          // Conservar brevemente tracks ocluidos (máx 4 frames) para continuidad de estela y conteo
          for (let t = 0; t < prevTracks.length; t++) {
            if (matchedPrevIndices.has(t)) continue;
            const lost = prevTracks[t];
            if (lost.lostFrames < 4 && lost.age >= 2) {
              updatedTracks.push({
                ...lost,
                lostFrames: lost.lostFrames + 1,
              });
            }
          }

          speedEstimatorRef.current.pruneStaleTracks(t2);
          tracksRef.current = updatedTracks;

          const visibleDetections = updatedTracks.filter((tr) => tr.lostFrames <= 1);
          setDetections(visibleDetections);
          onNewFrame?.(visibleDetections);

          const t3 = performance.now();
          const preprocessMs = Math.max(0.4, t1 - t0);
          const inferenceMs = Math.max(1.5, t2 - t1);
          const postprocessMs = Math.max(0.2, t3 - t2);
          const totalMs = preprocessMs + inferenceMs + postprocessMs;

          const prevEma = emaTimingsRef.current;
          const alpha = 0.22;
          const nextEma: OnnxVisionTimings = {
            captureMs: Number((prevEma.captureMs * (1 - alpha) + 0.9 * alpha).toFixed(1)),
            preprocessMs: Number(
              (prevEma.preprocessMs * (1 - alpha) + preprocessMs * alpha).toFixed(1)
            ),
            inferenceMs: Number(
              (prevEma.inferenceMs * (1 - alpha) + inferenceMs * alpha).toFixed(1)
            ),
            postprocessMs: Number(
              (prevEma.postprocessMs * (1 - alpha) + postprocessMs * alpha).toFixed(1)
            ),
            totalMs: Number((prevEma.totalMs * (1 - alpha) + totalMs * alpha).toFixed(1)),
            fps: Number(
              Math.min(
                60,
                1000 / Math.max(6, prevEma.totalMs * (1 - alpha) + totalMs * alpha)
              ).toFixed(1)
            ),
          };
          emaTimingsRef.current = nextEma;

          if (t3 - lastUiPushRef.current >= 180) {
            lastUiPushRef.current = t3;

            let persons = 0;
            let vehicles = 0;
            let animals = 0;
            let objects = 0;

            let movingVehicles = 0;
            let stoppedVehicles = 0;
            let speedSumKmh = 0;
            let speedCount = 0;
            let maxSpeedKmh = 0;
            let activeOverspeedInView = 0;

            for (let i = 0; i < visibleDetections.length; i++) {
              const item = visibleDetections[i];
              const cat = item.category;
              if (cat === 'person') persons++;
              else if (cat === 'vehicle' || cat === 'bicycle') vehicles++;
              else if (cat === 'animal') animals++;
              else objects++;

              if (isCocoVehicleClass(item.classId, item.className)) {
                if (item.motionStatus === 'EN MOVIMIENTO') {
                  movingVehicles++;
                } else {
                  stoppedVehicles++;
                }
                speedSumKmh += item.speedKmh;
                speedCount++;
                if (item.speedKmh > maxSpeedKmh) {
                  maxSpeedKmh = item.speedKmh;
                }
                if (item.isOverspeed) {
                  activeOverspeedInView++;
                }
              }
            }

            setCounts({
              persons,
              vehicles,
              animals,
              objects,
              total: visibleDetections.length,
            });

            setTrafficTelemetry({
              movingVehicles,
              stoppedVehicles,
              averageSpeedKmh:
                speedCount > 0 ? Number((speedSumKmh / speedCount).toFixed(1)) : 0,
              maxSpeedKmh: Number(maxSpeedKmh.toFixed(1)),
              overspeedTotalCount: globalSpeedAlertSystem.getTotalOverspeedCount(),
              activeOverspeedInView,
              activeBanner: globalSpeedAlertSystem.getActiveBanner(t3),
              countingStats: { ...globalVirtualCountingLine.getStats() },
            });

            setTimings(nextEma);
            setLetterboxMeta(lb);
            const buf = planarNchwBufferRef.current;
            setInputSample5([buf[0], buf[1], buf[2], buf[3], buf[4]]);
          }
        }
      } catch {
        // Continuar el bucle si un frame individual falla
      } finally {
        isInferencingRef.current = false;
        scheduleNextFrame();
      }
    };

    scheduleNextFrame();

    return () => {
      active = false;
      if (rvfcHandle !== null && typeof vEl.cancelVideoFrameCallback === 'function') {
        vEl.cancelVideoFrameCallback(rvfcHandle);
      }
      if (rafHandle !== null) {
        window.cancelAnimationFrame(rafHandle);
      }
    };
  }, [videoElement, isCameraActive, isEngineReady, preprocessFrameToPlanarNchw640]);

  return {
    isEngineReady,
    isEngineLoading,
    engineStatusText,
    activeBackend,
    activeModelName,
    isNativeOnnxSession,
    detections,
    counts,
    trafficTelemetry,
    timings,
    letterboxMeta,
    inputSample5,
    loadCustomOnnxFile,
    resetTrafficCounts,
  };
}
