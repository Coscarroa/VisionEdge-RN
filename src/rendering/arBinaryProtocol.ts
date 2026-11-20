/**
 * Protocolo Binario Zero-Copy para Transferencia de Detecciones AR (Main Thread -> Web Worker)
 * (src/rendering/arBinaryProtocol.ts)
 *
 * - Evita serializar grafos JSON complejos en cada fotograma de inferencia ONNX.
 * - Empaqueta todas las detecciones activas (cajas, velocidad km/h, banderas de estado,
 *   17 keypoints articulares y 15 puntos de estela de movimiento) en un único Float32Array plano.
 * - Transfiere el ArrayBuffer subyacente mediante Transferable Objects:
 *   worker.postMessage({ type: 'FRAME_DATA', buffer }, [buffer]) (< 0.05 ms Zero-Copy).
 */

import { SemanticCategory } from '../utils/visionPostProcessor';
import { VehicleMotionStatus } from '../engine/cvTypes';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { ProximityHazardLink } from '../modules/HumanSafetyAnalytics';
import { RoiSnapGuide, RoiVerticesNorm } from '../modules/SpatialZonesROI';
import {
  CrossingDirection,
  TrafficCountBreakdown,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';

/**
 * Cantidad fija de valores Float32 por cada detección empaquetada (Stride = 99 floats = 396 bytes):
 * [0]  trackId
 * [1]  classId
 * [2]  categoryCode (0=person, 1=vehicle, 2=animal, 3=object)
 * [3]  x (screenBox.x)
 * [4]  y (screenBox.y)
 * [5]  w (screenBox.width)
 * [6]  h (screenBox.height)
 * [7]  confidence (0..1)
 * [8]  speedKmh
 * [9]  speedMetersPerSec
 * [10] speedPxPerSec
 * [11] stateCode (0=DETENIDO, 1=EN MOVIMIENTO, 2=EXCESO)
 * [12] alertFlagsBitmask:
 *      bit 0 (1):  isOverspeed
 *      bit 1 (2):  isRoiViolation
 *      bit 2 (4):  isFallDetected
 *      bit 3 (8):  hasProximityRisk
 *      bit 4 (16): insideRoi
 *      bit 5 (32): hasCrossedLine
 * [13] roiDwellSeconds
 * [14] fallDurationSeconds
 * [15..65] 17 Keypoints * 3 floats (x, y, score) = 51 floats
 * [66] trailCount (0..15)
 * [67..96] 15 Trail Points * 2 floats (x, y) = 30 floats
 * [97] vx
 * [98] vy
 */
export const DETECTION_FLOAT32_STRIDE = 99;
export const MAX_PACKED_TRAIL_POINTS = 15;
export const PACKED_KEYPOINTS_COUNT = 17;

export const ALERT_FLAG_OVERSPEED = 1 << 0;
export const ALERT_FLAG_ROI_VIOLATION = 1 << 1;
export const ALERT_FLAG_FALL_DETECTED = 1 << 2;
export const ALERT_FLAG_PROXIMITY_RISK = 1 << 3;
export const ALERT_FLAG_INSIDE_ROI = 1 << 4;
export const ALERT_FLAG_CROSSED_LINE = 1 << 5;

export interface ARDetectionStringMeta {
  className: string;
  licensePlate: string | null;
  crossingDirection: CrossingDirection | null;
}

export interface ARFrameMetadataPayload {
  timestampMs: number;
  detectionCount: number;
  /** Mapa ligero indexado por posición [0..detectionCount-1] con cadenas cortas */
  stringsByIndex: ARDetectionStringMeta[];
  countingStats: TrafficCountBreakdown;
  activeRoiViolationInView: boolean;
  proximityLinks: ProximityHazardLink[];
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
}

export interface ARPointerDragPayload {
  active: boolean;
  pointerX: number;
  pointerY: number;
  activeDragHandle: 'a' | 'b' | null;
  activeRoiVertexIndex: 0 | 1 | 2 | 3 | null;
  lineEndpointsNorm: VirtualLineEndpointsNorm;
  roiVerticesNorm?: RoiVerticesNorm;
  roiSnapGuides: RoiSnapGuide[];
}

function encodeCategoryCode(cat: SemanticCategory): number {
  switch (cat) {
    case 'person':
      return 0;
    case 'vehicle':
      return 1;
    case 'animal':
      return 2;
    default:
      return 3;
  }
}

function decodeCategoryCode(code: number): SemanticCategory {
  switch (Math.round(code)) {
    case 0:
      return 'person';
    case 1:
      return 'vehicle';
    case 2:
      return 'animal';
    default:
      return 'object';
  }
}

function encodeMotionStateCode(
  status: VehicleMotionStatus,
  isOverspeed: boolean
): number {
  if (isOverspeed) return 2;
  return status === 'EN MOVIMIENTO' ? 1 : 0;
}

function decodeMotionStateCode(code: number): VehicleMotionStatus {
  return Math.round(code) >= 1 ? 'EN MOVIMIENTO' : 'DETENIDO';
}

/**
 * Codifica el arreglo de detecciones activas en un ArrayBuffer plano (Float32Array)
 * listo para transferirse mediante Transferable Objects (Zero-Copy) al Web Worker.
 */
export function encodeDetectionsToBinaryBuffer(params: {
  detections: OnnxTrackedDetection[];
  countingStats: TrafficCountBreakdown;
  activeRoiViolationInView: boolean;
  proximityLinks: ProximityHazardLink[];
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  nowMs?: number;
}): {
  buffer: ArrayBuffer;
  meta: ARFrameMetadataPayload;
} {
  const {
    detections,
    countingStats,
    activeRoiViolationInView,
    proximityLinks,
    showHeatmapOverlay,
    heatmapPoints,
    nowMs = performance.now(),
  } = params;

  const count = detections.length;
  const floatArray = new Float32Array(count * DETECTION_FLOAT32_STRIDE);
  const stringsByIndex: ARDetectionStringMeta[] = new Array(count);

  for (let i = 0; i < count; i++) {
    const det = detections[i];
    const base = i * DETECTION_FLOAT32_STRIDE;

    let flags = 0;
    if (det.isOverspeed) flags |= ALERT_FLAG_OVERSPEED;
    if (det.isRoiViolation) flags |= ALERT_FLAG_ROI_VIOLATION;
    if (det.isFallDetected) flags |= ALERT_FLAG_FALL_DETECTED;
    if (det.hasProximityRisk) flags |= ALERT_FLAG_PROXIMITY_RISK;
    if (det.insideRoi) flags |= ALERT_FLAG_INSIDE_ROI;
    if (det.hasCrossedLine) flags |= ALERT_FLAG_CROSSED_LINE;

    floatArray[base + 0] = det.trackId;
    floatArray[base + 1] = det.classId;
    floatArray[base + 2] = encodeCategoryCode(det.category);
    floatArray[base + 3] = det.screenBox.x;
    floatArray[base + 4] = det.screenBox.y;
    floatArray[base + 5] = det.screenBox.width;
    floatArray[base + 6] = det.screenBox.height;
    floatArray[base + 7] = det.confidence;
    floatArray[base + 8] = det.speedKmh ?? 0;
    floatArray[base + 9] = det.speedMetersPerSec ?? 0;
    floatArray[base + 10] = det.speedPxPerSec ?? 0;
    floatArray[base + 11] = encodeMotionStateCode(
      det.motionStatus,
      det.isOverspeed
    );
    floatArray[base + 12] = flags;
    floatArray[base + 13] = det.roiDwellSeconds ?? 0;
    floatArray[base + 14] = det.fallDurationSeconds ?? 0;

    // Empaquetar 17 keypoints (índices 15..65)
    const kps = det.keypoints;
    const kpCount = Math.min(PACKED_KEYPOINTS_COUNT, kps ? kps.length : 0);
    for (let k = 0; k < PACKED_KEYPOINTS_COUNT; k++) {
      const kOffset = base + 15 + k * 3;
      if (k < kpCount) {
        floatArray[kOffset + 0] = kps[k].x;
        floatArray[kOffset + 1] = kps[k].y;
        floatArray[kOffset + 2] = kps[k].score;
      } else {
        floatArray[kOffset + 0] = 0;
        floatArray[kOffset + 1] = 0;
        floatArray[kOffset + 2] = 0;
      }
    }

    // Empaquetar hasta 15 puntos de estela (índices 66..96)
    const trail = det.trail ?? [];
    const sliceStart = Math.max(0, trail.length - MAX_PACKED_TRAIL_POINTS);
    const tCount = Math.min(MAX_PACKED_TRAIL_POINTS, trail.length - sliceStart);
    floatArray[base + 66] = tCount;

    for (let t = 0; t < MAX_PACKED_TRAIL_POINTS; t++) {
      const tOffset = base + 67 + t * 2;
      if (t < tCount) {
        const pt = trail[sliceStart + t];
        floatArray[tOffset + 0] = pt.x;
        floatArray[tOffset + 1] = pt.y;
      } else {
        floatArray[tOffset + 0] = 0;
        floatArray[tOffset + 1] = 0;
      }
    }

    floatArray[base + 97] = det.vx ?? 0;
    floatArray[base + 98] = det.vy ?? 0;

    stringsByIndex[i] = {
      className: det.className,
      licensePlate: det.licensePlate ?? null,
      crossingDirection: det.crossingDirection ?? null,
    };
  }

  return {
    buffer: floatArray.buffer,
    meta: {
      timestampMs: nowMs,
      detectionCount: count,
      stringsByIndex,
      countingStats,
      activeRoiViolationInView,
      proximityLinks,
      showHeatmapOverlay,
      heatmapPoints,
    },
  };
}

/**
 * Decodifica en el Web Worker el ArrayBuffer transferido sin copias adicionales,
 * reconstruyendo estructuras tipadas listas para interpolación LERP a 60 FPS.
 */
export function decodeDetectionsFromBinaryBuffer(
  buffer: ArrayBuffer,
  meta: ARFrameMetadataPayload
): OnnxTrackedDetection[] {
  const f32 = new Float32Array(buffer);
  const count = Math.min(
    meta.detectionCount,
    Math.floor(f32.length / DETECTION_FLOAT32_STRIDE)
  );
  const result: OnnxTrackedDetection[] = new Array(count);

  for (let i = 0; i < count; i++) {
    const base = i * DETECTION_FLOAT32_STRIDE;
    const strMeta = meta.stringsByIndex[i] ?? {
      className: 'object',
      licensePlate: null,
      crossingDirection: null,
    };

    const trackId = Math.round(f32[base + 0]);
    const classId = Math.round(f32[base + 1]);
    const category = decodeCategoryCode(f32[base + 2]);
    const x = f32[base + 3];
    const y = f32[base + 4];
    const width = f32[base + 5];
    const height = f32[base + 6];
    const confidence = f32[base + 7];
    const speedKmh = f32[base + 8];
    const speedMetersPerSec = f32[base + 9];
    const speedPxPerSec = f32[base + 10];
    const motionStatus = decodeMotionStateCode(f32[base + 11]);
    const flags = Math.round(f32[base + 12]);
    const roiDwellSeconds = f32[base + 13];
    const fallDurationSeconds = f32[base + 14];

    const isOverspeed = (flags & ALERT_FLAG_OVERSPEED) !== 0;
    const isRoiViolation = (flags & ALERT_FLAG_ROI_VIOLATION) !== 0;
    const isFallDetected = (flags & ALERT_FLAG_FALL_DETECTED) !== 0;
    const hasProximityRisk = (flags & ALERT_FLAG_PROXIMITY_RISK) !== 0;
    const insideRoi = (flags & ALERT_FLAG_INSIDE_ROI) !== 0;
    const hasCrossedLine = (flags & ALERT_FLAG_CROSSED_LINE) !== 0;

    // Desempaquetar 17 keypoints si el primero o alguno tiene score > 0
    const keypoints: OnnxKeypoint2D[] = [];
    let hasValidKeypoint = false;
    for (let k = 0; k < PACKED_KEYPOINTS_COUNT; k++) {
      if (f32[base + 15 + k * 3 + 2] > 0.01) {
        hasValidKeypoint = true;
        break;
      }
    }

    if (hasValidKeypoint) {
      for (let k = 0; k < PACKED_KEYPOINTS_COUNT; k++) {
        const kOffset = base + 15 + k * 3;
        keypoints.push({
          index: k,
          name: '',
          x: f32[kOffset + 0],
          y: f32[kOffset + 1],
          score: f32[kOffset + 2],
        });
      }
    }

    const tCount = Math.min(
      MAX_PACKED_TRAIL_POINTS,
      Math.max(0, Math.round(f32[base + 66]))
    );
    const trail: Array<{ x: number; y: number; timestampMs: number }> =
      new Array(tCount);
    for (let t = 0; t < tCount; t++) {
      const tOffset = base + 67 + t * 2;
      trail[t] = {
        x: f32[tOffset + 0],
        y: f32[tOffset + 1],
        timestampMs: meta.timestampMs,
      };
    }

    const vx = f32[base + 97];
    const vy = f32[base + 98];

    result[i] = {
      trackId,
      classId,
      className: strMeta.className,
      category,
      confidence,
      screenBox: { x, y, width, height },
      letterboxBox: { x, y, width, height },
      centerX: x + width * 0.5,
      centerY: y + height * 0.5,
      vx,
      vy,
      speedPxPerSec,
      speedMetersPerSec,
      speedKmh,
      motionStatus,
      isOverspeed,
      hasCrossedLine,
      crossingDirection: strMeta.crossingDirection,
      insideRoi,
      roiDwellSeconds,
      isRoiViolation,
      licensePlate: strMeta.licensePlate,
      isFallDetected,
      fallDurationSeconds,
      hasProximityRisk,
      age: 1,
      lostFrames: 0,
      trail,
      keypoints,
    };
  }

  return result;
}
