/**
 * Módulo 2: Inteligencia Espacial — Zonas Poligonales ROI (Ray-Casting),
 * Ajuste Magnético Geométrico (Snapping a Carriles/Arquitectura) y ALPR Ligero (tesseract.js)
 * (src/modules/SpatialZonesROI.ts)
 *
 * - Polígono ROI interactivo de 4 vértices arrastrables directamente sobre el visor AR.
 * - Motor de Ajuste Magnético (Vertex Snapping):
 *   1. Alineación Ortogonal Horizontal/Vertical con vértices hermanos (V1..V4) y ejes centrales.
 *   2. Alineación a Rayos de Punto de Fuga (Perspectiva de Carril Vial) y Simetría de Calzada.
 *   3. Alineación Paralela al borde opuesto del cuadrilátero.
 *   4. Acoplamiento magnético directo a los nodos A/B o segmento de la Línea Virtual de Aforo.
 * - Algoritmo matemático Ray-Casting (isPointInPolygon) para evaluar si el centroide de un
 *   vehículo se encuentra dentro del polígono de exclusión / estacionamiento prohibido.
 * - ALPR Ligero con tesseract.js en Web Worker sobre el recorte del tercio inferior del vehículo.
 */

import { createWorker, Worker as TesseractWorker } from 'tesseract.js';
import { VehicleMotionStatus } from '../engine/cvTypes';
import {
  Point2D,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';

export type RoiVerticesNorm = [Point2D, Point2D, Point2D, Point2D];

export type RoiSnapKind =
  | 'ORTHOGONAL_H'
  | 'ORTHOGONAL_V'
  | 'LANE_PERSPECTIVE'
  | 'LANE_SYMMETRY'
  | 'PARALLEL_EDGE'
  | 'TRIPWIRE_LOCK'
  | 'SCREEN_GRID';

export interface RoiSnapGuide {
  kind: RoiSnapKind;
  /** Coordenadas normalizadas (0..1) de la línea guía láser */
  x1Norm: number;
  y1Norm: number;
  x2Norm: number;
  y2Norm: number;
  /** Punto exacto de anclaje normalizado */
  snapXNorm: number;
  snapYNorm: number;
  label: string;
  color: string;
}

export type RoiPolygonPreset =
  | 'lane-trapezoid'
  | 'architectural-rect'
  | 'curb-dropoff';

export interface RoiTrackEvaluation {
  insideRoi: boolean;
  dwellSeconds: number;
  isViolation: boolean;
  justTriggeredViolation: boolean;
}

/**
 * Algoritmo matemático Ray-Casting (Even-Odd Rule) para determinar si un punto 2D
 * se encuentra dentro de un polígono cerrado de N vértices.
 */
export function isPointInPolygon(point: Point2D, polygon: ReadonlyArray<Point2D>): boolean {
  if (polygon.length < 3) return false;
  let inside = false;
  const x = point.x;
  const y = point.y;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-10) + xi;
    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
}

/**
 * Proyecta un punto P sobre la recta infinita que pasa por A y B.
 */
function projectPointOntoLine(
  p: Point2D,
  a: Point2D,
  b: Point2D,
  clampToSegment = false
): { projected: Point2D; distance: number; t: number } {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-8) {
    return {
      projected: { x: a.x, y: a.y },
      distance: Math.hypot(p.x - a.x, p.y - a.y),
      t: 0,
    };
  }
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
  if (clampToSegment) {
    t = Math.max(0, Math.min(1, t));
  }
  const projX = a.x + t * dx;
  const projY = a.y + t * dy;
  return {
    projected: { x: projX, y: projY },
    distance: Math.hypot(p.x - projX, p.y - projY),
    t,
  };
}

export class SpatialZonesROIManager {
  /** 4 vértices normalizados (0..1) en orden horario: V1 (Sup-Izq), V2 (Sup-Der), V3 (Inf-Der), V4 (Inf-Izq) */
  private verticesNorm: RoiVerticesNorm = [
    { x: 0.24, y: 0.36 },
    { x: 0.76, y: 0.36 },
    { x: 0.84, y: 0.74 },
    { x: 0.16, y: 0.74 },
  ];

  private activeSnapGuides: RoiSnapGuide[] = [];
  private lastSnapSignature = '';

  private readonly stoppedInsideSinceByTrack = new Map<number, number>();
  private readonly lastSeenInsideByTrack = new Map<number, number>();
  private readonly violationCreditedTracks = new Set<number>();
  private totalRoiViolations = 0;

  // Estado ALPR (Tesseract.js Web Worker)
  private tesseractWorker: TesseractWorker | null = null;
  private isWorkerInitializing = false;
  private isOcrBusy = false;
  private alprCanvas: HTMLCanvasElement | null = null;
  private readonly plateByTrackId = new Map<number, string>();
  private readonly lastAlprAttemptByTrackId = new Map<number, number>();

  public getVerticesNorm(): RoiVerticesNorm {
    return this.verticesNorm;
  }

  public getActiveSnapGuides(): RoiSnapGuide[] {
    return this.activeSnapGuides;
  }

  public clearActiveSnapGuides(): void {
    this.activeSnapGuides = [];
    this.lastSnapSignature = '';
  }

  public getVerticesPixels(screenWidth: number, screenHeight: number): Point2D[] {
    return this.verticesNorm.map((v) => ({
      x: v.x * screenWidth,
      y: v.y * screenHeight,
    }));
  }

  public setVertexNorm(index: 0 | 1 | 2 | 3, xNorm: number, yNorm: number): RoiVerticesNorm {
    const cx = Math.max(0.03, Math.min(0.97, xNorm));
    const cy = Math.max(0.08, Math.min(0.92, yNorm));
    const next: RoiVerticesNorm = [
      { ...this.verticesNorm[0] },
      { ...this.verticesNorm[1] },
      { ...this.verticesNorm[2] },
      { ...this.verticesNorm[3] },
    ];
    next[index] = { x: cx, y: cy };
    this.verticesNorm = next;
    return this.verticesNorm;
  }

  /**
   * Actualiza un vértice del Polígono ROI aplicando Ajuste Magnético Inteligente (Snapping)
   * para facilitar la alineación exacta con carriles viales en perspectiva, bordes arquitectónicos
   * ortogonales/paralelos y la Línea Virtual de Aforo.
   */
  public setVertexNormWithSnapping(params: {
    index: 0 | 1 | 2 | 3;
    rawXNorm: number;
    rawYNorm: number;
    screenWidth: number;
    screenHeight: number;
    enableSnapping?: boolean;
    countingLineNorm?: VirtualLineEndpointsNorm;
    snapThresholdPx?: number;
  }): { verticesNorm: RoiVerticesNorm; snapGuides: RoiSnapGuide[] } {
    const {
      index,
      rawXNorm,
      rawYNorm,
      screenWidth,
      screenHeight,
      enableSnapping = true,
      countingLineNorm,
      snapThresholdPx = 15,
    } = params;

    let candidateX = Math.max(0.03, Math.min(0.97, rawXNorm));
    let candidateY = Math.max(0.08, Math.min(0.92, rawYNorm));

    if (!enableSnapping || screenWidth <= 0 || screenHeight <= 0) {
      this.activeSnapGuides = [];
      return {
        verticesNorm: this.setVertexNorm(index, candidateX, candidateY),
        snapGuides: [],
      };
    }

    const guides: RoiSnapGuide[] = [];
    const threshXNorm = snapThresholdPx / screenWidth;
    const threshYNorm = snapThresholdPx / screenHeight;

    let lockedX = false;
    let lockedY = false;

    // 1. Acoplamiento magnético a la Línea Virtual de Aforo (Nodos A/B o segmento A-B)
    if (countingLineNorm) {
      const aPx = {
        x: countingLineNorm.x1 * screenWidth,
        y: countingLineNorm.y1 * screenHeight,
      };
      const bPx = {
        x: countingLineNorm.x2 * screenWidth,
        y: countingLineNorm.y2 * screenHeight,
      };
      const curPx = { x: candidateX * screenWidth, y: candidateY * screenHeight };

      const distNodeA = Math.hypot(curPx.x - aPx.x, curPx.y - aPx.y);
      const distNodeB = Math.hypot(curPx.x - bPx.x, curPx.y - bPx.y);

      if (distNodeA <= snapThresholdPx * 1.15) {
        candidateX = countingLineNorm.x1;
        candidateY = countingLineNorm.y1;
        lockedX = true;
        lockedY = true;
        guides.push({
          kind: 'TRIPWIRE_LOCK',
          x1Norm: countingLineNorm.x1,
          y1Norm: countingLineNorm.y1,
          x2Norm: countingLineNorm.x2,
          y2Norm: countingLineNorm.y2,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: '🧲 SNAP: NODO A (LÍNEA AFORO)',
          color: '#38BDF8',
        });
      } else if (distNodeB <= snapThresholdPx * 1.15) {
        candidateX = countingLineNorm.x2;
        candidateY = countingLineNorm.y2;
        lockedX = true;
        lockedY = true;
        guides.push({
          kind: 'TRIPWIRE_LOCK',
          x1Norm: countingLineNorm.x1,
          y1Norm: countingLineNorm.y1,
          x2Norm: countingLineNorm.x2,
          y2Norm: countingLineNorm.y2,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: '🧲 SNAP: NODO B (LÍNEA AFORO)',
          color: '#38BDF8',
        });
      } else {
        const segProj = projectPointOntoLine(curPx, aPx, bPx, true);
        if (segProj.distance <= snapThresholdPx * 0.9) {
          candidateX = segProj.projected.x / screenWidth;
          candidateY = segProj.projected.y / screenHeight;
          lockedX = true;
          lockedY = true;
          guides.push({
            kind: 'TRIPWIRE_LOCK',
            x1Norm: countingLineNorm.x1,
            y1Norm: countingLineNorm.y1,
            x2Norm: countingLineNorm.x2,
            y2Norm: countingLineNorm.y2,
            snapXNorm: candidateX,
            snapYNorm: candidateY,
            label: '🧲 SNAP: BORDE LÍNEA DE AFORO',
            color: '#38BDF8',
          });
        }
      }
    }

    // 2. Alineación Ortogonal (Horizontal y Vertical) con los otros 3 vértices hermanos
    if (!lockedY) {
      let bestDy = threshYNorm;
      let bestSiblingIdx: number | null = null;
      for (let i = 0; i < 4; i++) {
        if (i === index) continue;
        const dy = Math.abs(candidateY - this.verticesNorm[i].y);
        if (dy <= bestDy) {
          bestDy = dy;
          bestSiblingIdx = i;
        }
      }
      if (bestSiblingIdx !== null) {
        const targetV = this.verticesNorm[bestSiblingIdx];
        candidateY = targetV.y;
        lockedY = true;
        guides.push({
          kind: 'ORTHOGONAL_H',
          x1Norm: 0.04,
          y1Norm: candidateY,
          x2Norm: 0.96,
          y2Norm: candidateY,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: `🧲 SNAP H: ALINEADO CON V${bestSiblingIdx + 1}`,
          color: '#22D3EE',
        });
      }
    }

    if (!lockedX) {
      let bestDx = threshXNorm;
      let bestSiblingIdx: number | null = null;
      for (let i = 0; i < 4; i++) {
        if (i === index) continue;
        const dx = Math.abs(candidateX - this.verticesNorm[i].x);
        if (dx <= bestDx) {
          bestDx = dx;
          bestSiblingIdx = i;
        }
      }
      if (bestSiblingIdx !== null) {
        const targetV = this.verticesNorm[bestSiblingIdx];
        candidateX = targetV.x;
        lockedX = true;
        guides.push({
          kind: 'ORTHOGONAL_V',
          x1Norm: candidateX,
          y1Norm: 0.08,
          x2Norm: candidateX,
          y2Norm: 0.92,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: `🧲 SNAP V: ALINEADO CON V${bestSiblingIdx + 1}`,
          color: '#22D3EE',
        });
      }
    }

    // 3. Simetría de Carril Vial respecto al eje central óptico (x = 1 - hermanoHorizontal.x)
    const horizontalPairIndex =
      index === 0 ? 1 : index === 1 ? 0 : index === 2 ? 3 : 2;
    const pairVertex = this.verticesNorm[horizontalPairIndex];
    const symmetricTargetX = 1.0 - pairVertex.x;
    if (!lockedX && Math.abs(candidateX - symmetricTargetX) <= threshXNorm * 0.95) {
      candidateX = symmetricTargetX;
      lockedX = true;
      guides.push({
        kind: 'LANE_SYMMETRY',
        x1Norm: 0.5,
        y1Norm: 0.14,
        x2Norm: 0.5,
        y2Norm: 0.88,
        snapXNorm: candidateX,
        snapYNorm: candidateY,
        label: `🧲 SNAP: SIMETRÍA DE CARRIL (V${horizontalPairIndex + 1})`,
        color: '#34D399',
      });
    }

    // 4. Rayo de Punto de Fuga (Perspectiva de Carril Vial Monocular)
    // El punto de fuga estándar del horizonte vial se ubica en VP(0.50, 0.22)
    if (!lockedX) {
      const vpPx = { x: 0.5 * screenWidth, y: 0.22 * screenHeight };
      // Pareja longitudinal del carril: V1(0)<->V4(3) carril izquierdo, V2(1)<->V3(2) carril derecho
      const laneAnchorIndex =
        index === 3 ? 0 : index === 0 ? 3 : index === 2 ? 1 : 2;
      const anchorNorm = this.verticesNorm[laneAnchorIndex];
      const anchorPx = {
        x: anchorNorm.x * screenWidth,
        y: anchorNorm.y * screenHeight,
      };
      const curPx = { x: candidateX * screenWidth, y: candidateY * screenHeight };

      const rayProj = projectPointOntoLine(curPx, vpPx, anchorPx, false);
      if (rayProj.distance <= snapThresholdPx * 0.95 && rayProj.t > 0.25) {
        candidateX = Math.max(0.03, Math.min(0.97, rayProj.projected.x / screenWidth));
        if (!lockedY) {
          candidateY = Math.max(0.08, Math.min(0.92, rayProj.projected.y / screenHeight));
        }
        lockedX = true;
        guides.push({
          kind: 'LANE_PERSPECTIVE',
          x1Norm: 0.5,
          y1Norm: 0.22,
          x2Norm: candidateX,
          y2Norm: candidateY,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: `🧲 SNAP: PERSPECTIVA CARRIL (VP→V${laneAnchorIndex + 1})`,
          color: '#10B981',
        });
      }
    }

    // 5. Alineación Paralela al Borde Opuesto (Geometría Arquitectónica)
    if (!lockedX && !lockedY) {
      // Si arrastramos index, el borde longitudinal opuesto es (oppA -> oppB)
      const [adjIdx, oppAIdx, oppBIdx] =
        index === 0
          ? [3, 1, 2]
          : index === 3
          ? [0, 2, 1]
          : index === 1
          ? [2, 0, 3]
          : [1, 3, 0];

      const adjPx = {
        x: this.verticesNorm[adjIdx].x * screenWidth,
        y: this.verticesNorm[adjIdx].y * screenHeight,
      };
      const oppVecX =
        (this.verticesNorm[oppAIdx].x - this.verticesNorm[oppBIdx].x) * screenWidth;
      const oppVecY =
        (this.verticesNorm[oppAIdx].y - this.verticesNorm[oppBIdx].y) * screenHeight;

      const parallelTargetPx = {
        x: adjPx.x + oppVecX,
        y: adjPx.y + oppVecY,
      };
      const curPx = { x: candidateX * screenWidth, y: candidateY * screenHeight };
      const parProj = projectPointOntoLine(curPx, adjPx, parallelTargetPx, false);

      if (parProj.distance <= snapThresholdPx * 0.85) {
        candidateX = Math.max(0.03, Math.min(0.97, parProj.projected.x / screenWidth));
        candidateY = Math.max(0.08, Math.min(0.92, parProj.projected.y / screenHeight));
        lockedX = true;
        lockedY = true;
        guides.push({
          kind: 'PARALLEL_EDGE',
          x1Norm: this.verticesNorm[adjIdx].x,
          y1Norm: this.verticesNorm[adjIdx].y,
          x2Norm: candidateX,
          y2Norm: candidateY,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: `🧲 SNAP: PARALELO A BORDE V${oppBIdx + 1}-V${oppAIdx + 1}`,
          color: '#A78BFA',
        });
      }
    }

    // 6. Rejilla Arquitectónica Suave (pasos del 5% de pantalla cuando se mueve lentamente)
    if (!lockedX && !lockedY) {
      const gridStep = 0.05;
      const nearestGridX = Math.round(candidateX / gridStep) * gridStep;
      const nearestGridY = Math.round(candidateY / gridStep) * gridStep;
      const gridThreshX = (snapThresholdPx * 0.42) / screenWidth;
      const gridThreshY = (snapThresholdPx * 0.42) / screenHeight;

      if (
        Math.abs(candidateX - nearestGridX) <= gridThreshX &&
        Math.abs(candidateY - nearestGridY) <= gridThreshY
      ) {
        candidateX = Number(nearestGridX.toFixed(2));
        candidateY = Number(nearestGridY.toFixed(2));
        guides.push({
          kind: 'SCREEN_GRID',
          x1Norm: Math.max(0.03, candidateX - 0.06),
          y1Norm: candidateY,
          x2Norm: Math.min(0.97, candidateX + 0.06),
          y2Norm: candidateY,
          snapXNorm: candidateX,
          snapYNorm: candidateY,
          label: `🧲 SNAP REJILLA (${Math.round(candidateX * 100)}%, ${Math.round(
            candidateY * 100
          )}%)`,
          color: '#94A3B8',
        });
      }
    }

    // Retroalimentación háptica sutil (8ms) cuando se engancha a una nueva guía magnética
    const newSignature = guides.map((g) => `${g.kind}:${g.label}`).join('|');
    if (
      newSignature.length > 0 &&
      newSignature !== this.lastSnapSignature &&
      typeof navigator !== 'undefined' &&
      typeof navigator.vibrate === 'function'
    ) {
      try {
        navigator.vibrate(8);
      } catch {
        // Ignorar si el navegador bloquea vibrate
      }
    }
    this.lastSnapSignature = newSignature;
    this.activeSnapGuides = guides;

    return {
      verticesNorm: this.setVertexNorm(index, candidateX, candidateY),
      snapGuides: guides,
    };
  }

  /**
   * Aplica una plantilla geométrica precalibrada para alinear el Polígono ROI
   * en un solo toque con carriles en perspectiva, fachadas ortogonales o bahías de acera.
   */
  public applyPolygonPreset(preset: RoiPolygonPreset): RoiVerticesNorm {
    this.clearActiveSnapGuides();
    if (preset === 'architectural-rect') {
      this.verticesNorm = [
        { x: 0.20, y: 0.30 },
        { x: 0.80, y: 0.30 },
        { x: 0.80, y: 0.75 },
        { x: 0.20, y: 0.75 },
      ];
    } else if (preset === 'curb-dropoff') {
      this.verticesNorm = [
        { x: 0.56, y: 0.34 },
        { x: 0.86, y: 0.34 },
        { x: 0.94, y: 0.80 },
        { x: 0.52, y: 0.80 },
      ];
    } else {
      // 'lane-trapezoid' (Perspectiva de carril central hacia el punto de fuga)
      this.verticesNorm = [
        { x: 0.28, y: 0.36 },
        { x: 0.72, y: 0.36 },
        { x: 0.86, y: 0.76 },
        { x: 0.14, y: 0.76 },
      ];
    }
    return this.verticesNorm;
  }

  public resetDefaultPolygon(): RoiVerticesNorm {
    return this.applyPolygonPreset('lane-trapezoid');
  }

  public getTotalRoiViolations(): number {
    return this.totalRoiViolations;
  }

  public getLicensePlate(trackId: number): string | null {
    return this.plateByTrackId.get(trackId) ?? null;
  }

  public getAllRecognizedPlates(): Array<{ trackId: number; plate: string }> {
    return Array.from(this.plateByTrackId.entries()).map(([trackId, plate]) => ({
      trackId,
      plate,
    }));
  }

  /**
   * Evalúa si un vehículo está dentro del Polígono ROI y acumula el tiempo en estado DETENIDO.
   */
  public evaluateVehicleInRoi(params: {
    enabled: boolean;
    trackId: number;
    centerX: number;
    centerY: number;
    motionStatus: VehicleMotionStatus;
    screenWidth: number;
    screenHeight: number;
    dwellThresholdSeconds: number;
    nowMs: number;
  }): RoiTrackEvaluation {
    const {
      enabled,
      trackId,
      centerX,
      centerY,
      motionStatus,
      screenWidth,
      screenHeight,
      dwellThresholdSeconds,
      nowMs,
    } = params;

    if (!enabled || screenWidth <= 0 || screenHeight <= 0) {
      return {
        insideRoi: false,
        dwellSeconds: 0,
        isViolation: false,
        justTriggeredViolation: false,
      };
    }

    const polyPx = this.getVerticesPixels(screenWidth, screenHeight);
    const insideRoi = isPointInPolygon({ x: centerX, y: centerY }, polyPx);

    if (!insideRoi) {
      const lastIn = this.lastSeenInsideByTrack.get(trackId) ?? 0;
      if (nowMs - lastIn > 900) {
        this.stoppedInsideSinceByTrack.delete(trackId);
      }
      return {
        insideRoi: false,
        dwellSeconds: 0,
        isViolation: false,
        justTriggeredViolation: false,
      };
    }

    this.lastSeenInsideByTrack.set(trackId, nowMs);

    if (motionStatus === 'DETENIDO') {
      let sinceMs = this.stoppedInsideSinceByTrack.get(trackId);
      if (sinceMs === undefined) {
        sinceMs = nowMs;
        this.stoppedInsideSinceByTrack.set(trackId, sinceMs);
      }
      const dwellSeconds = Number(((nowMs - sinceMs) / 1000).toFixed(1));
      const threshold = Math.max(3, dwellThresholdSeconds || 10);
      const isViolation = dwellSeconds >= threshold;
      let justTriggeredViolation = false;

      if (isViolation && !this.violationCreditedTracks.has(trackId)) {
        this.violationCreditedTracks.add(trackId);
        this.totalRoiViolations += 1;
        justTriggeredViolation = true;
      }

      return {
        insideRoi: true,
        dwellSeconds,
        isViolation,
        justTriggeredViolation,
      };
    } else {
      // Si el vehículo vuelve a moverse dentro de la zona, reinicia el cronómetro de detención
      this.stoppedInsideSinceByTrack.set(trackId, nowMs);
      return {
        insideRoi: true,
        dwellSeconds: 0,
        isViolation: false,
        justTriggeredViolation: false,
      };
    }
  }

  /**
   * Inicializa perezosamente el Web Worker de tesseract.js sólo cuando el usuario activa ALPR.
   */
  private async ensureAlprWorker(): Promise<TesseractWorker | null> {
    if (this.tesseractWorker) return this.tesseractWorker;
    if (this.isWorkerInitializing) return null;

    this.isWorkerInitializing = true;
    try {
      const worker = await createWorker('eng');
      await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
      });
      this.tesseractWorker = worker;
      return worker;
    } catch {
      return null;
    } finally {
      this.isWorkerInitializing = false;
    }
  }

  /**
   * Ejecuta lectura asíncrona de matrícula (ALPR) en un Web Worker sobre el tercio inferior
   * del vehículo detenido sin bloquear jamás el hilo de inferencia ONNX.
   */
  public maybeScheduleAlprScan(params: {
    enableAlpr: boolean;
    videoElement: HTMLVideoElement | null;
    screenWidth: number;
    screenHeight: number;
    isFrontCamera: boolean;
    trackId: number;
    screenBox: { x: number; y: number; width: number; height: number };
    motionStatus: VehicleMotionStatus;
    nowMs: number;
    onPlateRecognized?: (trackId: number, plate: string) => void;
  }): void {
    const {
      enableAlpr,
      videoElement,
      screenWidth,
      screenHeight,
      isFrontCamera,
      trackId,
      screenBox,
      motionStatus,
      nowMs,
      onPlateRecognized,
    } = params;

    if (!enableAlpr || !videoElement || this.isOcrBusy) return;
    if (this.plateByTrackId.has(trackId)) return;

    if (motionStatus !== 'DETENIDO' && screenBox.height < 110) return;
    if (screenBox.width < 56 || screenBox.height < 48) return;

    const lastTry = this.lastAlprAttemptByTrackId.get(trackId) ?? 0;
    if (nowMs - lastTry < 7500) return;
    this.lastAlprAttemptByTrackId.set(trackId, nowMs);

    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;
    if (vw === 0 || vh === 0) return;

    if (!this.alprCanvas && typeof document !== 'undefined') {
      this.alprCanvas = document.createElement('canvas');
    }
    const canvas = this.alprCanvas;
    if (!canvas) return;

    const outW = 240;
    const outH = 84;
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const plateScreenX = screenBox.x + screenBox.width * 0.14;
    const plateScreenY = screenBox.y + screenBox.height * 0.54;
    const plateScreenW = screenBox.width * 0.72;
    const plateScreenH = screenBox.height * 0.40;

    const coverScale = Math.max(
      screenWidth / Math.max(1, vw),
      screenHeight / Math.max(1, vh)
    );
    const renderedW = vw * coverScale;
    const renderedH = vh * coverScale;
    const cropOffsetX = (renderedW - screenWidth) * 0.5;
    const cropOffsetY = (renderedH - screenHeight) * 0.5;

    let srcX = (plateScreenX + cropOffsetX) / coverScale;
    const srcY = (plateScreenY + cropOffsetY) / coverScale;
    const srcW = Math.max(16, plateScreenW / coverScale);
    const srcH = Math.max(12, plateScreenH / coverScale);

    if (isFrontCamera) {
      srcX = Math.max(0, vw - srcX - srcW);
    }

    ctx.drawImage(
      videoElement,
      Math.max(0, srcX),
      Math.max(0, srcY),
      Math.min(vw, srcW),
      Math.min(vh, srcH),
      0,
      0,
      outW,
      outH
    );

    const imgData = ctx.getImageData(0, 0, outW, outH);
    const d = imgData.data;
    let sumLum = 0;
    const totalPx = outW * outH;

    for (let i = 0; i < d.length; i += 4) {
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      sumLum += lum;
    }
    const meanLum = sumLum / Math.max(1, totalPx);

    for (let i = 0; i < d.length; i += 4) {
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      const contrasted = Math.max(0, Math.min(255, (lum - meanLum) * 2.15 + 128));
      d[i] = contrasted;
      d[i + 1] = contrasted;
      d[i + 2] = contrasted;
    }
    ctx.putImageData(imgData, 0, 0);

    const cropDataUrl = canvas.toDataURL('image/png');
    this.isOcrBusy = true;

    this.ensureAlprWorker()
      .then(async (worker) => {
        if (!worker) return;
        const res = await worker.recognize(cropDataUrl);
        const rawText = (res?.data?.text ?? '')
          .toUpperCase()
          .replace(/[^A-Z0-9-]/g, '')
          .trim();

        if (
          rawText.length >= 5 &&
          rawText.length <= 9 &&
          /[A-Z]/.test(rawText) &&
          /[0-9]/.test(rawText)
        ) {
          this.plateByTrackId.set(trackId, rawText);
          onPlateRecognized?.(trackId, rawText);
        }
      })
      .catch(() => {})
      .finally(() => {
        this.isOcrBusy = false;
      });
  }

  public resetStats(): void {
    this.stoppedInsideSinceByTrack.clear();
    this.lastSeenInsideByTrack.clear();
    this.violationCreditedTracks.clear();
    this.totalRoiViolations = 0;
  }
}

export const globalSpatialZonesROI = new SpatialZonesROIManager();
