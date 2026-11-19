/**
 * Motor Puro de Renderizado AR 2D (Compatible con OffscreenCanvas en Web Worker y Canvas 2D)
 * (src/workers/arOverlayRendererCore.ts)
 *
 * - Ejecuta el bucle de dibujo de alta densidad (retículas tácticas, estelas de 15 posiciones,
 *   esqueleto de 17 keypoints, polígono ROI complejo con Ray-Casting visual, línea de aforo
 *   bidireccional y vectores de riesgo de colisión peatón-vehículo) fuera del hilo principal.
 * - Incluye caché LRU de métricas de texto (ctx.measureText) y filtros de Kalman 1D + LERP
 *   por trackId para mantener 60/120 FPS estables incluso con +40 objetos simultáneos en escena.
 */

import {
  AR_THEMES,
  ARThemeId,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { SKELETON_CONNECTIONS } from '../engine/cvTypes';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { isCocoVehicleClass } from '../utils/VehicleSpeedEstimator';
import {
  TrafficCountBreakdown,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';
import { RoiSnapGuide, RoiVerticesNorm } from '../modules/SpatialZonesROI';
import { ProximityHazardLink } from '../modules/HumanSafetyAnalytics';

export type Any2DContext =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D;

export interface ARSceneWorkerState {
  detections: OnnxTrackedDetection[];
  arTheme: ARThemeId;
  enablePose: boolean;
  enableGyro: boolean;
  onlyVehiclesMode: boolean;
  showCountingLine: boolean;
  lineEndpointsNorm: VirtualLineEndpointsNorm;
  activeDragHandle: 'a' | 'b' | null;
  countingStats: TrafficCountBreakdown;
  enableRoiPolygon: boolean;
  roiVerticesNorm?: RoiVerticesNorm;
  activeRoiVertexIndex: 0 | 1 | 2 | 3 | null;
  activeRoiViolationInView: boolean;
  roiSnapGuides?: RoiSnapGuide[];
  proximityLinks: ProximityHazardLink[];
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  screenWidth: number;
  screenHeight: number;
  dpr: number;
}

export interface ARGyroWorkerState {
  offsetX: number;
  offsetY: number;
  enabled: boolean;
}

/**
 * Filtro de Kalman 1D ligero autocontenido para ejecutarse dentro del Worker de OffscreenCanvas.
 */
class WorkerScalarKalmanFilter {
  private x: number;
  private v = 0;
  private p00 = 1.0;
  private p01 = 0.0;
  private p10 = 0.0;
  private p11 = 1.0;
  private readonly qPos: number;
  private readonly qVel: number;
  private readonly rMeas: number;

  constructor(initialValue = 0, qPos = 0.018, qVel = 0.08, rMeas = 0.14) {
    this.x = initialValue;
    this.qPos = qPos;
    this.qVel = qVel;
    this.rMeas = rMeas;
  }

  public predictAndUpdate(measurement: number, dtSec: number): number {
    const dt = Math.max(0.004, Math.min(0.08, dtSec));
    const predX = this.x + this.v * dt;
    const predV = this.v;

    const p00 = this.p00 + dt * (this.p10 + this.p01) + dt * dt * this.p11 + this.qPos;
    const p01 = this.p01 + dt * this.p11;
    const p10 = this.p10 + dt * this.p11;
    const p11 = this.p11 + this.qVel;

    const y = measurement - predX;
    const s = p00 + this.rMeas;
    const invS = 1.0 / Math.max(1e-6, s);
    const k0 = p00 * invS;
    const k1 = p10 * invS;

    this.x = predX + k0 * y;
    this.v = predV + k1 * y;

    this.p00 = (1 - k0) * p00;
    this.p01 = (1 - k0) * p01;
    this.p10 = p10 - k1 * p00;
    this.p11 = p11 - k1 * p01;

    return this.x;
  }
}

interface WorkerSmoothedTrack {
  x: number;
  y: number;
  width: number;
  height: number;
  keypoints: OnnxKeypoint2D[];
  lastSeenMs: number;
  kx: WorkerScalarKalmanFilter;
  ky: WorkerScalarKalmanFilter;
  kw: WorkerScalarKalmanFilter;
  kh: WorkerScalarKalmanFilter;
}

export class AROverlayRendererEngine {
  private readonly spatialTrackMap = new Map<number, WorkerSmoothedTrack>();
  private readonly textWidthCache = new Map<string, number>();
  private gyroX = 0;
  private gyroY = 0;
  private gyroEnabled = true;

  public updateGyroOffset(gyro: ARGyroWorkerState): void {
    this.gyroEnabled = gyro.enabled;
    if (!gyro.enabled) {
      this.gyroX = 0;
      this.gyroY = 0;
      return;
    }
    this.gyroX = gyro.offsetX;
    this.gyroY = gyro.offsetY;
  }

  /**
   * Mide el ancho de una etiqueta reutilizando una caché LRU en memoria para evitar
   * llamadas costosas a measureText() cuando hay decenas de objetos en escena.
   */
  private measureCachedTextWidth(ctx: Any2DContext, font: string, text: string): number {
    const key = `${font}|${text}`;
    const cached = this.textWidthCache.get(key);
    if (cached !== undefined) return cached;

    const width = ctx.measureText(text).width;
    if (this.textWidthCache.size > 600) {
      const firstKey = this.textWidthCache.keys().next().value;
      if (firstKey !== undefined) {
        this.textWidthCache.delete(firstKey);
      }
    }
    this.textWidthCache.set(key, width);
    return width;
  }

  /**
   * Calcula la posición interpolada con Filtro de Kalman 1D + LERP + Offset Inercial
   * directamente dentro del Worker de renderizado.
   */
  private computeAnchoredDetection(
    det: OnnxTrackedDetection,
    nowMs: number,
    dtSec: number,
    gx: number,
    gy: number
  ): {
    x: number;
    y: number;
    width: number;
    height: number;
    keypoints: OnnxKeypoint2D[];
  } {
    const existing = this.spatialTrackMap.get(det.trackId);
    const lerpAlpha = 0.36;

    let sx = det.screenBox.x;
    let sy = det.screenBox.y;
    let sW = det.screenBox.width;
    let sH = det.screenBox.height;
    let smoothKps = det.keypoints;

    if (existing && nowMs - existing.lastSeenMs < 550) {
      const kalmanX = existing.kx.predictAndUpdate(det.screenBox.x, dtSec);
      const kalmanY = existing.ky.predictAndUpdate(det.screenBox.y, dtSec);
      const kalmanW = existing.kw.predictAndUpdate(det.screenBox.width, dtSec);
      const kalmanH = existing.kh.predictAndUpdate(det.screenBox.height, dtSec);

      sx = existing.x + (kalmanX - existing.x) * lerpAlpha;
      sy = existing.y + (kalmanY - existing.y) * lerpAlpha;
      sW = existing.width + (kalmanW - existing.width) * lerpAlpha;
      sH = existing.height + (kalmanH - existing.height) * lerpAlpha;

      if (det.keypoints.length === 17 && existing.keypoints.length === 17) {
        smoothKps = det.keypoints.map((kp, idx) => ({
          ...kp,
          x: existing.keypoints[idx].x + (kp.x - existing.keypoints[idx].x) * lerpAlpha,
          y: existing.keypoints[idx].y + (kp.y - existing.keypoints[idx].y) * lerpAlpha,
        }));
      }

      existing.x = sx;
      existing.y = sy;
      existing.width = sW;
      existing.height = sH;
      existing.keypoints = smoothKps;
      existing.lastSeenMs = nowMs;
    } else {
      this.spatialTrackMap.set(det.trackId, {
        x: sx,
        y: sy,
        width: sW,
        height: sH,
        keypoints: smoothKps,
        lastSeenMs: nowMs,
        kx: new WorkerScalarKalmanFilter(sx),
        ky: new WorkerScalarKalmanFilter(sy),
        kw: new WorkerScalarKalmanFilter(sW),
        kh: new WorkerScalarKalmanFilter(sH),
      });
    }

    if (this.spatialTrackMap.size > 64) {
      for (const [tid, rec] of this.spatialTrackMap.entries()) {
        if (nowMs - rec.lastSeenMs > 1600) {
          this.spatialTrackMap.delete(tid);
        }
      }
    }

    return {
      x: sx + gx,
      y: sy + gy,
      width: sW,
      height: sH,
      keypoints: smoothKps,
    };
  }

  /**
   * Renderiza un fotograma completo del HUD AR en el contexto 2D (OffscreenCanvas o Canvas estándar).
   */
  public renderFrame(
    ctx: Any2DContext,
    scene: ARSceneWorkerState,
    nowMs: number,
    dtSec: number
  ): void {
    const sw = scene.screenWidth;
    const sh = scene.screenHeight;
    const dpr = scene.dpr;
    if (sw <= 0 || sh <= 0) return;

    // Decaimiento suave inter-frame del offset inercial en el hilo de renderizado
    this.gyroX *= 0.95;
    this.gyroY *= 0.95;
    const gx = this.gyroEnabled && scene.enableGyro ? this.gyroX : 0;
    const gy = this.gyroEnabled && scene.enableGyro ? this.gyroY : 0;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, sw, sh);

    const themeId = scene.arTheme;
    const theme = AR_THEMES[themeId] ?? AR_THEMES.tactical;
    const pulseFactor = 0.5 + 0.5 * Math.sin(nowMs * 0.012);

    // A) Superposición térmica opcional (Spatial Heatmap)
    if (scene.showHeatmapOverlay && scene.heatmapPoints.length > 0) {
      for (let i = 0; i < scene.heatmapPoints.length; i++) {
        const pt = scene.heatmapPoints[i];
        const px = pt.xNorm * sw + gx;
        const py = pt.yNorm * sh + gy;
        const rad = 36;
        const grad = ctx.createRadialGradient(px, py, 2, px, py, rad);
        grad.addColorStop(0, `rgba(249, 115, 22, ${0.22 * pt.weight})`);
        grad.addColorStop(0.55, `rgba(56, 189, 248, ${0.1 * pt.weight})`);
        grad.addColorStop(1, 'rgba(56, 189, 248, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // B) Retícula Central de Horizonte Óptico AR
    if (theme.showHorizonReticle) {
      const rCx = sw * 0.5 + gx * 0.5;
      const rCy = sh * 0.5 + gy * 0.5;
      const reticleColor =
        themeId === 'industrial'
          ? 'rgba(250, 204, 21, 0.35)'
          : scene.onlyVehiclesMode
          ? 'rgba(249, 115, 22, 0.36)'
          : 'rgba(56, 189, 248, 0.32)';

      ctx.save();
      ctx.strokeStyle = reticleColor;
      ctx.lineWidth = 1.1;

      ctx.beginPath();
      ctx.arc(rCx, rCy, 22, 0, Math.PI * 2);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(rCx - 44, rCy);
      ctx.lineTo(rCx - 12, rCy);
      ctx.moveTo(rCx + 12, rCy);
      ctx.lineTo(rCx + 44, rCy);
      ctx.moveTo(rCx, rCy - 28);
      ctx.lineTo(rCx, rCy - 10);
      ctx.moveTo(rCx, rCy + 10);
      ctx.lineTo(rCx, rCy + 28);
      ctx.stroke();

      ctx.fillStyle = reticleColor;
      ctx.beginPath();
      ctx.arc(rCx, rCy, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // C) POLÍGONO ROI INTERACTIVO DE 4 VÉRTICES (ZONA DE EXCLUSIÓN / ESTACIONAMIENTO PROHIBIDO)
    if (scene.enableRoiPolygon && scene.roiVerticesNorm && scene.roiVerticesNorm.length === 4) {
      const pts = scene.roiVerticesNorm.map((v) => ({ x: v.x * sw, y: v.y * sh }));
      const roiAlertOn = scene.activeRoiViolationInView;
      const roiStrokeColor = roiAlertOn ? '#EF4444' : '#F59E0B';
      const roiFillColor = roiAlertOn
        ? `rgba(239, 68, 68, ${0.18 + pulseFactor * 0.18})`
        : 'rgba(245, 158, 11, 0.11)';

      ctx.save();

      // Relleno del polígono ROI
      ctx.fillStyle = roiFillColor;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.lineTo(pts[2].x, pts[2].y);
      ctx.lineTo(pts[3].x, pts[3].y);
      ctx.closePath();
      ctx.fill();

      // Malla táctica interna diagonal ligera para mejorar la percepción espacial del polígono ROI
      ctx.strokeStyle = roiAlertOn
        ? 'rgba(239, 68, 68, 0.22)'
        : 'rgba(245, 158, 11, 0.14)';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[2].x, pts[2].y);
      ctx.moveTo(pts[1].x, pts[1].y);
      ctx.lineTo(pts[3].x, pts[3].y);
      ctx.stroke();

      // Contorno perimetral punteado con brillo neón
      ctx.strokeStyle = roiStrokeColor;
      ctx.lineWidth = roiAlertOn ? 3.2 : 2.2;
      ctx.shadowColor = roiStrokeColor;
      ctx.shadowBlur = roiAlertOn ? 18 : 8;
      ctx.setLineDash([8, 5]);
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.lineTo(pts[2].x, pts[2].y);
      ctx.lineTo(pts[3].x, pts[3].y);
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;

      // Etiqueta en el centroide geométrico del Polígono ROI
      const polyCx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) * 0.25;
      const polyCy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) * 0.25;
      const roiTitle = roiAlertOn
        ? '⚠️ INVASIÓN / DETENCIÓN PROHIBIDA EN ZONA ROI'
        : 'ZONA ROI EXCLUSIÓN (Arrastra V1..V4)';

      const roiFont = '700 10.5px "JetBrains Mono", monospace';
      ctx.font = roiFont;
      const rTw = this.measureCachedTextWidth(ctx, roiFont, roiTitle);
      const rPw = Math.ceil(rTw + 16);
      const rPh = 22;

      ctx.fillStyle = roiAlertOn
        ? 'rgba(127, 29, 29, 0.94)'
        : 'rgba(15, 23, 42, 0.86)';
      ctx.beginPath();
      ctx.roundRect(polyCx - rPw * 0.5, polyCy - rPh * 0.5, rPw, rPh, 6);
      ctx.fill();

      ctx.strokeStyle = roiStrokeColor;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.roundRect(polyCx - rPw * 0.5, polyCy - rPh * 0.5, rPw, rPh, 6);
      ctx.stroke();

      ctx.fillStyle = roiAlertOn ? '#FEE2E2' : '#FDE68A';
      ctx.textBaseline = 'middle';
      ctx.fillText(roiTitle, polyCx - rPw * 0.5 + 8, polyCy + 0.5);

      // 4 Vértices circulares arrastrables (V1, V2, V3, V4)
      const hasActiveSnap =
        Boolean(scene.roiSnapGuides) && (scene.roiSnapGuides?.length ?? 0) > 0;

      // Guías Magnéticas de Alineación (Snapping a Carriles en Perspectiva, Ortogonales y Línea de Aforo)
      if (hasActiveSnap && scene.roiSnapGuides) {
        const snapFont = '700 10px "JetBrains Mono", monospace';
        for (let gIdx = 0; gIdx < scene.roiSnapGuides.length; gIdx++) {
          const guide = scene.roiSnapGuides[gIdx];
          const gx1 = guide.x1Norm * sw;
          const gy1 = guide.y1Norm * sh;
          const gx2 = guide.x2Norm * sw;
          const gy2 = guide.y2Norm * sh;
          const sx = guide.snapXNorm * sw;
          const sy = guide.snapYNorm * sh;

          // Línea guía láser punteada de extremo a extremo
          ctx.strokeStyle = guide.color;
          ctx.lineWidth = 1.9;
          ctx.shadowColor = guide.color;
          ctx.shadowBlur = 12;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(gx1, gy1);
          ctx.lineTo(gx2, gy2);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.shadowBlur = 0;

          // Si es un rayo de perspectiva de carril, dibuja el marcador del Punto de Fuga (VP)
          if (guide.kind === 'LANE_PERSPECTIVE') {
            ctx.strokeStyle = guide.color;
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.arc(gx1, gy1, 7, 0, Math.PI * 2);
            ctx.moveTo(gx1 - 11, gy1);
            ctx.lineTo(gx1 + 11, gy1);
            ctx.moveTo(gx1, gy1 - 11);
            ctx.lineTo(gx1, gy1 + 11);
            ctx.stroke();
          }

          // Anillo magnético concéntrico en el vértice acoplado
          ctx.strokeStyle = guide.color;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          ctx.arc(sx, sy, 20, 0, Math.PI * 2);
          ctx.stroke();

          // Píldora flotante indicando el tipo de acople magnético activo
          ctx.font = snapFont;
          const sTw = this.measureCachedTextWidth(ctx, snapFont, guide.label);
          const sPw = Math.ceil(sTw + 14);
          const sPh = 20;
          const badgeX = Math.max(8, Math.min(sw - sPw - 8, sx - sPw * 0.5));
          const badgeY = Math.max(24, sy - 36 - gIdx * 24);

          ctx.fillStyle = 'rgba(9, 13, 22, 0.94)';
          ctx.beginPath();
          ctx.roundRect(badgeX, badgeY, sPw, sPh, 5);
          ctx.fill();

          ctx.strokeStyle = guide.color;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(badgeX, badgeY, sPw, sPh, 5);
          ctx.stroke();

          ctx.fillStyle = '#F8FAFC';
          ctx.textBaseline = 'middle';
          ctx.fillText(guide.label, badgeX + 7, badgeY + sPh * 0.5 + 0.5);
        }
      }

      for (let idx = 0; idx < pts.length; idx++) {
        const pt = pts[idx];
        const isDraggingThis = scene.activeRoiVertexIndex === idx;
        const isSnappedVertex = isDraggingThis && hasActiveSnap;
        const rad = isDraggingThis ? 15 : 11;

        ctx.fillStyle = isSnappedVertex
          ? 'rgba(34, 211, 238, 0.36)'
          : isDraggingThis
          ? 'rgba(245, 158, 11, 0.35)'
          : 'rgba(15, 23, 42, 0.85)';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, rad, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = isSnappedVertex
          ? '#22D3EE'
          : isDraggingThis
          ? '#FBBF24'
          : roiStrokeColor;
        ctx.lineWidth = isSnappedVertex ? 2.8 : 2.2;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, rad, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = '#F8FAFC';
        ctx.font = '700 9.5px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`V${idx + 1}`, pt.x, pt.y + 0.5);
        ctx.textAlign = 'start';
      }

      ctx.restore();
    }

    // D) LÍNEA VIRTUAL DE CONTEO Y AFORO VEHICULAR (TRIPWIRE AR BIDIRECCIONAL)
    if (scene.showCountingLine) {
      const ax = scene.lineEndpointsNorm.x1 * sw;
      const ay = scene.lineEndpointsNorm.y1 * sh;
      const bx = scene.lineEndpointsNorm.x2 * sw;
      const by = scene.lineEndpointsNorm.y2 * sh;

      const countStats = scene.countingStats;
      const msSinceFlash = nowMs - countStats.lastFlashTimestampMs;
      const isFlashing = countStats.lastFlashTimestampMs > 0 && msSinceFlash >= 0 && msSinceFlash < 550;
      const flashProgress = isFlashing ? 1 - msSinceFlash / 550 : 0;

      const linePrimaryColor = isFlashing
        ? '#22C55E'
        : themeId === 'industrial'
        ? '#FACC15'
        : '#38BDF8';

      ctx.save();

      ctx.strokeStyle = isFlashing
        ? `rgba(34, 197, 94, ${0.4 + flashProgress * 0.55})`
        : 'rgba(56, 189, 248, 0.25)';
      ctx.lineWidth = isFlashing ? 11 + flashProgress * 7 : 7;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();

      ctx.strokeStyle = linePrimaryColor;
      ctx.lineWidth = isFlashing ? 3.4 : 2.4;
      ctx.shadowColor = linePrimaryColor;
      ctx.shadowBlur = isFlashing ? 18 : 10;
      ctx.setLineDash([10, 6]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;

      if (isFlashing && countStats.lastIntersectionPoint) {
        const ix = countStats.lastIntersectionPoint.x;
        const iy = countStats.lastIntersectionPoint.y;
        const rippleRadius = 10 + (1 - flashProgress) * 32;

        ctx.strokeStyle = `rgba(34, 197, 94, ${flashProgress})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(ix, iy, rippleRadius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = '#22C55E';
        ctx.beginPath();
        ctx.arc(ix, iy, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }

      const midX = (ax + bx) * 0.5;
      const midY = (ay + by) * 0.5;
      const badgeText = isFlashing
        ? `✓ CRUCE #${countStats.lastCrossingTrackId} (${countStats.lastCrossingDirection}) · AFORO: ${countStats.total}`
        : `LÍNEA AFORO AR · ↑${countStats.inbound} ↓${countStats.outbound} (Arrastra Nodos A/B)`;

      const lineFont = '700 10.5px "JetBrains Mono", monospace';
      ctx.font = lineFont;
      const tw = this.measureCachedTextWidth(ctx, lineFont, badgeText);
      const pw = Math.ceil(tw + 16);
      const ph = 22;

      ctx.fillStyle = isFlashing
        ? 'rgba(6, 78, 59, 0.92)'
        : 'rgba(9, 13, 22, 0.88)';
      ctx.beginPath();
      ctx.roundRect(midX - pw * 0.5, midY - ph - 8, pw, ph, 6);
      ctx.fill();

      ctx.strokeStyle = linePrimaryColor;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.roundRect(midX - pw * 0.5, midY - ph - 8, pw, ph, 6);
      ctx.stroke();

      ctx.fillStyle = isFlashing ? '#86EFAC' : '#F8FAFC';
      ctx.textBaseline = 'middle';
      ctx.fillText(badgeText, midX - pw * 0.5 + 8, midY - ph * 0.5 - 7.5);

      const drawHandleNode = (
        hx: number,
        hy: number,
        label: string,
        isActive: boolean
      ) => {
        const radius = isActive ? 15 : 12;
        ctx.fillStyle = isActive
          ? 'rgba(34, 197, 94, 0.32)'
          : 'rgba(56, 189, 248, 0.24)';
        ctx.beginPath();
        ctx.arc(hx, hy, radius + 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#090D16';
        ctx.beginPath();
        ctx.arc(hx, hy, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = isActive ? '#22C55E' : linePrimaryColor;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.arc(hx, hy, radius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = '#F8FAFC';
        ctx.font = '700 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, hx, hy + 0.5);
        ctx.textAlign = 'start';
      };

      drawHandleNode(ax, ay, 'A', scene.activeDragHandle === 'a');
      drawHandleNode(bx, by, 'B', scene.activeDragHandle === 'b');

      ctx.restore();
    }

    // E) LÍNEAS DE ADVERTENCIA DE PROXIMIDAD PEATÓN-VEHÍCULO (RIESGO DE COLISIÓN)
    if (scene.proximityLinks && scene.proximityLinks.length > 0) {
      ctx.save();
      const proxFont = '700 10px "JetBrains Mono", monospace';
      for (let i = 0; i < scene.proximityLinks.length; i++) {
        const link = scene.proximityLinks[i];
        const px = link.personX + gx;
        const py = link.personY + gy;
        const vx = link.vehicleX + gx;
        const vy = link.vehicleY + gy;
        const isCrit = link.severity === 'CRITICAL';
        const linkCol = isCrit ? '#EF4444' : '#F59E0B';

        ctx.strokeStyle = linkCol;
        ctx.lineWidth = isCrit ? 2.8 : 2.2;
        ctx.shadowColor = linkCol;
        ctx.shadowBlur = 12;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(vx, vy);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;

        ctx.fillStyle = linkCol;
        ctx.beginPath();
        ctx.arc(px, py, 4.5, 0, Math.PI * 2);
        ctx.arc(vx, vy, 4.5, 0, Math.PI * 2);
        ctx.fill();

        const mx = (px + vx) * 0.5;
        const my = (py + vy) * 0.5;
        ctx.font = proxFont;
        const lTw = this.measureCachedTextWidth(ctx, proxFont, link.label);
        const lPw = Math.ceil(lTw + 14);
        const lPh = 20;

        ctx.fillStyle = isCrit
          ? 'rgba(127, 29, 29, 0.94)'
          : 'rgba(120, 53, 15, 0.92)';
        ctx.beginPath();
        ctx.roundRect(mx - lPw * 0.5, my - lPh * 0.5, lPw, lPh, 5);
        ctx.fill();

        ctx.strokeStyle = linkCol;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.roundRect(mx - lPw * 0.5, my - lPh * 0.5, lPw, lPh, 5);
        ctx.stroke();

        ctx.fillStyle = '#FEF2F2';
        ctx.textBaseline = 'middle';
        ctx.fillText(link.label, mx - lPw * 0.5 + 7, my + 0.5);
      }
      ctx.restore();
    }

    // F) Renderizado de Objetos Rastreados por ByteTrack + Cinemática (km/h) + Alertas Rojas (#EF4444)
    const activeList = scene.detections;
    const labelFont = theme.useMonospaceFont
      ? '700 10.5px "JetBrains Mono", monospace'
      : '600 11.5px "Plus Jakarta Sans", -apple-system, sans-serif';

    for (let i = 0; i < activeList.length; i++) {
      const det = activeList[i];
      const anchored = this.computeAnchoredDetection(det, nowMs, dtSec, gx, gy);

      const bx = Math.max(4, Math.min(sw - 12, anchored.x));
      const by = Math.max(4, Math.min(sh - 12, anchored.y));
      const bw = Math.max(10, Math.min(sw - bx - 2, anchored.width));
      const bh = Math.max(10, Math.min(sh - by - 2, anchored.height));
      const smoothKps = anchored.keypoints;

      const isVehicle = isCocoVehicleClass(det.classId, det.className);
      const isOverspeed = det.isOverspeed;
      const isRoiViol = Boolean(det.isRoiViolation);
      const isFall = Boolean(det.isFallDetected);
      const isCriticalAlert = isOverspeed || isRoiViol || isFall;

      const baseCatStyle = theme.categories[det.category] ?? theme.categories.object;
      const strokeColor = isCriticalAlert
        ? '#EF4444'
        : det.hasProximityRisk
        ? '#F59E0B'
        : baseCatStyle.stroke;
      const fillColor = isCriticalAlert
        ? `rgba(239, 68, 68, ${0.16 + pulseFactor * 0.14})`
        : baseCatStyle.fill;
      const glowColor = isCriticalAlert
        ? 'rgba(239, 68, 68, 0.65)'
        : baseCatStyle.glow;

      ctx.save();

      // 1. Estela de Movimiento (Motion Trails)
      if (det.trail && det.trail.length >= 2) {
        const firstPt = det.trail[0];
        const lastPt = det.trail[det.trail.length - 1];
        const grad = ctx.createLinearGradient(
          firstPt.x + gx,
          firstPt.y + gy,
          lastPt.x + gx,
          lastPt.y + gy
        );
        grad.addColorStop(0, 'rgba(15, 23, 42, 0.05)');
        grad.addColorStop(0.5, glowColor);
        grad.addColorStop(1, strokeColor);

        ctx.save();
        ctx.strokeStyle = grad;
        ctx.lineWidth = isCriticalAlert ? 2.6 : 2.0;
        ctx.setLineDash([4, 4]);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        for (let tIdx = 0; tIdx < det.trail.length; tIdx++) {
          const tp = det.trail[tIdx];
          const tx = tp.x + gx;
          const ty = tp.y + gy;
          if (tIdx === 0) {
            ctx.moveTo(tx, ty);
          } else {
            ctx.lineTo(tx, ty);
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);

        for (let tIdx = 1; tIdx < det.trail.length; tIdx += 3) {
          const tp = det.trail[tIdx];
          const alpha = (tIdx + 1) / det.trail.length;
          ctx.fillStyle = strokeColor;
          ctx.globalAlpha = alpha * 0.78;
          ctx.beginPath();
          ctx.arc(tp.x + gx, tp.y + gy, 2.0, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // 2. Aura de Advertencia Pulsante cuando hay Exceso de Velocidad, Invasión ROI o Caída Humana
      if (isCriticalAlert) {
        const auraPad = 4 + pulseFactor * 5;
        ctx.strokeStyle = `rgba(239, 68, 68, ${0.35 + pulseFactor * 0.45})`;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.roundRect(
          bx - auraPad,
          by - auraPad,
          bw + auraPad * 2,
          bh + auraPad * 2,
          theme.cornerRadius + 3
        );
        ctx.stroke();
      }

      // 3. Relleno translúcido de la caja
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
      ctx.fill();

      // 4. Retículas Tácticas Adaptables (Corner Brackets)
      if (theme.useCornerBrackets) {
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = isCriticalAlert ? 1.5 : 0.9;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
        ctx.stroke();

        const cLen = Math.min(theme.cornerBracketLength, bw * 0.34, bh * 0.34);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isCriticalAlert
          ? theme.boxStrokeWidth + 1.4
          : theme.boxStrokeWidth + 0.6;
        ctx.shadowColor = strokeColor;
        ctx.shadowBlur = isCriticalAlert ? 14 + pulseFactor * 8 : 10;
        ctx.lineCap = 'square';

        ctx.beginPath();
        // Top-Left
        ctx.moveTo(bx, by + cLen);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + cLen, by);
        // Top-Right
        ctx.moveTo(bx + bw - cLen, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + cLen);
        // Bottom-Left
        ctx.moveTo(bx, by + bh - cLen);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + cLen, by + bh);
        // Bottom-Right
        ctx.moveTo(bx + bw - cLen, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - cLen);
        ctx.stroke();
        ctx.shadowBlur = 0;

        const cx = bx + bw * 0.5;
        const cy = by + bh * 0.5;
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, isCriticalAlert ? 4.2 : 3.2, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isCriticalAlert ? theme.boxStrokeWidth + 1.2 : theme.boxStrokeWidth;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
        ctx.stroke();
      }

      // 5. Etiquetas Flotantes con Línea Guía (Callout Pins)
      const confPct = Math.round(det.confidence * 100);
      const distMeters = estimateObjectDistanceMeters(det.category, bh, sh);
      const classUpper = det.className.toUpperCase();
      const crossedTag = det.hasCrossedLine ? ' · ✓AFORO' : '';
      const plateTag = det.licensePlate ? ` · [${det.licensePlate}]` : '';
      const roiTag =
        det.insideRoi && !isRoiViol
          ? ` · ROI:${(det.roiDwellSeconds ?? 0).toFixed(0)}s`
          : '';

      let labelText: string;
      if (isFall) {
        labelText = `⚠️ POSIBLE CAÍDA DETECTADA #${det.trackId} (${(
          det.fallDurationSeconds ?? 2.5
        ).toFixed(1)}s)`;
      } else if (isRoiViol) {
        labelText = `⚠️ ${classUpper} #${det.trackId} • DETENCIÓN PROHIBIDA (${(
          det.roiDwellSeconds ?? 10
        ).toFixed(0)}s)${plateTag}`;
      } else if (isOverspeed) {
        labelText = `⚠️ ${classUpper} #${det.trackId} • ${Math.round(
          det.speedKmh
        )} km/h • EXCESO${plateTag}${crossedTag}`;
      } else if (isVehicle) {
        if (det.motionStatus === 'DETENIDO') {
          labelText = `${classUpper} #${det.trackId} • DETENIDO · ~${distMeters.toFixed(
            1
          )}m${plateTag}${roiTag}${crossedTag}`;
        } else {
          labelText = `${classUpper} #${det.trackId} • ${Math.round(
            det.speedKmh
          )} km/h • EN MOVIMIENTO${plateTag}${crossedTag}`;
        }
      } else {
        const speedText =
          det.speedMetersPerSec >= 0.2
            ? `${det.speedMetersPerSec.toFixed(1)}m/s`
            : `${Math.round(det.speedPxPerSec)}px/s`;
        const proxTag = det.hasProximityRisk ? ' · ⚠️PROXIMIDAD' : '';
        labelText = theme.showDistanceEstimate
          ? `#${det.trackId} ${classUpper} ${confPct}% · ${speedText} · ~${distMeters.toFixed(
              1
            )}m${proxTag}`
          : `#${det.trackId} ${det.className} ${confPct}%${proxTag}`;
      }

      ctx.font = labelFont;
      const textW = this.measureCachedTextWidth(ctx, labelFont, labelText);
      const pillW = Math.ceil(textW + 16);
      const pillH = 22;

      const pillBgColor = isCriticalAlert
        ? 'rgba(127, 29, 29, 0.94)'
        : det.motionStatus === 'DETENIDO' && isVehicle
        ? 'rgba(30, 41, 59, 0.90)'
        : theme.pillBackground;

      const pillTextCol = isCriticalAlert ? '#FEE2E2' : theme.pillTextColor;

      if (theme.showLeaderLinePins) {
        const anchorX = bx + Math.min(28, bw * 0.25);
        const anchorY = by;
        const elbowX = Math.max(8, Math.min(sw - pillW - 8, anchorX + 14));
        const elbowY = Math.max(32, anchorY - 14);

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isCriticalAlert ? 1.8 : 1.4;
        ctx.beginPath();
        ctx.moveTo(anchorX, anchorY);
        ctx.lineTo(elbowX, elbowY);
        ctx.stroke();

        ctx.fillStyle = strokeColor;
        ctx.beginPath();
        ctx.arc(anchorX, anchorY, 2.8, 0, Math.PI * 2);
        ctx.fill();

        const pillY = elbowY - pillH;
        ctx.fillStyle = pillBgColor;
        ctx.beginPath();
        ctx.roundRect(elbowX, pillY, pillW, pillH, 5);
        ctx.fill();

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = isCriticalAlert ? 1.8 : 1.2;
        ctx.beginPath();
        ctx.roundRect(elbowX, pillY, pillW, pillH, 5);
        ctx.stroke();

        ctx.fillStyle = pillTextCol;
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, elbowX + 8, pillY + pillH * 0.5 + 0.5);
      } else {
        const pillX = Math.max(6, Math.min(sw - pillW - 6, bx));
        const pillY = by > pillH + 8 ? by - pillH - 4 : by + 4;
        const rad = themeId === 'native' ? 10 : 3;

        ctx.fillStyle = isCriticalAlert
          ? 'rgba(127, 29, 29, 0.95)'
          : themeId === 'industrial'
          ? '#090D16'
          : pillBgColor;
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, rad);
        ctx.fill();

        ctx.strokeStyle =
          isCriticalAlert || themeId === 'industrial'
            ? strokeColor
            : 'rgba(255, 255, 255, 0.26)';
        ctx.lineWidth = isCriticalAlert || themeId === 'industrial' ? 1.8 : 1;
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, rad);
        ctx.stroke();

        ctx.fillStyle = isCriticalAlert
          ? '#FEE2E2'
          : themeId === 'industrial'
          ? strokeColor
          : pillTextCol;
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, pillX + 8, pillY + pillH * 0.5 + 0.5);
      }

      // 6. Esqueleto Biomecánico Fluido: 17 Keypoints
      if (!scene.onlyVehiclesMode && scene.enablePose && smoothKps.length === 17) {
        ctx.strokeStyle = isFall ? '#EF4444' : theme.skeletonBoneColor;
        ctx.lineWidth = isFall ? 3.0 : themeId === 'native' ? 1.5 : 2.3;
        ctx.lineCap = 'round';

        for (let sIdx = 0; sIdx < SKELETON_CONNECTIONS.length; sIdx++) {
          const [idxA, idxB] = SKELETON_CONNECTIONS[sIdx];
          const kpA = smoothKps[idxA];
          const kpB = smoothKps[idxB];
          if (kpA && kpB && kpA.score > 0.5 && kpB.score > 0.5) {
            ctx.beginPath();
            ctx.moveTo(kpA.x + gx, kpA.y + gy);
            ctx.lineTo(kpB.x + gx, kpB.y + gy);
            ctx.stroke();
          }
        }

        for (let k = 0; k < smoothKps.length; k++) {
          const kp = smoothKps[k];
          if (kp.score > 0.5) {
            const kx = kp.x + gx;
            const ky = kp.y + gy;
            ctx.fillStyle = '#090D16';
            ctx.beginPath();
            ctx.arc(kx, ky, themeId === 'native' ? 2.8 : 4.0, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = isFall ? '#FCA5A5' : theme.skeletonJointColor;
            ctx.beginPath();
            ctx.arc(kx, ky, themeId === 'native' ? 2.0 : 2.8, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      ctx.restore();
    }

    ctx.restore();
  }
}
