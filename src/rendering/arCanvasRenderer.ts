/**
 * Motor de Dibujo AR Agnóstico del Contexto (OffscreenCanvas 2D & HTML5 Canvas 2D)
 * (src/rendering/arCanvasRenderer.ts)
 *
 * - Cero uso de ctx.shadowBlur (evita rasterización por software lenta en GPUs móviles iOS/Android).
 * - Implementa la técnica de Doble Trazo Neón de alto rendimiento:
 *   1) Trazo exterior ancho con globalAlpha = 0.25 (halo luminoso).
 *   2) Trazo interior nítido de 2px con globalAlpha = 1.0.
 * - Ejecuta interpolación lineal (LERP) + Filtro de Kalman 1D a 60 FPS dentro del ciclo de dibujo.
 * - Renderiza las 6 capas AR: Retículas tácticas, Estelas degradadas, Callout Pins (km/h + Placa ALPR),
 *   Esqueletos de 17 keypoints, Línea Virtual de Aforo y Polígono ROI con Guías Magnéticas + Lupa Táctica.
 */

import {
  AR_THEMES,
  ARThemeId,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { SKELETON_CONNECTIONS } from '../engine/cvTypes';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { ProximityHazardLink } from '../modules/HumanSafetyAnalytics';
import { RoiSnapGuide, RoiVerticesNorm } from '../modules/SpatialZonesROI';
import { isCocoVehicleClass } from '../utils/VehicleSpeedEstimator';
import {
  TrafficCountBreakdown,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';

export type AR2DRenderingContext =
  | CanvasRenderingContext2D
  | OffscreenCanvasRenderingContext2D;

export interface ARCanvasRenderState {
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
  /** Estado activo del puntero táctil para renderizar la Lupa Táctica de Precisión a 60 FPS */
  activePointerLoupe?: {
    active: boolean;
    x: number;
    y: number;
    label: string;
    accentColor: string;
  } | null;
  proximityLinks: ProximityHazardLink[];
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  screenWidth: number;
  screenHeight: number;
  dpr: number;
}

export interface ARInertialGyroState {
  offsetX: number;
  offsetY: number;
  enabled: boolean;
}

/**
 * Filtro de Kalman 1D con velocidad para suavizado espacial a 60 FPS libre de jitter.
 */
class FastKalman1D {
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

    const p00 =
      this.p00 + dt * (this.p10 + this.p01) + dt * dt * this.p11 + this.qPos;
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

interface LerpTrackRecord {
  x: number;
  y: number;
  width: number;
  height: number;
  keypoints: OnnxKeypoint2D[];
  lastSeenMs: number;
  kx: FastKalman1D;
  ky: FastKalman1D;
  kw: FastKalman1D;
  kh: FastKalman1D;
}

/**
 * Técnica de Doble Trazo de Alto Rendimiento (reemplaza ctx.shadowBlur en GPUs móviles):
 * Dibuja primero un halo exterior grueso con globalAlpha = 0.25 y luego el trazo interior nítido.
 */
function strokeDoubleNeonSegment(
  ctx: AR2DRenderingContext,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  innerWidth = 2.0,
  outerGlowExtra = 5.0,
  dashPattern?: number[]
): void {
  const prevAlpha = ctx.globalAlpha;
  ctx.strokeStyle = color;
  if (dashPattern && dashPattern.length > 0) {
    ctx.setLineDash(dashPattern);
  }

  // 1. Trazo exterior neón (globalAlpha = 0.25)
  ctx.globalAlpha = prevAlpha * 0.25;
  ctx.lineWidth = innerWidth + outerGlowExtra;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // 2. Trazo interior nítido
  ctx.globalAlpha = prevAlpha;
  ctx.lineWidth = innerWidth;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  if (dashPattern && dashPattern.length > 0) {
    ctx.setLineDash([]);
  }
}

/**
 * Dibuja un polígono cerrado usando la técnica de Doble Trazo Neón sin ctx.shadowBlur.
 */
function strokeDoubleNeonPolygon(
  ctx: AR2DRenderingContext,
  pts: ReadonlyArray<{ x: number; y: number }>,
  color: string,
  innerWidth = 2.2,
  outerGlowExtra = 5.5,
  dashPattern?: number[]
): void {
  if (pts.length < 3) return;
  const prevAlpha = ctx.globalAlpha;
  ctx.strokeStyle = color;
  if (dashPattern && dashPattern.length > 0) {
    ctx.setLineDash(dashPattern);
  }

  const tracePoly = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.closePath();
    ctx.stroke();
  };

  // 1. Halo exterior (alpha 0.25)
  ctx.globalAlpha = prevAlpha * 0.25;
  ctx.lineWidth = innerWidth + outerGlowExtra;
  tracePoly();

  // 2. Trazo interior nítido
  ctx.globalAlpha = prevAlpha;
  ctx.lineWidth = innerWidth;
  tracePoly();

  if (dashPattern && dashPattern.length > 0) {
    ctx.setLineDash([]);
  }
}

export class ARCanvasRendererEngine {
  private readonly lerpTrackMap = new Map<number, LerpTrackRecord>();
  private readonly textWidthCache = new Map<string, number>();
  private gyroX = 0;
  private gyroY = 0;
  private gyroEnabled = true;
  private renderedFramesCounter = 0;
  private lastFpsTickMs = performance.now();
  private measuredRenderFps = 60;

  public updateGyroOffset(gyro: ARInertialGyroState): void {
    this.gyroEnabled = gyro.enabled;
    if (!gyro.enabled) {
      this.gyroX = 0;
      this.gyroY = 0;
      return;
    }
    this.gyroX = gyro.offsetX;
    this.gyroY = gyro.offsetY;
  }

  public getMeasuredRenderFps(): number {
    return this.measuredRenderFps;
  }

  private measureCachedTextWidth(
    ctx: AR2DRenderingContext,
    font: string,
    text: string
  ): number {
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
   * Interpolación Lineal (LERP) + Filtro de Kalman 1D ejecutada a 60 FPS en cada tick de renderizado.
   */
  private interpolateDetectionAt60Fps(
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
    const existing = this.lerpTrackMap.get(det.trackId);
    const lerpAlpha = Math.min(0.65, Math.max(0.22, dtSec * 22));

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
          x:
            existing.keypoints[idx].x +
            (kp.x - existing.keypoints[idx].x) * lerpAlpha,
          y:
            existing.keypoints[idx].y +
            (kp.y - existing.keypoints[idx].y) * lerpAlpha,
        }));
      }

      existing.x = sx;
      existing.y = sy;
      existing.width = sW;
      existing.height = sH;
      existing.keypoints = smoothKps;
      existing.lastSeenMs = nowMs;
    } else {
      this.lerpTrackMap.set(det.trackId, {
        x: sx,
        y: sy,
        width: sW,
        height: sH,
        keypoints: smoothKps,
        lastSeenMs: nowMs,
        kx: new FastKalman1D(sx),
        ky: new FastKalman1D(sy),
        kw: new FastKalman1D(sW),
        kh: new FastKalman1D(sH),
      });
    }

    if (this.lerpTrackMap.size > 64) {
      for (const [tid, rec] of this.lerpTrackMap.entries()) {
        if (nowMs - rec.lastSeenMs > 1600) {
          this.lerpTrackMap.delete(tid);
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
   * Dibuja una Lupa Táctica de Precisión flotante encima del dedo/puntero cuando el usuario
   * arrastra un vértice del Polígono ROI (V1..V4) o un nodo de la Línea Virtual de Aforo (A/B).
   */
  private drawTacticalTouchLoupe(
    ctx: AR2DRenderingContext,
    loupe: NonNullable<ARCanvasRenderState['activePointerLoupe']>,
    sw: number,
    sh: number,
    roiPts?: Array<{ x: number; y: number }>,
    linePts?: { ax: number; ay: number; bx: number; by: number }
  ): void {
    if (!loupe.active) return;

    const radius = 42;
    const loupeX = Math.max(radius + 12, Math.min(sw - radius - 12, loupe.x));
    const loupeY =
      loupe.y > radius * 2 + 48
        ? loupe.y - radius - 34
        : Math.min(sh - radius - 12, loupe.y + radius + 34);

    ctx.save();

    // Fondo circular oscuro de alto contraste
    ctx.fillStyle = 'rgba(6, 9, 15, 0.92)';
    ctx.beginPath();
    ctx.arc(loupeX, loupeY, radius, 0, Math.PI * 2);
    ctx.fill();

    // Recorte circular (clip) para proyectar la geometría ampliada 2x alrededor del punto táctil
    ctx.save();
    ctx.beginPath();
    ctx.arc(loupeX, loupeY, radius - 2, 0, Math.PI * 2);
    ctx.clip();

    // Rejilla milimétrica interna de la lupa
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.lineWidth = 1;
    for (let offset = -32; offset <= 32; offset += 16) {
      ctx.beginPath();
      ctx.moveTo(loupeX - radius, loupeY + offset);
      ctx.lineTo(loupeX + radius, loupeY + offset);
      ctx.moveTo(loupeX + offset, loupeY - radius);
      ctx.lineTo(loupeX + offset, loupeY + radius);
      ctx.stroke();
    }

    // Dibujar segmentos vecinos ampliados 1.8x relativos al punto táctil
    const zoom = 1.8;
    const toLoupe = (px: number, py: number) => ({
      x: loupeX + (px - loupe.x) * zoom,
      y: loupeY + (py - loupe.y) * zoom,
    });

    if (roiPts && roiPts.length === 4) {
      const zoomedRoi = roiPts.map((p) => toLoupe(p.x, p.y));
      strokeDoubleNeonPolygon(ctx, zoomedRoi, loupe.accentColor, 2.0, 4.0, [5, 3]);
    }

    if (linePts) {
      const za = toLoupe(linePts.ax, linePts.ay);
      const zb = toLoupe(linePts.bx, linePts.by);
      strokeDoubleNeonSegment(
        ctx,
        za.x,
        za.y,
        zb.x,
        zb.y,
        '#38BDF8',
        2.0,
        4.0,
        [6, 4]
      );
    }

    // Mira óptica central de precisión dentro de la lupa
    ctx.strokeStyle = loupe.accentColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(loupeX - 18, loupeY);
    ctx.lineTo(loupeX - 5, loupeY);
    ctx.moveTo(loupeX + 5, loupeY);
    ctx.lineTo(loupeX + 18, loupeY);
    ctx.moveTo(loupeX, loupeY - 18);
    ctx.lineTo(loupeX, loupeY - 5);
    ctx.moveTo(loupeX, loupeY + 5);
    ctx.lineTo(loupeX, loupeY + 18);
    ctx.stroke();

    ctx.fillStyle = '#F8FAFC';
    ctx.beginPath();
    ctx.arc(loupeX, loupeY, 2.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    // Anillo perimetral de la lupa con doble trazo neón
    const prevAlpha = ctx.globalAlpha;
    ctx.strokeStyle = loupe.accentColor;
    ctx.globalAlpha = 0.25;
    ctx.lineWidth = 6.5;
    ctx.beginPath();
    ctx.arc(loupeX, loupeY, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.globalAlpha = prevAlpha;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(loupeX, loupeY, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Etiqueta de coordenadas normalizadas debajo de la lupa
    const coordText = `${loupe.label} · ${Math.round(
      (loupe.x / Math.max(1, sw)) * 100
    )}%, ${Math.round((loupe.y / Math.max(1, sh)) * 100)}%`;
    const font = '700 9.5px "JetBrains Mono", monospace';
    ctx.font = font;
    const tw = this.measureCachedTextWidth(ctx, font, coordText);
    const pw = Math.ceil(tw + 12);
    const ph = 18;
    const bx = loupeX - pw * 0.5;
    const by = loupeY + radius - 6;

    ctx.fillStyle = 'rgba(9, 13, 22, 0.96)';
    ctx.beginPath();
    ctx.roundRect(bx, by, pw, ph, 5);
    ctx.fill();

    ctx.strokeStyle = loupe.accentColor;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.roundRect(bx, by, pw, ph, 5);
    ctx.stroke();

    ctx.fillStyle = '#F8FAFC';
    ctx.textBaseline = 'middle';
    ctx.fillText(coordText, bx + 6, by + ph * 0.5 + 0.5);

    ctx.restore();
  }

  /**
   * Renderiza un fotograma completo de Realidad Aumentada a 60 FPS en cualquier contexto 2D.
   */
  public renderARFrame(
    ctx: AR2DRenderingContext,
    scene: ARCanvasRenderState,
    nowMs: number,
    dtSec: number
  ): void {
    const sw = scene.screenWidth;
    const sh = scene.screenHeight;
    const dpr = scene.dpr;
    if (sw <= 0 || sh <= 0) return;

    // Medición interna de FPS de renderizado
    this.renderedFramesCounter += 1;
    if (nowMs - this.lastFpsTickMs >= 500) {
      this.measuredRenderFps = Math.min(
        60,
        Math.round(
          (this.renderedFramesCounter * 1000) /
            Math.max(1, nowMs - this.lastFpsTickMs)
        )
      );
      this.renderedFramesCounter = 0;
      this.lastFpsTickMs = nowMs;
    }

    // Decaimiento inter-frame del offset inercial del giroscopio
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

    // 1. Superposición Térmica Espacial (Spatial Heatmap)
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

    // 2. Retícula Central de Horizonte Óptico AR
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

    // 3. POLÍGONO ROI INTERACTIVO DE 4 VÉRTICES + GUÍAS MAGNÉTICAS DE SNAPPING (Doble Trazo Neón)
    let roiPixelPoints: Array<{ x: number; y: number }> | undefined;
    if (
      scene.enableRoiPolygon &&
      scene.roiVerticesNorm &&
      scene.roiVerticesNorm.length === 4
    ) {
      const pts = scene.roiVerticesNorm.map((v) => ({
        x: v.x * sw,
        y: v.y * sh,
      }));
      roiPixelPoints = pts;
      const roiAlertOn = scene.activeRoiViolationInView;
      const roiStrokeColor = roiAlertOn ? '#EF4444' : '#F59E0B';
      const roiFillColor = roiAlertOn
        ? `rgba(239, 68, 68, ${0.18 + pulseFactor * 0.18})`
        : 'rgba(245, 158, 11, 0.11)';

      ctx.save();

      // Relleno translúcido del polígono
      ctx.fillStyle = roiFillColor;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.lineTo(pts[2].x, pts[2].y);
      ctx.lineTo(pts[3].x, pts[3].y);
      ctx.closePath();
      ctx.fill();

      // Diagonales espaciales internas
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

      // Contorno con técnica de Doble Trazo Neón (sin shadowBlur)
      strokeDoubleNeonPolygon(
        ctx,
        pts,
        roiStrokeColor,
        roiAlertOn ? 3.0 : 2.0,
        roiAlertOn ? 7.0 : 5.0,
        [8, 5]
      );

      // Etiqueta central del Polígono ROI
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
        : 'rgba(15, 23, 42, 0.88)';
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

      // Guías Magnéticas de Alineación (Snapping a Carriles, Ortogonales y Línea de Aforo)
      const hasActiveSnap =
        Boolean(scene.roiSnapGuides) && (scene.roiSnapGuides?.length ?? 0) > 0;

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

          // Línea láser con Doble Trazo Neón
          strokeDoubleNeonSegment(
            ctx,
            gx1,
            gy1,
            gx2,
            gy2,
            guide.color,
            1.8,
            5.0,
            [5, 4]
          );

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

          ctx.strokeStyle = guide.color;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.arc(sx, sy, 20, 0, Math.PI * 2);
          ctx.stroke();

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

      // 4 Vértices circulares arrastrables (V1..V4)
      for (let idx = 0; idx < pts.length; idx++) {
        const pt = pts[idx];
        const isDraggingThis = scene.activeRoiVertexIndex === idx;
        const isSnappedVertex = isDraggingThis && hasActiveSnap;
        const rad = isDraggingThis ? 15 : 11;

        ctx.fillStyle = isSnappedVertex
          ? 'rgba(34, 211, 238, 0.36)'
          : isDraggingThis
          ? 'rgba(245, 158, 11, 0.35)'
          : 'rgba(15, 23, 42, 0.88)';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, rad, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = isSnappedVertex
          ? '#22D3EE'
          : isDraggingThis
          ? '#FBBF24'
          : roiStrokeColor;
        ctx.lineWidth = isSnappedVertex ? 2.6 : 2.0;
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

    // 4. LÍNEA VIRTUAL DE CONTEO Y AFORO VEHICULAR (TRIPWIRE AR BIDIRECCIONAL)
    let linePixelPoints:
      | { ax: number; ay: number; bx: number; by: number }
      | undefined;
    if (scene.showCountingLine) {
      const ax = scene.lineEndpointsNorm.x1 * sw;
      const ay = scene.lineEndpointsNorm.y1 * sh;
      const bx = scene.lineEndpointsNorm.x2 * sw;
      const by = scene.lineEndpointsNorm.y2 * sh;
      linePixelPoints = { ax, ay, bx, by };

      const countStats = scene.countingStats;
      const msSinceFlash = nowMs - countStats.lastFlashTimestampMs;
      const isFlashing =
        countStats.lastFlashTimestampMs > 0 &&
        msSinceFlash >= 0 &&
        msSinceFlash < 550;
      const flashProgress = isFlashing ? 1 - msSinceFlash / 550 : 0;

      const linePrimaryColor = isFlashing
        ? '#22C55E'
        : themeId === 'industrial'
        ? '#FACC15'
        : '#38BDF8';

      ctx.save();

      // Doble Trazo Neón para la línea virtual de aforo
      strokeDoubleNeonSegment(
        ctx,
        ax,
        ay,
        bx,
        by,
        linePrimaryColor,
        isFlashing ? 3.2 : 2.2,
        isFlashing ? 9.0 : 5.5,
        [10, 6]
      );

      if (isFlashing && countStats.lastIntersectionPoint) {
        const ix = countStats.lastIntersectionPoint.x;
        const iy = countStats.lastIntersectionPoint.y;
        const rippleRadius = 10 + (1 - flashProgress) * 32;

        ctx.strokeStyle = `rgba(34, 197, 94, ${flashProgress})`;
        ctx.lineWidth = 2.4;
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
        ctx.lineWidth = 2.2;
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

    // 5. LÍNEAS DE ADVERTENCIA DE PROXIMIDAD PEATÓN-VEHÍCULO (Doble Trazo Neón)
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

        strokeDoubleNeonSegment(
          ctx,
          px,
          py,
          vx,
          vy,
          linkCol,
          isCrit ? 2.4 : 2.0,
          5.0,
          [6, 4]
        );

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

    // 6. OBJETOS RASTREADOS POR BYTETRACK + LERP 60 FPS + RETÍCULAS DOBLE TRAZO + ESQUELETO 17KP
    const activeList = scene.detections;
    const labelFont = theme.useMonospaceFont
      ? '700 10.5px "JetBrains Mono", monospace'
      : '600 11.5px "Plus Jakarta Sans", -apple-system, sans-serif';

    for (let i = 0; i < activeList.length; i++) {
      const det = activeList[i];
      const anchored = this.interpolateDetectionAt60Fps(
        det,
        nowMs,
        dtSec,
        gx,
        gy
      );

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

      const baseCatStyle =
        theme.categories[det.category] ?? theme.categories.object;

      // Codificación cromática por clase y estado vehicular (EN MOVIMIENTO, DETENIDO, EXCESO)
      const strokeColor = isCriticalAlert
        ? '#EF4444'
        : det.hasProximityRisk
        ? '#F59E0B'
        : isVehicle && det.motionStatus === 'DETENIDO'
        ? '#FBBF24'
        : baseCatStyle.stroke;

      const fillColor = isCriticalAlert
        ? `rgba(239, 68, 68, ${0.16 + pulseFactor * 0.14})`
        : isVehicle && det.motionStatus === 'DETENIDO'
        ? 'rgba(251, 191, 36, 0.10)'
        : baseCatStyle.fill;

      ctx.save();

      // 6.1 Estela de Movimiento (Motion Trails) con Doble Trazo Neón
      if (det.trail && det.trail.length >= 2) {
        const traceTrail = () => {
          ctx.beginPath();
          for (let tIdx = 0; tIdx < det.trail.length; tIdx++) {
            const tp = det.trail[tIdx];
            const tx = tp.x + gx;
            const ty = tp.y + gy;
            if (tIdx === 0) ctx.moveTo(tx, ty);
            else ctx.lineTo(tx, ty);
          }
          ctx.stroke();
        };

        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Halo exterior
        ctx.globalAlpha = 0.22;
        ctx.lineWidth = isCriticalAlert ? 6.0 : 4.8;
        traceTrail();

        // Línea interior punteada
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = isCriticalAlert ? 2.2 : 1.8;
        ctx.setLineDash([4, 4]);
        traceTrail();
        ctx.setLineDash([]);

        for (let tIdx = 1; tIdx < det.trail.length; tIdx += 3) {
          const tp = det.trail[tIdx];
          ctx.fillStyle = strokeColor;
          ctx.globalAlpha = ((tIdx + 1) / det.trail.length) * 0.78;
          ctx.beginPath();
          ctx.arc(tp.x + gx, tp.y + gy, 2.0, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // 6.2 Aura Pulsante en Alertas Críticas (EXCESO / ROI / CAÍDA)
      if (isCriticalAlert) {
        const auraPad = 4 + pulseFactor * 5;
        ctx.strokeStyle = `rgba(239, 68, 68, ${0.35 + pulseFactor * 0.45})`;
        ctx.lineWidth = 2.0;
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

      // 6.3 Relleno translúcido de la caja
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
      ctx.fill();

      // 6.4 Retículas Tácticas (Corner Brackets) con Doble Trazo Neón (0% shadowBlur)
      if (theme.useCornerBrackets) {
        ctx.strokeStyle = strokeColor;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = isCriticalAlert ? 1.6 : 1.0;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
        ctx.stroke();

        const cLen = Math.min(theme.cornerBracketLength, bw * 0.34, bh * 0.34);
        const traceBrackets = () => {
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
        };

        ctx.lineCap = 'square';
        // Trazo exterior neón (globalAlpha = 0.25)
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = isCriticalAlert ? 7.5 : 6.0;
        traceBrackets();

        // Trazo interior nítido de 2px
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = isCriticalAlert ? 2.8 : 2.0;
        traceBrackets();

        // Punto de mira en el centroide
        const cx = bx + bw * 0.5;
        const cy = by + bh * 0.5;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, isCriticalAlert ? 4.2 : 3.2, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.strokeStyle = strokeColor;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = theme.boxStrokeWidth + 4.5;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
        ctx.stroke();

        ctx.globalAlpha = 1.0;
        ctx.lineWidth = isCriticalAlert
          ? theme.boxStrokeWidth + 1.0
          : theme.boxStrokeWidth;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
        ctx.stroke();
      }

      // 6.5 Etiquetas Flotantes con Línea Guía (Callout Pins)
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

      // 6.6 Esqueleto Biomecánico Fluido: 17 Keypoints con Doble Trazo Neón
      if (
        !scene.onlyVehiclesMode &&
        scene.enablePose &&
        smoothKps.length === 17
      ) {
        const boneColor = isFall ? '#EF4444' : theme.skeletonBoneColor;
        ctx.lineCap = 'round';

        const traceSkeletonBones = () => {
          ctx.beginPath();
          for (let sIdx = 0; sIdx < SKELETON_CONNECTIONS.length; sIdx++) {
            const [idxA, idxB] = SKELETON_CONNECTIONS[sIdx];
            const kpA = smoothKps[idxA];
            const kpB = smoothKps[idxB];
            if (kpA && kpB && kpA.score > 0.5 && kpB.score > 0.5) {
              ctx.moveTo(kpA.x + gx, kpA.y + gy);
              ctx.lineTo(kpB.x + gx, kpB.y + gy);
            }
          }
          ctx.stroke();
        };

        ctx.strokeStyle = boneColor;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = isFall ? 6.5 : 5.2;
        traceSkeletonBones();

        ctx.globalAlpha = 1.0;
        ctx.lineWidth = isFall ? 2.8 : themeId === 'native' ? 1.6 : 2.1;
        traceSkeletonBones();

        for (let k = 0; k < smoothKps.length; k++) {
          const kp = smoothKps[k];
          if (kp.score > 0.5) {
            const kx = kp.x + gx;
            const ky = kp.y + gy;
            ctx.fillStyle = '#090D16';
            ctx.beginPath();
            ctx.arc(
              kx,
              ky,
              themeId === 'native' ? 2.8 : 4.0,
              0,
              Math.PI * 2
            );
            ctx.fill();

            ctx.fillStyle = isFall ? '#FCA5A5' : theme.skeletonJointColor;
            ctx.beginPath();
            ctx.arc(
              kx,
              ky,
              themeId === 'native' ? 2.0 : 2.8,
              0,
              Math.PI * 2
            );
            ctx.fill();
          }
        }
      }

      ctx.restore();
    }

    // 7. LUPA TÁCTICA DE PRECISIÓN (cuando el usuario arrastra un nodo ROI V1..V4 o Aforo A/B)
    if (scene.activePointerLoupe?.active) {
      this.drawTacticalTouchLoupe(
        ctx,
        scene.activePointerLoupe,
        sw,
        sh,
        roiPixelPoints,
        linePixelPoints
      );
    }

    ctx.restore();
  }
}

/**
 * Función de conveniencia agnóstica del contexto para invocar el renderizado AR
 * directamente desde el Web Worker o desde el bucle de Fallback en el hilo principal.
 */
export function renderARFrame(
  ctx: AR2DRenderingContext,
  state: ARCanvasRenderState,
  nowMs: number,
  dtSec: number,
  engineInstance: ARCanvasRendererEngine
): void {
  engineInstance.renderARFrame(ctx, state, nowMs, dtSec);
}
