import React, { useEffect, useRef } from 'react';
import {
  AR_THEMES,
  ARThemeId,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { SKELETON_CONNECTIONS } from '../engine/cvTypes';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { InertialARState } from '../hooks/useInertialARTracker';
import { isCocoVehicleClass } from '../utils/VehicleSpeedEstimator';
import {
  globalVirtualCountingLine,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';

export interface ARWebCanvasOverlayProps {
  detections: OnnxTrackedDetection[];
  arTheme: ARThemeId;
  enablePose: boolean;
  enableGyro: boolean;
  onlyVehiclesMode: boolean;
  showCountingLine: boolean;
  lineEndpointsNorm: VirtualLineEndpointsNorm;
  activeDragHandle: 'a' | 'b' | null;
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  screenWidth: number;
  screenHeight: number;
  /** Referencia mutable compartida con useInertialARTracker (Kalman + LERP 60/120Hz) */
  inertialStateRef: React.MutableRefObject<InertialARState>;
  computeAnchoredDetectionState: (
    det: OnnxTrackedDetection,
    nowMs: number,
    dtSec: number
  ) => {
    x: number;
    y: number;
    width: number;
    height: number;
    keypoints: OnnxKeypoint2D[];
    gyroOffsetX: number;
    gyroOffsetY: number;
  };
  decayInertialOffsetOnRenderFrame: () => { gx: number; gy: number };
}

export const ARWebCanvasOverlay: React.FC<ARWebCanvasOverlayProps> = ({
  detections,
  arTheme,
  enablePose,
  enableGyro,
  onlyVehiclesMode,
  showCountingLine,
  lineEndpointsNorm,
  activeDragHandle,
  showHeatmapOverlay,
  heatmapPoints,
  screenWidth,
  screenHeight,
  computeAnchoredDetectionState,
  decayInertialOffsetOnRenderFrame,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectionsRef = useRef<OnnxTrackedDetection[]>(detections);
  detectionsRef.current = detections;

  const propsRef = useRef({
    arTheme,
    enablePose,
    enableGyro,
    onlyVehiclesMode,
    showCountingLine,
    lineEndpointsNorm,
    activeDragHandle,
    showHeatmapOverlay,
    heatmapPoints,
    screenWidth,
    screenHeight,
    computeAnchoredDetectionState,
    decayInertialOffsetOnRenderFrame,
  });
  propsRef.current = {
    arTheme,
    enablePose,
    enableGyro,
    onlyVehiclesMode,
    showCountingLine,
    lineEndpointsNorm,
    activeDragHandle,
    showHeatmapOverlay,
    heatmapPoints,
    screenWidth,
    screenHeight,
    computeAnchoredDetectionState,
    decayInertialOffsetOnRenderFrame,
  };

  useEffect(() => {
    let rafId: number;
    let lastFrameTime = performance.now();

    const renderARLoop = (now: number) => {
      const dtSec = Math.min(0.05, Math.max(0.004, (now - lastFrameTime) * 0.001));
      lastFrameTime = now;

      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        const {
          arTheme: themeId,
          enablePose: poseOn,
          onlyVehiclesMode: vehiclesOnly,
          showCountingLine: tripwireOn,
          lineEndpointsNorm: lineNorm,
          activeDragHandle: dragNode,
          showHeatmapOverlay: heatmapOn,
          heatmapPoints: hmPts,
          screenWidth: sw,
          screenHeight: sh,
          computeAnchoredDetectionState: anchorFn,
          decayInertialOffsetOnRenderFrame: decayFn,
        } = propsRef.current;

        if (ctx && sw > 0 && sh > 0) {
          const dpr = Math.min(2.5, window.devicePixelRatio || 1);
          const targetPixelW = Math.round(sw * dpr);
          const targetPixelH = Math.round(sh * dpr);

          if (canvas.width !== targetPixelW || canvas.height !== targetPixelH) {
            canvas.width = targetPixelW;
            canvas.height = targetPixelH;
          }

          ctx.save();
          ctx.scale(dpr, dpr);
          ctx.clearRect(0, 0, sw, sh);

          const theme = AR_THEMES[themeId] ?? AR_THEMES.tactical;
          const { gx, gy } = decayFn();

          // A) Superposición térmica opcional (Spatial Heatmap)
          if (heatmapOn && hmPts.length > 0) {
            for (let i = 0; i < hmPts.length; i++) {
              const pt = hmPts[i];
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
                : vehiclesOnly
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

          // C) LÍNEA VIRTUAL DE CONTEO Y AFORO VEHICULAR (TRIPWIRE AR BIDIRECCIONAL)
          if (tripwireOn) {
            const ax = lineNorm.x1 * sw;
            const ay = lineNorm.y1 * sh;
            const bx = lineNorm.x2 * sw;
            const by = lineNorm.y2 * sh;

            const countStats = globalVirtualCountingLine.getStats();
            const msSinceFlash = now - countStats.lastFlashTimestampMs;
            const isFlashing = countStats.lastFlashTimestampMs > 0 && msSinceFlash < 550;
            const flashProgress = isFlashing ? 1 - msSinceFlash / 550 : 0;

            const linePrimaryColor = isFlashing
              ? '#22C55E'
              : themeId === 'industrial'
              ? '#FACC15'
              : '#38BDF8';

            ctx.save();

            // Destello luminoso sobre la línea virtual en el instante exacto del cruce
            ctx.strokeStyle = isFlashing
              ? `rgba(34, 197, 94, ${0.4 + flashProgress * 0.55})`
              : 'rgba(56, 189, 248, 0.25)';
            ctx.lineWidth = isFlashing ? 11 + flashProgress * 7 : 7;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();

            // Línea central láser de aforo
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

            // Onda expansiva en el punto exacto de intersección (computeSegmentIntersection)
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

            // Etiqueta central sobre la línea virtual con conteo ↑ Entrada / ↓ Salida
            const midX = (ax + bx) * 0.5;
            const midY = (ay + by) * 0.5;
            const badgeText = isFlashing
              ? `✓ CRUCE #${countStats.lastCrossingTrackId} (${countStats.lastCrossingDirection}) · AFORO: ${countStats.total}`
              : `LÍNEA AFORO AR · ↑${countStats.inbound} ↓${countStats.outbound} (Arrastra Nodos A/B)`;

            ctx.font = '700 10.5px "JetBrains Mono", monospace';
            const tw = ctx.measureText(badgeText).width;
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

            // Nodos circulares interactivos en ambos extremos (A y B) para arrastrar con el dedo
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

            drawHandleNode(ax, ay, 'A', dragNode === 'a');
            drawHandleNode(bx, by, 'B', dragNode === 'b');

            ctx.restore();
          }

          // D) Renderizado de Objetos Rastreados por ByteTrack + Cinemática (km/h) + Alerta de Exceso (#EF4444)
          const activeList = detectionsRef.current;
          const pulseFactor = 0.5 + 0.5 * Math.sin(now * 0.012); // Pulso a ~2Hz para alerta roja

          for (let i = 0; i < activeList.length; i++) {
            const det = activeList[i];
            const anchored = anchorFn(det, now, dtSec);

            const bx = Math.max(4, Math.min(sw - 12, anchored.x));
            const by = Math.max(4, Math.min(sh - 12, anchored.y));
            const bw = Math.max(10, Math.min(sw - bx - 2, anchored.width));
            const bh = Math.max(10, Math.min(sh - by - 2, anchored.height));
            const smoothKps = anchored.keypoints;

            const isVehicle = isCocoVehicleClass(det.classId, det.className);
            const isOverspeed = det.isOverspeed;

            // Si hay exceso de velocidad, el color cambia inmediatamente a Rojo Carmesí Pulsante (#EF4444)
            const baseCatStyle = theme.categories[det.category] ?? theme.categories.object;
            const strokeColor = isOverspeed ? '#EF4444' : baseCatStyle.stroke;
            const fillColor = isOverspeed
              ? `rgba(239, 68, 68, ${0.16 + pulseFactor * 0.14})`
              : baseCatStyle.fill;
            const glowColor = isOverspeed ? 'rgba(239, 68, 68, 0.65)' : baseCatStyle.glow;

            ctx.save();

            // 1. Estela de Movimiento (Motion Trails): Línea punteada degradada con las últimas 15 posiciones
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
              ctx.lineWidth = isOverspeed ? 2.6 : 2.0;
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

            // 2. Aura de Advertencia Pulsante cuando un vehículo excede speedLimitKmh
            if (isOverspeed) {
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

            // 3. Relleno translúcido de la caja
            ctx.fillStyle = fillColor;
            ctx.beginPath();
            ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
            ctx.fill();

            // 4. Retículas Tácticas Adaptables (Corner Brackets con Brillo Neón o Rojo Carmesí Pulsante)
            if (theme.useCornerBrackets) {
              ctx.strokeStyle = glowColor;
              ctx.lineWidth = isOverspeed ? 1.5 : 0.9;
              ctx.beginPath();
              ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
              ctx.stroke();

              const cLen = Math.min(theme.cornerBracketLength, bw * 0.34, bh * 0.34);
              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = isOverspeed
                ? theme.boxStrokeWidth + 1.4
                : theme.boxStrokeWidth + 0.6;
              ctx.shadowColor = strokeColor;
              ctx.shadowBlur = isOverspeed ? 14 + pulseFactor * 8 : 10;
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

              // Punto de anclaje central (Target Lock Centroid)
              const cx = bx + bw * 0.5;
              const cy = by + bh * 0.5;
              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.arc(cx, cy, isOverspeed ? 4.2 : 3.2, 0, Math.PI * 2);
              ctx.stroke();
            } else {
              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = isOverspeed ? theme.boxStrokeWidth + 1.2 : theme.boxStrokeWidth;
              ctx.beginPath();
              ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
              ctx.stroke();
            }

            // 5. Etiquetas Flotantes con Línea Guía (Callout Pins):
            //    - Si excede velocidad: "⚠️ CAR #12 • 78 km/h • EXCESO"
            //    - Si es vehículo detenido: "CAR #12 • DETENIDO (0 km/h) · ~4.2m"
            //    - Si es vehículo en movimiento: "CAR #12 • 46 km/h • EN MOVIMIENTO · ~5.8m"
            const confPct = Math.round(det.confidence * 100);
            const distMeters = estimateObjectDistanceMeters(det.category, bh, sh);
            const classUpper = det.className.toUpperCase();
            const crossedTag = det.hasCrossedLine ? ' · ✓AFORO' : '';

            let labelText: string;
            if (isOverspeed) {
              labelText = `⚠️ ${classUpper} #${det.trackId} • ${Math.round(det.speedKmh)} km/h • EXCESO${crossedTag}`;
            } else if (isVehicle) {
              if (det.motionStatus === 'DETENIDO') {
                labelText = `${classUpper} #${det.trackId} • DETENIDO · ~${distMeters.toFixed(1)}m${crossedTag}`;
              } else {
                labelText = `${classUpper} #${det.trackId} • ${Math.round(det.speedKmh)} km/h • EN MOVIMIENTO${crossedTag}`;
              }
            } else {
              const speedText =
                det.speedMetersPerSec >= 0.2
                  ? `${det.speedMetersPerSec.toFixed(1)}m/s`
                  : `${Math.round(det.speedPxPerSec)}px/s`;
              labelText = theme.showDistanceEstimate
                ? `#${det.trackId} ${classUpper} ${confPct}% · ${speedText} · ~${distMeters.toFixed(1)}m`
                : `#${det.trackId} ${det.className} ${confPct}%`;
            }

            ctx.font = theme.useMonospaceFont
              ? '700 10.5px "JetBrains Mono", monospace'
              : '600 11.5px "Plus Jakarta Sans", -apple-system, sans-serif';
            const textW = ctx.measureText(labelText).width;
            const pillW = Math.ceil(textW + 16);
            const pillH = 22;

            const pillBgColor = isOverspeed
              ? 'rgba(127, 29, 29, 0.94)'
              : det.motionStatus === 'DETENIDO' && isVehicle
              ? 'rgba(30, 41, 59, 0.90)'
              : theme.pillBackground;

            const pillTextCol = isOverspeed ? '#FEE2E2' : theme.pillTextColor;

            if (theme.showLeaderLinePins) {
              const anchorX = bx + Math.min(28, bw * 0.25);
              const anchorY = by;
              const elbowX = Math.max(8, Math.min(sw - pillW - 8, anchorX + 14));
              const elbowY = Math.max(32, anchorY - 14);

              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = isOverspeed ? 1.8 : 1.4;
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
              ctx.lineWidth = isOverspeed ? 1.8 : 1.2;
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

              ctx.fillStyle = isOverspeed
                ? 'rgba(127, 29, 29, 0.95)'
                : themeId === 'industrial'
                ? '#090D16'
                : pillBgColor;
              ctx.beginPath();
              ctx.roundRect(pillX, pillY, pillW, pillH, rad);
              ctx.fill();

              ctx.strokeStyle =
                isOverspeed || themeId === 'industrial'
                  ? strokeColor
                  : 'rgba(255, 255, 255, 0.26)';
              ctx.lineWidth = isOverspeed || themeId === 'industrial' ? 1.8 : 1;
              ctx.beginPath();
              ctx.roundRect(pillX, pillY, pillW, pillH, rad);
              ctx.stroke();

              ctx.fillStyle = isOverspeed
                ? '#FEE2E2'
                : themeId === 'industrial'
                ? strokeColor
                : pillTextCol;
              ctx.textBaseline = 'middle';
              ctx.fillText(labelText, pillX + 8, pillY + pillH * 0.5 + 0.5);
            }

            // 6. Esqueleto Biomecánico Fluido: 17 Keypoints (Pausado automáticamente en Modo Solo Vehículos)
            if (!vehiclesOnly && poseOn && smoothKps.length === 17) {
              ctx.strokeStyle = theme.skeletonBoneColor;
              ctx.lineWidth = themeId === 'native' ? 1.5 : 2.3;
              ctx.lineCap = 'round';

              for (const [idxA, idxB] of SKELETON_CONNECTIONS) {
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

                  ctx.fillStyle = theme.skeletonJointColor;
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

      rafId = window.requestAnimationFrame(renderARLoop);
    };

    rafId = window.requestAnimationFrame(renderARLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-screen h-dvh pointer-events-none select-none z-10"
    />
  );
};
