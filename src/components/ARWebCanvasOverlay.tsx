import React, { useEffect, useRef } from 'react';
import {
  AR_THEMES,
  ARThemeId,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { SKELETON_CONNECTIONS } from '../engine/cvTypes';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { InertialARState } from '../hooks/useInertialARTracker';

export interface ARWebCanvasOverlayProps {
  detections: OnnxTrackedDetection[];
  arTheme: ARThemeId;
  enablePose: boolean;
  enableGyro: boolean;
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
    showHeatmapOverlay,
    heatmapPoints,
    screenWidth,
    screenHeight,
    computeAnchoredDetectionState,
    decayInertialOffsetOnRenderFrame,
  };

  // Bucle de Renderizado AR HiDPI a 60/120 FPS con Anclaje Espacial Inercial (Kalman + LERP)
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

          // C) Renderizado de Objetos Rastreados por ByteTrack + Kalman/LERP
          const activeList = detectionsRef.current;

          for (let i = 0; i < activeList.length; i++) {
            const det = activeList[i];
            const anchored = anchorFn(det, now, dtSec);

            const bx = Math.max(4, Math.min(sw - 12, anchored.x));
            const by = Math.max(4, Math.min(sh - 12, anchored.y));
            const bw = Math.max(10, Math.min(sw - bx - 2, anchored.width));
            const bh = Math.max(10, Math.min(sh - by - 2, anchored.height));
            const smoothKps = anchored.keypoints;

            // Color por Categoría Semántica:
            // Cian (#38BDF8) Personas | Naranja (#F97316) Vehículos | Esmeralda (#10B981) Animales | Violeta (#A855F7) Objetos
            const catStyle = theme.categories[det.category] ?? theme.categories.object;

            ctx.save();

            // 1. Estela de Movimiento (Motion Trails): Línea punteada degradada con las últimas 15 posiciones del centroide
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
              grad.addColorStop(0.5, catStyle.glow);
              grad.addColorStop(1, catStyle.stroke);

              ctx.save();
              ctx.strokeStyle = grad;
              ctx.lineWidth = 2.0;
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

              // Nodos históricos discretos a lo largo de la trayectoria
              for (let tIdx = 1; tIdx < det.trail.length; tIdx += 3) {
                const tp = det.trail[tIdx];
                const alpha = (tIdx + 1) / det.trail.length;
                ctx.fillStyle = catStyle.stroke;
                ctx.globalAlpha = alpha * 0.75;
                ctx.beginPath();
                ctx.arc(tp.x + gx, tp.y + gy, 2.0, 0, Math.PI * 2);
                ctx.fill();
              }
              ctx.restore();
            }

            // 2. Relleno translúcido de la caja
            ctx.fillStyle = catStyle.fill;
            ctx.beginPath();
            ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
            ctx.fill();

            // 3. Retículas Tácticas Adaptables (Corner Brackets con Brillo Neón según categoría)
            if (theme.useCornerBrackets) {
              ctx.strokeStyle = catStyle.glow;
              ctx.lineWidth = 0.9;
              ctx.beginPath();
              ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
              ctx.stroke();

              const cLen = Math.min(theme.cornerBracketLength, bw * 0.34, bh * 0.34);
              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = theme.boxStrokeWidth + 0.6;
              ctx.shadowColor = catStyle.stroke;
              ctx.shadowBlur = 10;
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
              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = 1.1;
              ctx.beginPath();
              ctx.arc(cx, cy, 3.2, 0, Math.PI * 2);
              ctx.stroke();
            } else {
              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = theme.boxStrokeWidth;
              ctx.beginPath();
              ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
              ctx.stroke();
            }

            // 4. Etiquetas Flotantes con Línea Guía (Callout Pins):
            //    Muestra clase, ID único (#1), confianza exacta (94%), velocidad relativa y distancia estimada (~2.8m)
            const confPct = Math.round(det.confidence * 100);
            const distMeters = estimateObjectDistanceMeters(det.category, bh, sh);
            const speedText =
              det.speedMetersPerSec >= 0.2
                ? `${det.speedMetersPerSec.toFixed(1)}m/s`
                : `${Math.round(det.speedPxPerSec)}px/s`;

            const labelText = theme.showDistanceEstimate
              ? `#${det.trackId} ${det.className.toUpperCase()} ${confPct}% · ${speedText} · ~${distMeters.toFixed(1)}m`
              : `#${det.trackId} ${det.className} ${confPct}%`;

            ctx.font = theme.useMonospaceFont
              ? '700 10.5px "JetBrains Mono", monospace'
              : '600 11.5px "Plus Jakarta Sans", -apple-system, sans-serif';
            const textW = ctx.measureText(labelText).width;
            const pillW = Math.ceil(textW + 16);
            const pillH = 22;

            if (theme.showLeaderLinePins) {
              const anchorX = bx + Math.min(28, bw * 0.25);
              const anchorY = by;
              const elbowX = Math.max(8, Math.min(sw - pillW - 8, anchorX + 14));
              const elbowY = Math.max(32, anchorY - 14);

              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = 1.4;
              ctx.beginPath();
              ctx.moveTo(anchorX, anchorY);
              ctx.lineTo(elbowX, elbowY);
              ctx.stroke();

              ctx.fillStyle = catStyle.stroke;
              ctx.beginPath();
              ctx.arc(anchorX, anchorY, 2.6, 0, Math.PI * 2);
              ctx.fill();

              const pillY = elbowY - pillH;
              ctx.fillStyle = theme.pillBackground;
              ctx.beginPath();
              ctx.roundRect(elbowX, pillY, pillW, pillH, 5);
              ctx.fill();

              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.roundRect(elbowX, pillY, pillW, pillH, 5);
              ctx.stroke();

              ctx.fillStyle = theme.pillTextColor;
              ctx.textBaseline = 'middle';
              ctx.fillText(labelText, elbowX + 8, pillY + pillH * 0.5 + 0.5);
            } else {
              const pillX = Math.max(6, Math.min(sw - pillW - 6, bx));
              const pillY = by > pillH + 8 ? by - pillH - 4 : by + 4;
              const rad = themeId === 'native' ? 10 : 3;

              ctx.fillStyle = themeId === 'industrial' ? '#090D16' : theme.pillBackground;
              ctx.beginPath();
              ctx.roundRect(pillX, pillY, pillW, pillH, rad);
              ctx.fill();

              ctx.strokeStyle =
                themeId === 'industrial' ? catStyle.stroke : 'rgba(255, 255, 255, 0.26)';
              ctx.lineWidth = themeId === 'industrial' ? 1.8 : 1;
              ctx.beginPath();
              ctx.roundRect(pillX, pillY, pillW, pillH, rad);
              ctx.stroke();

              ctx.fillStyle =
                themeId === 'industrial' ? catStyle.stroke : theme.pillTextColor;
              ctx.textBaseline = 'middle';
              ctx.fillText(labelText, pillX + 8, pillY + pillH * 0.5 + 0.5);
            }

            // 5. Esqueleto Biomecánico Fluido: 17 Keypoints Articulares con Conexiones Suavizadas
            if (poseOn && smoothKps.length === 17) {
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
