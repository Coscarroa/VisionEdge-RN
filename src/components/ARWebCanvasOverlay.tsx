import React, { useEffect, useRef } from 'react';
import {
  AR_THEMES,
  ARThemeId,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { SKELETON_CONNECTIONS } from '../engine/cvTypes';
import { WebKeypoint2D, WebTrackedDetection } from '../hooks/useWebVisionEngine';

export interface WebGyroState {
  omegaX: number; // rad/s Pitch
  omegaY: number; // rad/s Yaw
  omegaZ: number; // rad/s Roll
  offsetX: number; // px inter-frame X
  offsetY: number; // px inter-frame Y
  hasHardwareSensor: boolean;
}

export interface ARWebCanvasOverlayProps {
  detections: WebTrackedDetection[];
  arTheme: ARThemeId;
  enablePose: boolean;
  enableGyro: boolean;
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  screenWidth: number;
  screenHeight: number;
  /** Referencia mutable compartida para telemetría de giroscopio a 60Hz */
  gyroStateRef: React.MutableRefObject<WebGyroState>;
}

interface LerpTrackRecord {
  x: number;
  y: number;
  width: number;
  height: number;
  keypoints: WebKeypoint2D[];
  lastSeenMs: number;
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
  gyroStateRef,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectionsRef = useRef<WebTrackedDetection[]>(detections);
  detectionsRef.current = detections;

  const lerpMapRef = useRef<Map<number, LerpTrackRecord>>(new Map());
  const prevOrientationRef = useRef<{ alpha: number | null; beta: number | null; gamma: number | null }>({
    alpha: null,
    beta: null,
    gamma: null,
  });

  const propsRef = useRef({
    arTheme,
    enablePose,
    enableGyro,
    showHeatmapOverlay,
    heatmapPoints,
    screenWidth,
    screenHeight,
  });
  propsRef.current = {
    arTheme,
    enablePose,
    enableGyro,
    showHeatmapOverlay,
    heatmapPoints,
    screenWidth,
    screenHeight,
  };

  // 1. Al llegar un nuevo lote de inferencia real, amortiguamos sincronizadamente el offset acumulado del giroscopio
  useEffect(() => {
    gyroStateRef.current.offsetX *= 0.22;
    gyroStateRef.current.offsetY *= 0.22;
  }, [detections, gyroStateRef]);

  // 2. Suscripción real a sensores web: devicemotion (velocidad angular rad/s) y deviceorientation (ángulos Euler)
  useEffect(() => {
    const focalPx = Math.max(320, propsRef.current.screenWidth) * 0.85;
    let lastMotionTs = performance.now();

    const handleDeviceMotion = (ev: DeviceMotionEvent) => {
      if (!propsRef.current.enableGyro) return;
      const rot = ev.rotationRate;
      if (!rot) return;

      const now = performance.now();
      const dtSec = Math.min(0.05, Math.max(0.004, (now - lastMotionTs) * 0.001));
      lastMotionTs = now;

      // En Web API DeviceMotionEvent.rotationRate viene en grados/segundo (deg/s)
      const degToRad = Math.PI / 180;
      const pitchRadS = (rot.alpha ?? 0) * degToRad;
      const yawRadS = (rot.beta ?? 0) * degToRad;
      const rollRadS = (rot.gamma ?? 0) * degToRad;

      if (Math.abs(pitchRadS) > 0.001 || Math.abs(yawRadS) > 0.001) {
        gyroStateRef.current.hasHardwareSensor = true;
      }

      // Zona muerta (deadzone) de 0.015 rad/s para filtrar ruido térmico del MEMS
      const wy = Math.abs(yawRadS) > 0.015 ? yawRadS : 0;
      const wx = Math.abs(pitchRadS) > 0.015 ? pitchRadS : 0;

      const g = gyroStateRef.current;
      g.omegaX = wx;
      g.omegaY = wy;
      g.omegaZ = rollRadS;

      const maxClamp = 68;
      g.offsetX = Math.max(-maxClamp, Math.min(maxClamp, (g.offsetX - wy * dtSec * focalPx) * 0.93));
      g.offsetY = Math.max(-maxClamp, Math.min(maxClamp, (g.offsetY + wx * dtSec * focalPx) * 0.93));
    };

    const handleDeviceOrientation = (ev: DeviceOrientationEvent) => {
      if (!propsRef.current.enableGyro || gyroStateRef.current.hasHardwareSensor) return;
      if (ev.gamma === null || ev.beta === null) return;

      const prev = prevOrientationRef.current;
      if (prev.gamma !== null && prev.beta !== null) {
        const dGamma = ev.gamma - prev.gamma;
        const dBeta = ev.beta - prev.beta;

        if (Math.abs(dGamma) < 25 && Math.abs(dBeta) < 25) {
          const g = gyroStateRef.current;
          g.offsetX = Math.max(-64, Math.min(64, (g.offsetX - dGamma * 2.4) * 0.92));
          g.offsetY = Math.max(-64, Math.min(64, (g.offsetY + dBeta * 2.4) * 0.92));
        }
      }
      prevOrientationRef.current = { alpha: ev.alpha, beta: ev.beta, gamma: ev.gamma };
    };

    window.addEventListener('devicemotion', handleDeviceMotion, { passive: true });
    window.addEventListener('deviceorientation', handleDeviceOrientation, { passive: true });

    return () => {
      window.removeEventListener('devicemotion', handleDeviceMotion);
      window.removeEventListener('deviceorientation', handleDeviceOrientation);
    };
  }, [gyroStateRef]);

  // 3. Bucle de Renderizado AR a 60 FPS con window.devicePixelRatio + Suavizado LERP + Offset de Giroscopio
  useEffect(() => {
    let rafId: number;

    const renderARLoop = (now: number) => {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        const {
          arTheme: themeId,
          enablePose: poseOn,
          enableGyro: gyroOn,
          showHeatmapOverlay: heatmapOn,
          heatmapPoints: hmPts,
          screenWidth: sw,
          screenHeight: sh,
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
          const g = gyroStateRef.current;

          // Decaimiento suave de offset en cada frame de 16ms
          g.offsetX *= 0.95;
          g.offsetY *= 0.95;

          const gx = gyroOn ? g.offsetX : 0;
          const gy = gyroOn ? g.offsetY : 0;

          // A) Superposición térmica opcional (Spatial Heatmap)
          if (heatmapOn && hmPts.length > 0) {
            for (let i = 0; i < hmPts.length; i++) {
              const pt = hmPts[i];
              const px = pt.xNorm * sw + gx;
              const py = pt.yNorm * sh + gy;
              const rad = 36;
              const grad = ctx.createRadialGradient(px, py, 2, px, py, rad);
              grad.addColorStop(0, `rgba(249, 115, 22, ${0.22 * pt.weight})`);
              grad.addColorStop(0.55, `rgba(56, 189, 248, ${0.10 * pt.weight})`);
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

          // C) Interpolación Lineal (LERP) + Trazado de las 80 Clases de COCO y Esqueleto 17kp
          const activeList = detectionsRef.current;
          const lerpAlpha = 0.34;
          const cache = lerpMapRef.current;

          for (let i = 0; i < activeList.length; i++) {
            const det = activeList[i];
            const prev = cache.get(det.trackId);

            let sx = det.screenBox.x;
            let sy = det.screenBox.y;
            let sW = det.screenBox.width;
            let sH = det.screenBox.height;
            let smoothKps = det.keypoints;

            if (prev && now - prev.lastSeenMs < 450) {
              sx = prev.x + (det.screenBox.x - prev.x) * lerpAlpha;
              sy = prev.y + (det.screenBox.y - prev.y) * lerpAlpha;
              sW = prev.width + (det.screenBox.width - prev.width) * lerpAlpha;
              sH = prev.height + (det.screenBox.height - prev.height) * lerpAlpha;

              if (det.keypoints.length === 17 && prev.keypoints.length === 17) {
                smoothKps = det.keypoints.map((kp, kIdx) => ({
                  ...kp,
                  x: prev.keypoints[kIdx].x + (kp.x - prev.keypoints[kIdx].x) * lerpAlpha,
                  y: prev.keypoints[kIdx].y + (kp.y - prev.keypoints[kIdx].y) * lerpAlpha,
                }));
              }
            }

            cache.set(det.trackId, {
              x: sx,
              y: sy,
              width: sW,
              height: sH,
              keypoints: smoothKps,
              lastSeenMs: now,
            });

            // Aplicar compensación inercial del giroscopio
            const bx = Math.max(4, Math.min(sw - 12, sx + gx));
            const by = Math.max(4, Math.min(sh - 12, sy + gy));
            const bw = Math.max(10, Math.min(sw - bx - 2, sW));
            const bh = Math.max(10, Math.min(sh - by - 2, sH));

            const catStyle = theme.categories[det.category] ?? theme.categories.object;

            ctx.save();

            // Relleno translúcido
            ctx.fillStyle = catStyle.fill;
            ctx.beginPath();
            ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
            ctx.fill();

            if (theme.useCornerBrackets) {
              // Borde guía tenue + Corner Brackets con brillo neón (#38BDF8 / categoría)
              ctx.strokeStyle = 'rgba(56, 189, 248, 0.20)';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
              ctx.stroke();

              const cLen = Math.min(theme.cornerBracketLength, bw * 0.34, bh * 0.34);
              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = theme.boxStrokeWidth + 0.6;
              ctx.shadowColor = catStyle.stroke;
              ctx.shadowBlur = 9;
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
            } else {
              // Modo Minimalista Nativo o Alto Contraste Industrial
              ctx.strokeStyle = catStyle.stroke;
              ctx.lineWidth = theme.boxStrokeWidth;
              ctx.beginPath();
              ctx.roundRect(bx, by, bw, bh, theme.cornerRadius);
              ctx.stroke();
            }

            // Etiqueta Flotante con Línea Guía (Callout Pin) y Distancia Estimada (~Xm)
            const confPct = Math.round(det.confidence * 100);
            const distMeters = estimateObjectDistanceMeters(det.category, bh, sh);
            const labelText = theme.showDistanceEstimate
              ? `#${det.trackId} ${det.className.toUpperCase()} ${confPct}% · ~${distMeters.toFixed(1)}m`
              : `${det.className} ${confPct}%`;

            ctx.font = theme.useMonospaceFont
              ? '700 11px "JetBrains Mono", monospace'
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

            // Esqueleto Biomecánico Verde Neón (#22C55E) de 17 Puntos Clave
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

          // Limpiar entradas antiguas del caché LERP
          if (cache.size > 48) {
            for (const [tid, rec] of cache.entries()) {
              if (now - rec.lastSeenMs > 1500) {
                cache.delete(tid);
              }
            }
          }

          ctx.restore();
        }
      }

      rafId = window.requestAnimationFrame(renderARLoop);
    };

    rafId = window.requestAnimationFrame(renderARLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [gyroStateRef]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-screen h-dvh pointer-events-none select-none z-10"
    />
  );
};
