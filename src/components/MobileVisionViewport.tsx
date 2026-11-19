import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  Camera,
  Compass,
  Eye,
  EyeOff,
  Flame,
  Layers,
  Radar,
  SlidersHorizontal,
  Sparkles,
  Video,
} from 'lucide-react';
import {
  CameraSourceMode,
  CategoryStats,
  GyroStabilizationTelemetry,
  PipelineMetrics,
  PipelineSettings,
  SKELETON_CONNECTIONS,
  TargetClass,
  TrackedObject,
} from '../engine/cvTypes';
import { RenderedSimEntity } from '../engine/sceneSimulation';
import {
  AR_THEMES,
  ARThemeSpec,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { SemanticCategory } from '../utils/visionPostProcessor';
import crosswalkRearCamImg from '../assets/images/urban_crosswalk_rear_cam_1795094285385.jpg';
import plazaBikeCamImg from '../assets/images/urban_plaza_bike_cam_1795094285161.jpg';

interface MobileVisionViewportProps {
  settings: PipelineSettings;
  metrics: PipelineMetrics;
  stats: CategoryStats;
  tracks: TrackedObject[];
  entities: RenderedSimEntity[];
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  gyroTelemetry: GyroStabilizationTelemetry;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  webcamError: string | null;
  onViewportResize: (width: number, height: number) => void;
  onUpdateSettings: (patch: Partial<PipelineSettings>) => void;
  onCycleARTheme: () => void;
  onSimulateGyroImpulse: (deltaYawRadS: number, deltaPitchRadS: number) => void;
  onOpenDiagnostics: () => void;
  onResetHeatmap: () => void;
}

function mapClassToCategory(cls: TargetClass): SemanticCategory {
  if (cls === 'person') return 'person';
  if (cls === 'car' || cls === 'bus' || cls === 'truck') return 'vehicle';
  if (cls === 'bicycle' || cls === 'motorcycle') return 'bicycle';
  if (cls === 'dog') return 'animal';
  return 'object';
}

export const MobileVisionViewport: React.FC<MobileVisionViewportProps> = ({
  settings,
  metrics,
  stats,
  tracks,
  entities,
  heatmapPoints,
  gyroTelemetry,
  videoRef,
  webcamError,
  onViewportResize,
  onUpdateSettings,
  onCycleARTheme,
  onSimulateGyroImpulse,
  onOpenDiagnostics,
  onResetHeatmap,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const radarCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgImagesRef = useRef<Record<string, HTMLImageElement>>({});
  const pointerDragRef = useRef<{ active: boolean; lastX: number; lastY: number }>({
    active: false,
    lastX: 0,
    lastY: 0,
  });

  const [canvasSize, setCanvasSize] = useState({ width: 412, height: 840 });
  const [radarExpanded, setRadarExpanded] = useState(true);
  const [showFullHeatmapOverlay, setShowFullHeatmapOverlay] = useState(false);

  const activeTheme: ARThemeSpec = AR_THEMES[settings.arTheme] ?? AR_THEMES.tactical;

  // Preload realistic urban camera background textures
  useEffect(() => {
    const imgCrosswalk = new Image();
    imgCrosswalk.src = crosswalkRearCamImg;
    bgImagesRef.current['rear-crosswalk'] = imgCrosswalk;

    const imgBike = new Image();
    imgBike.src = plazaBikeCamImg;
    bgImagesRef.current['bike-boulevard'] = imgBike;
  }, []);

  // Measure full-screen edge-to-edge viewport
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.max(280, Math.floor(entry.contentRect.width));
        const h = Math.max(320, Math.floor(entry.contentRect.height));
        setCanvasSize({ width: w, height: h });
        onViewportResize(w, h);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [onViewportResize]);

  // Draw Mini-Radar / Spatial Heatmap in bottom-right collapsible widget
  useEffect(() => {
    const rCanvas = radarCanvasRef.current;
    if (!rCanvas || !radarExpanded) return;
    const ctx = rCanvas.getContext('2d');
    if (!ctx) return;

    const w = rCanvas.width;
    const h = rCanvas.height;
    ctx.clearRect(0, 0, w, h);

    // Subtle radar grid
    ctx.strokeStyle =
      settings.arTheme === 'industrial'
        ? 'rgba(250, 204, 21, 0.22)'
        : 'rgba(56, 189, 248, 0.20)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.85, w * 0.36, Math.PI, 0);
    ctx.arc(w * 0.5, h * 0.85, w * 0.68, Math.PI, 0);
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.5, h);
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    // Spatial density accumulation points
    for (const pt of heatmapPoints.slice(-90)) {
      const px = pt.xNorm * w;
      const py = pt.yNorm * h;
      const rad = 11;
      const grad = ctx.createRadialGradient(px, py, 1, px, py, rad);
      grad.addColorStop(0, 'rgba(249, 115, 22, 0.34)');
      grad.addColorStop(0.55, 'rgba(56, 189, 248, 0.16)');
      grad.addColorStop(1, 'rgba(56, 189, 248, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fill();
    }

    // Active live tracked targets as crisp radar blips
    for (const tr of tracks) {
      const cat = mapClassToCategory(tr.className);
      const col = activeTheme.categories[cat]?.stroke ?? '#38BDF8';
      const bx = Math.max(4, Math.min(w - 4, (tr.centerX / Math.max(1, canvasSize.width)) * w));
      const by = Math.max(4, Math.min(h - 4, (tr.centerY / Math.max(1, canvasSize.height)) * h));

      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(bx, by, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [heatmapPoints, tracks, radarExpanded, settings.arTheme, activeTheme, canvasSize]);

  // Draw Full-Screen 100% Edge-to-Edge AR Skia Simulator (Camera + 60Hz Gyro Offset + 3 AR Themes)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvasSize;
    ctx.clearRect(0, 0, width, height);

    // Offset predictivo inter-frame del giroscopio (60Hz)
    const gx = settings.enableGyro ? gyroTelemetry.offsetX : 0;
    const gy = settings.enableGyro ? gyroTelemetry.offsetY : 0;

    // 1. Background Camera Feed (StyleSheet.absoluteFill + resizeMode="cover")
    const useLiveWebcam =
      settings.cameraMode === 'webcam' &&
      videoRef.current &&
      videoRef.current.readyState >= 2 &&
      !webcamError;

    if (useLiveWebcam && videoRef.current) {
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(videoRef.current, 0, 0, width, height);
      ctx.restore();
    } else {
      const activeMode =
        settings.cameraMode === 'webcam' ? 'rear-crosswalk' : settings.cameraMode;
      const bgImg = bgImagesRef.current[activeMode];

      if (bgImg && bgImg.complete && bgImg.naturalWidth > 0) {
        // Aspect-fill (resizeMode="cover") con ligero paralaje inverso del giroscopio
        const imgAspect = bgImg.naturalWidth / bgImg.naturalHeight;
        const canvasAspect = width / height;
        let drawW = width * 1.04;
        let drawH = height * 1.04;
        let offsetX = -width * 0.02 + gx * 0.35;
        let offsetY = -height * 0.02 + gy * 0.35;

        if (imgAspect > canvasAspect) {
          drawH = height * 1.04;
          drawW = drawH * imgAspect;
          offsetX = (width - drawW) / 2 + gx * 0.35;
        } else {
          drawW = width * 1.04;
          drawH = drawW / imgAspect;
          offsetY = (height - drawH) / 2 + gy * 0.35;
        }

        ctx.drawImage(bgImg, offsetX, offsetY, drawW, drawH);

        const vignette = ctx.createRadialGradient(
          width * 0.5,
          height * 0.5,
          Math.min(width, height) * 0.25,
          width * 0.5,
          height * 0.5,
          Math.max(width, height) * 0.78
        );
        vignette.addColorStop(0, 'rgba(11, 15, 23, 0.04)');
        vignette.addColorStop(1, 'rgba(11, 15, 23, 0.42)');
        ctx.fillStyle = vignette;
        ctx.fillRect(0, 0, width, height);
      } else {
        const skyGrad = ctx.createLinearGradient(0, 0, 0, height);
        skyGrad.addColorStop(0, '#0B111E');
        skyGrad.addColorStop(1, '#0F172A');
        ctx.fillStyle = skyGrad;
        ctx.fillRect(0, 0, width, height);
      }

      // Subtle dynamic lighting actors
      for (const ent of entities) {
        drawSimulatedActor(ctx, ent, width, height, gx * 0.35, gy * 0.35);
      }
    }

    // Optional Full-Screen Spatial Heatmap Layer
    if (showFullHeatmapOverlay && heatmapPoints.length > 0) {
      ctx.save();
      for (const pt of heatmapPoints) {
        const px = pt.xNorm * width + gx;
        const py = pt.yNorm * height + gy;
        const radius = 34;
        const radial = ctx.createRadialGradient(px, py, 2, px, py, radius);
        radial.addColorStop(0, `rgba(249, 115, 22, ${0.18 * pt.weight})`);
        radial.addColorStop(0.5, `rgba(56, 189, 248, ${0.09 * pt.weight})`);
        radial.addColorStop(1, 'rgba(56, 189, 248, 0)');
        ctx.fillStyle = radial;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // 2. Retícula Central de Horizonte AR (Táctico & Industrial)
    if (activeTheme.showHorizonReticle) {
      const rCx = width * 0.5 + gx * 0.55;
      const rCy = height * 0.5 + gy * 0.55;
      const reticleColor =
        settings.arTheme === 'industrial'
          ? 'rgba(250, 204, 21, 0.36)'
          : 'rgba(56, 189, 248, 0.34)';

      ctx.save();
      ctx.strokeStyle = reticleColor;
      ctx.lineWidth = 1.1;

      // Center optical ring
      ctx.beginPath();
      ctx.arc(rCx, rCy, 24, 0, Math.PI * 2);
      ctx.stroke();

      // Horizon pitch/roll wings
      ctx.beginPath();
      ctx.moveTo(rCx - 48, rCy);
      ctx.lineTo(rCx - 14, rCy);
      ctx.moveTo(rCx + 14, rCy);
      ctx.lineTo(rCx + 48, rCy);
      ctx.moveTo(rCx, rCy - 32);
      ctx.lineTo(rCx, rCy - 12);
      ctx.moveTo(rCx, rCy + 12);
      ctx.lineTo(rCx, rCy + 32);
      ctx.stroke();

      // Center precision dot
      ctx.fillStyle = reticleColor;
      ctx.beginPath();
      ctx.arc(rCx, rCy, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // 3. Renderizado de Objetos y Esqueletos con LERP + Offset Predictivo de Giroscopio (Zero-Lag)
    for (const track of tracks) {
      const category = mapClassToCategory(track.className);
      const catStyle = activeTheme.categories[category] ?? activeTheme.categories.object;

      // Aplicar compensación inter-frame (gx, gy) del giroscopio
      const bx = Math.max(4, Math.min(width - 12, track.screenBox.x + gx));
      const by = Math.max(4, Math.min(height - 12, track.screenBox.y + gy));
      const bw = Math.max(12, Math.min(width - bx - 2, track.screenBox.width));
      const bh = Math.max(12, Math.min(height - by - 2, track.screenBox.height));

      ctx.save();

      // Translucent interior fill
      ctx.fillStyle = catStyle.fill;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, activeTheme.cornerRadius);
      ctx.fill();

      if (activeTheme.useCornerBrackets) {
        // MODO 1: Táctico / Futurista Cian — Esquinas de enfoque óptico (Corner Brackets) + brillo neón
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, activeTheme.cornerRadius);
        ctx.stroke();

        const cLen = Math.min(activeTheme.cornerBracketLength, bw * 0.34, bh * 0.34);
        ctx.strokeStyle = catStyle.stroke;
        ctx.lineWidth = activeTheme.boxStrokeWidth + 0.5;
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
      } else {
        // MODO 2 (Nativo Frosted White) o MODO 3 (Alto Contraste Industrial)
        ctx.strokeStyle = catStyle.stroke;
        ctx.lineWidth = activeTheme.boxStrokeWidth;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, activeTheme.cornerRadius);
        ctx.stroke();
      }

      // 4. Etiquetas Flotantes AR (Callout Pins con Línea Guía / Píldoras Frosted / Industrial)
      const confPct = Math.round(track.confidence * 100);
      const distMeters = estimateObjectDistanceMeters(category, bh, height);
      const labelText = activeTheme.showDistanceEstimate
        ? `#${track.trackId} ${track.className.toUpperCase()} ${confPct}% • ~${distMeters.toFixed(1)}m`
        : `${track.className} ${confPct}%`;

      ctx.font = activeTheme.useMonospaceFont
        ? '700 10.5px "JetBrains Mono", monospace'
        : '600 11px "Plus Jakarta Sans", -apple-system, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const pillW = Math.ceil(textWidth + 14);
      const pillH = 20;

      if (activeTheme.showLeaderLinePins) {
        // Callout Pin flotante con línea guía oblicua y punto de anclaje
        const anchorX = bx + Math.min(26, bw * 0.28);
        const anchorY = by;
        const elbowX = Math.max(8, Math.min(width - pillW - 8, anchorX + 14));
        const elbowY = Math.max(30, anchorY - 14);

        ctx.strokeStyle = catStyle.stroke;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(anchorX, anchorY);
        ctx.lineTo(elbowX, elbowY);
        ctx.stroke();

        ctx.fillStyle = catStyle.stroke;
        ctx.beginPath();
        ctx.arc(anchorX, anchorY, 2.5, 0, Math.PI * 2);
        ctx.fill();

        const pillY = elbowY - pillH;
        ctx.fillStyle = activeTheme.pillBackground;
        ctx.beginPath();
        ctx.roundRect(elbowX, pillY, pillW, pillH, 5);
        ctx.fill();

        ctx.strokeStyle = catStyle.stroke;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(elbowX, pillY, pillW, pillH, 5);
        ctx.stroke();

        ctx.fillStyle = activeTheme.pillTextColor;
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, elbowX + 7, pillY + pillH / 2 + 0.5);
      } else {
        // Píldora directa (Nativo Frosted Glass o Industrial Sólido #090D16)
        const pillX = Math.max(6, Math.min(width - pillW - 6, bx));
        const pillY = by > pillH + 8 ? by - pillH - 4 : by + 4;
        const pillRadius = settings.arTheme === 'native' ? 10 : 2;

        ctx.fillStyle =
          settings.arTheme === 'industrial' ? '#090D16' : activeTheme.pillBackground;
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, pillRadius);
        ctx.fill();

        ctx.strokeStyle =
          settings.arTheme === 'industrial'
            ? catStyle.stroke
            : 'rgba(255, 255, 255, 0.28)';
        ctx.lineWidth = settings.arTheme === 'industrial' ? 1.8 : 1.0;
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, pillRadius);
        ctx.stroke();

        ctx.fillStyle =
          settings.arTheme === 'industrial' ? catStyle.stroke : activeTheme.pillTextColor;
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, pillX + 7, pillY + pillH / 2 + 0.5);
      }

      // 5. Esqueleto Biomecánico 17 Keypoints (Estabilizado con Giroscopio)
      if (settings.enablePose && track.keypoints.length === 17) {
        if (settings.arTheme !== 'native') {
          ctx.strokeStyle = activeTheme.skeletonBoneColor;
          ctx.lineWidth = settings.arTheme === 'industrial' ? 2.4 : 2.1;
          ctx.lineCap = 'round';

          for (const [idxA, idxB] of SKELETON_CONNECTIONS) {
            const kpA = track.keypoints[idxA];
            const kpB = track.keypoints[idxB];
            if (kpA && kpB && kpA.score > 0.5 && kpB.score > 0.5) {
              ctx.beginPath();
              ctx.moveTo(kpA.x + gx, kpA.y + gy);
              ctx.lineTo(kpB.x + gx, kpB.y + gy);
              ctx.stroke();
            }
          }
        } else {
          // Modo Nativo: líneas ultra sutiles translúcidas
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.42)';
          ctx.lineWidth = 1.3;
          for (const [idxA, idxB] of SKELETON_CONNECTIONS) {
            const kpA = track.keypoints[idxA];
            const kpB = track.keypoints[idxB];
            if (kpA && kpB && kpA.score > 0.5 && kpB.score > 0.5) {
              ctx.beginPath();
              ctx.moveTo(kpA.x + gx, kpA.y + gy);
              ctx.lineTo(kpB.x + gx, kpB.y + gy);
              ctx.stroke();
            }
          }
        }

        for (const kp of track.keypoints) {
          if (kp.score > 0.5) {
            const kx = kp.x + gx;
            const ky = kp.y + gy;
            ctx.fillStyle = '#090D16';
            ctx.beginPath();
            ctx.arc(kx, ky, settings.arTheme === 'native' ? 2.8 : 3.8, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = activeTheme.skeletonJointColor;
            ctx.beginPath();
            ctx.arc(kx, ky, settings.arTheme === 'native' ? 1.9 : 2.6, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      ctx.restore();
    }
  }, [
    canvasSize,
    tracks,
    entities,
    heatmapPoints,
    gyroTelemetry,
    settings,
    activeTheme,
    showFullHeatmapOverlay,
    videoRef,
    webcamError,
  ]);

  const cycleCameraLens = () => {
    const order: CameraSourceMode[] = ['rear-crosswalk', 'bike-boulevard', 'webcam'];
    const idx = order.indexOf(settings.cameraMode);
    onUpdateSettings({ cameraMode: order[(idx + 1) % order.length] });
  };

  const lensBadgeLabel =
    settings.cameraMode === 'rear-crosswalk'
      ? '1x Wide'
      : settings.cameraMode === 'bike-boulevard'
      ? '2x Tele'
      : 'Live Cam';

  // Permite arrastrar en cualquier parte del visor AR para simular giros rápidos de cámara (Gyro Zero-Lag)
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('button')) return;
    pointerDragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!pointerDragRef.current.active) return;
    const dx = e.clientX - pointerDragRef.current.lastX;
    const dy = e.clientY - pointerDragRef.current.lastY;
    pointerDragRef.current.lastX = e.clientX;
    pointerDragRef.current.lastY = e.clientY;
    onSimulateGyroImpulse(-dx * 0.045, dy * 0.045);
  };

  const handlePointerUp = () => {
    pointerDragRef.current.active = false;
  };

  const glassWidgetStyle: React.CSSProperties = {
    backgroundColor: 'rgba(11, 15, 23, 0.55)',
    borderColor: activeTheme.hudBorderColor,
  };

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className="relative w-full h-full overflow-hidden bg-[#06090F] cursor-grab active:cursor-grabbing"
    >
      {/* Hidden video element for live WebRTC stream */}
      <video ref={videoRef} className="hidden" playsInline muted />

      {/* 100% Edge-to-Edge Full-Screen Camera + Skia AR Canvas (StyleSheet.absoluteFill) */}
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        className="absolute inset-0 w-full h-full block pointer-events-none"
      />

      {/* =================================================================== */}
      {/* HUD FLOTANTE TRANSLÚCIDO (GLASSMORPHISM rgba(11, 15, 23, 0.55))     */}
      {/* =================================================================== */}

      {/* 1. BARRA SUPERIOR FLOTANTE (Respeta Safe Area Top) */}
      <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Píldora LIVE AR • XX FPS */}
        <div
          style={glassWidgetStyle}
          className="pointer-events-auto h-9 px-3 rounded-full backdrop-blur-md border flex items-center gap-2 shadow-lg"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              settings.isRunning
                ? 'bg-[#22C55E] shadow-[0_0_8px_#22C55E] animate-pulse'
                : 'bg-amber-400'
            }`}
          />
          <span className="text-[11px] font-bold tracking-wider uppercase text-slate-100 whitespace-nowrap">
            LIVE AR
          </span>
          <span className="text-slate-500">•</span>
          <span className="font-mono text-xs font-bold text-sky-400 tabular-nums whitespace-nowrap">
            {metrics.realFps.toFixed(1)} FPS
          </span>
        </div>

        {/* Controles Rápidos Superiores: Estilo AR (Táctico/Nativo/Industrial) + Esqueleto ON/OFF + Lente 1x */}
        <div className="pointer-events-auto flex items-center gap-1.5 flex-wrap">
          {/* Selector de Estilo Visual AR */}
          <button
            type="button"
            onClick={onCycleARTheme}
            style={glassWidgetStyle}
            title="Alternar estilo visual AR (Táctico / Nativo / Industrial)"
            className="h-9 px-3 rounded-full backdrop-blur-md border text-xs font-semibold text-slate-100 hover:bg-slate-900/75 flex items-center gap-1.5 transition-all cursor-pointer shadow-lg"
          >
            <Sparkles
              className={`w-3.5 h-3.5 ${
                settings.arTheme === 'tactical'
                  ? 'text-sky-400'
                  : settings.arTheme === 'industrial'
                  ? 'text-yellow-400'
                  : 'text-white'
              }`}
            />
            <span className="whitespace-nowrap">AR: {activeTheme.shortLabel}</span>
          </button>

          {/* Toggle Rápido de Capa Esqueleto 17kp ON/OFF */}
          <button
            type="button"
            onClick={() => onUpdateSettings({ enablePose: !settings.enablePose })}
            style={glassWidgetStyle}
            className={`h-9 px-3 rounded-full backdrop-blur-md border text-xs font-medium flex items-center gap-1.5 transition-all cursor-pointer shadow-lg ${
              settings.enablePose ? 'text-emerald-300' : 'text-slate-400'
            }`}
          >
            {settings.enablePose ? (
              <Eye className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <EyeOff className="w-3.5 h-3.5 text-slate-400" />
            )}
            <span className="whitespace-nowrap">
              Esqueleto {settings.enablePose ? 'ON' : 'OFF'}
            </span>
          </button>

          {/* Selector de Lente (1x / 2x / Live) */}
          <button
            type="button"
            onClick={cycleCameraLens}
            style={glassWidgetStyle}
            className="h-9 px-3 rounded-full backdrop-blur-md border text-xs font-mono font-semibold text-sky-300 hover:bg-slate-900/75 flex items-center gap-1.5 transition-all cursor-pointer shadow-lg"
          >
            {settings.cameraMode === 'webcam' ? (
              <Video className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Camera className="w-3.5 h-3.5 text-sky-400" />
            )}
            <span className="whitespace-nowrap">{lensBadgeLabel}</span>
          </button>
        </div>
      </div>

      {/* 2. PÍLDORAS DE CONTEO COMPACTAS FLOTANTES (Personas, Vehículos, Animales/Otros) */}
      <div className="absolute top-14 left-3 z-20 flex flex-wrap items-center gap-1.5 pointer-events-none">
        {/* Personas */}
        <div
          style={glassWidgetStyle}
          className="pointer-events-auto h-7 px-2.5 rounded-full backdrop-blur-md border flex items-center gap-1.5 shadow-md"
        >
          <span className="w-2 h-2 rounded-full bg-[#38BDF8]" />
          <span className="text-[11px] font-medium text-slate-200">Personas</span>
          <span className="font-mono text-xs font-bold text-white tabular-nums">
            {stats.persons}
          </span>
        </div>

        {/* Vehículos (Autos/Buses/Camiones + Bicis/Motos) */}
        <div
          style={glassWidgetStyle}
          className="pointer-events-auto h-7 px-2.5 rounded-full backdrop-blur-md border flex items-center gap-1.5 shadow-md"
        >
          <span className="w-2 h-2 rounded-full bg-[#F97316]" />
          <span className="text-[11px] font-medium text-slate-200">Vehículos</span>
          <span className="font-mono text-xs font-bold text-white tabular-nums">
            {stats.vehicles + stats.bicycles}
          </span>
        </div>

        {/* Animales / Otros (80 clases COCO) */}
        <div
          style={glassWidgetStyle}
          className="pointer-events-auto h-7 px-2.5 rounded-full backdrop-blur-md border flex items-center gap-1.5 shadow-md"
        >
          <span className="w-2 h-2 rounded-full bg-[#A855F7]" />
          <span className="text-[11px] font-medium text-slate-200">Animales/Otros</span>
          <span className="font-mono text-xs font-bold text-white tabular-nums">
            {stats.animalsOther}
          </span>
        </div>
      </div>

      {/* Aviso flotante si webcam falló */}
      {webcamError && settings.cameraMode === 'webcam' && (
        <div
          style={glassWidgetStyle}
          className="absolute top-24 left-3 z-20 px-3 py-1.5 rounded-xl backdrop-blur-md border text-[11px] text-amber-300 shadow-lg"
        >
          {webcamError}
        </div>
      )}

      {/* 3. MINI-RADAR / HEATMAP FLOTANTE COLAPSABLE (Esquina Inferior Derecha) */}
      <div className="absolute bottom-16 right-3 z-20 flex flex-col items-end gap-1.5 pointer-events-auto">
        {radarExpanded ? (
          <div
            style={glassWidgetStyle}
            className="w-[154px] rounded-2xl backdrop-blur-md border p-2.5 shadow-2xl flex flex-col gap-1.5"
          >
            <div className="flex items-center justify-between gap-1">
              <div className="flex items-center gap-1.5">
                <Radar className="w-3.5 h-3.5 text-sky-400" />
                <span className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-200">
                  Mini-Radar AR
                </span>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setShowFullHeatmapOverlay((v) => !v)}
                  title="Superponer mapa térmico en visor"
                  className={`p-1 rounded-md text-[10px] transition-colors cursor-pointer ${
                    showFullHeatmapOverlay
                      ? 'bg-orange-500/25 text-orange-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Flame className="w-3 h-3" />
                </button>
                <button
                  type="button"
                  onClick={() => setRadarExpanded(false)}
                  className="px-1.5 py-0.5 rounded text-[10px] font-mono text-slate-400 hover:text-white cursor-pointer"
                >
                  —
                </button>
              </div>
            </div>

            <div className="relative w-full h-[82px] rounded-xl bg-[#070A10]/70 border border-white/10 overflow-hidden">
              <canvas
                ref={radarCanvasRef}
                width={134}
                height={82}
                className="w-full h-full block"
              />
              <div className="absolute bottom-1 left-1.5 right-1.5 flex items-center justify-between text-[9.5px] font-mono text-slate-400">
                <span>{metrics.density}</span>
                <button
                  type="button"
                  onClick={onResetHeatmap}
                  className="text-sky-400 hover:underline cursor-pointer"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setRadarExpanded(true)}
            style={glassWidgetStyle}
            className="h-9 px-3 rounded-full backdrop-blur-md border text-xs font-medium text-slate-200 hover:text-white flex items-center gap-1.5 shadow-lg cursor-pointer"
          >
            <Radar className="w-3.5 h-3.5 text-sky-400" />
            <span>Radar ({tracks.length})</span>
          </button>
        )}
      </div>

      {/* 4. BARRA INFERIOR FLOTANTE (Estado AR + Gyro Active y Botón Ajustes / Telemetría) */}
      <div className="absolute bottom-3 left-3 right-3 z-20 flex items-center justify-between gap-2 pointer-events-none">
        {/* Estado del Sistema: AR + Gyro Active (60Hz Zero-Lag) */}
        <button
          type="button"
          onClick={() => onUpdateSettings({ enableGyro: !settings.enableGyro })}
          style={glassWidgetStyle}
          title="Activar/Desactivar compensación inercial de giroscopio a 60Hz (arrastra la pantalla para probar)"
          className="pointer-events-auto h-10 px-3.5 rounded-full backdrop-blur-md border flex items-center gap-2 shadow-xl cursor-pointer transition-all hover:bg-slate-900/75"
        >
          <Compass
            className={`w-4 h-4 ${
              settings.enableGyro ? 'text-emerald-400 animate-spin' : 'text-slate-500'
            }`}
            style={{ animationDuration: '9s' }}
          />
          <div className="flex flex-col items-start leading-tight">
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-semibold text-slate-100 whitespace-nowrap">
                {settings.enableGyro ? 'AR + Gyro Active' : 'Gyro Paused'}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-400/30">
                60Hz
              </span>
            </div>
            <span className="text-[10px] font-mono text-slate-400 tabular-nums">
              Δx:{gyroTelemetry.offsetX >= 0 ? '+' : ''}
              {gyroTelemetry.offsetX.toFixed(1)}px Δy:{gyroTelemetry.offsetY >= 0 ? '+' : ''}
              {gyroTelemetry.offsetY.toFixed(1)}px • {metrics.timings.totalMs.toFixed(1)}ms
            </span>
          </div>
        </button>

        {/* Botón Ajustes / Telemetría que abre el Bottom Sheet (Gráfica 20s, Latencia EMA, Tabla ByteTrack y Código RN) */}
        <button
          type="button"
          onClick={onOpenDiagnostics}
          style={glassWidgetStyle}
          className="pointer-events-auto h-10 px-4 rounded-full backdrop-blur-md border hover:bg-slate-900/80 text-slate-100 text-xs font-semibold flex items-center gap-2 shadow-xl transition-all cursor-pointer"
        >
          <SlidersHorizontal className="w-3.5 h-3.5 text-sky-400" />
          <span className="whitespace-nowrap">Ajustes / Telemetría</span>
          <span className="hidden sm:inline-flex items-center gap-1 font-mono text-[10.5px] px-1.5 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-400/30">
            <Activity className="w-3 h-3" />
            {metrics.thermalMode}
          </span>
        </button>
      </div>
    </div>
  );
};

function drawSimulatedActor(
  ctx: CanvasRenderingContext2D,
  ent: RenderedSimEntity,
  canvasWidth: number,
  canvasHeight: number,
  offsetX: number,
  offsetY: number
) {
  const w = ent.wNorm * canvasWidth;
  const h = ent.hNorm * canvasHeight;
  const x = (ent.xNorm - ent.wNorm * 0.5) * canvasWidth + offsetX;
  const y = (ent.yNorm - ent.hNorm * 0.5) * canvasHeight + offsetY;
  ctx.save();

  ctx.fillStyle = 'rgba(0, 0, 0, 0.26)';
  ctx.beginPath();
  ctx.ellipse(x + w * 0.5, y + h * 0.97, w * 0.4, Math.max(3, h * 0.045), 0, 0, Math.PI * 2);
  ctx.fill();

  if (ent.className === 'car' || ent.className === 'bus' || ent.className === 'truck') {
    const headlightGrad = ctx.createRadialGradient(
      x + w * 0.25,
      y + h * 0.72,
      1,
      x + w * 0.25,
      y + h * 0.72,
      w * 0.22
    );
    headlightGrad.addColorStop(0, 'rgba(254, 240, 138, 0.28)');
    headlightGrad.addColorStop(1, 'rgba(254, 240, 138, 0)');
    ctx.fillStyle = headlightGrad;
    ctx.fillRect(x, y + h * 0.45, w, h * 0.55);
  }

  ctx.restore();
}
