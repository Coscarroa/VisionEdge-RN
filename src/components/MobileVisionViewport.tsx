import React, { useEffect, useRef, useState } from 'react';
import { SlidersHorizontal, Video, VideoOff } from 'lucide-react';
import {
  CLASS_COLOR_MAP,
  CategoryStats,
  PipelineMetrics,
  PipelineSettings,
  SKELETON_CONNECTIONS,
  TimeSeriesPoint,
  TrackedObject,
} from '../engine/cvTypes';
import { RenderedSimEntity } from '../engine/sceneSimulation';
import { useResponsiveLayout } from '../hooks/useResponsiveLayout';

interface MobileVisionViewportProps {
  settings: PipelineSettings;
  tracks: TrackedObject[];
  entities: RenderedSimEntity[];
  stats: CategoryStats;
  metrics: PipelineMetrics;
  latencyHistory: number[];
  timeSeries: TimeSeriesPoint[];
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  webcamError: string | null;
  onViewportResize: (width: number, height: number) => void;
  onOpenSettings: () => void;
  onToggleCameraMode: () => void;
}

const CROSSWALK_IMAGE = '/src/assets/images/urban_crosswalk_rear_cam_1795094285385.jpg';
const BOULEVARD_IMAGE = '/src/assets/images/urban_plaza_bike_cam_1795094285161.jpg';

function drawRoundedRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

export const MobileVisionViewport: React.FC<MobileVisionViewportProps> = ({
  settings,
  tracks,
  entities,
  stats,
  metrics,
  latencyHistory,
  timeSeries,
  heatmapPoints,
  videoRef,
  webcamError,
  onViewportResize,
  onOpenSettings,
  onToggleCameraMode,
}) => {
  const viewfinderRef = useRef<HTMLDivElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [viewportSize, setViewportSize] = useState({ width: 390, height: 300 });
  const layout = useResponsiveLayout();

  // Measure actual flex-1 viewfinder dimensions so Skia-like overlays match 1:1 on any phone/screen
  useEffect(() => {
    const el = viewfinderRef.current;
    if (!el) return;

    const updateSize = (w: number, h: number) => {
      const nextW = Math.max(240, Math.round(w));
      const nextH = Math.max(150, Math.round(h));
      setViewportSize((prev) =>
        prev.width === nextW && prev.height === nextH ? prev : { width: nextW, height: nextH }
      );
      onViewportResize(nextW, nextH);
    };

    const rect = el.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      updateSize(rect.width, rect.height);
    }

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        updateSize(entry.contentRect.width, entry.contentRect.height);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [onViewportResize]);

  // Draw 60 FPS Skia-equivalent bounding boxes, confidence tags, and neon green pose skeleton
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const W = viewportSize.width;
    const H = viewportSize.height;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // Subtle animated subject silhouettes aligned to active detections
    if (settings.cameraMode !== 'webcam') {
      for (const ent of entities) {
        const cx = ent.xNorm * W;
        const cy = ent.yNorm * H;
        const w = ent.wNorm * W;
        const h = ent.hNorm * H;

        ctx.save();
        if (ent.className === 'person') {
          ctx.fillStyle = 'rgba(15, 23, 42, 0.52)';
          ctx.beginPath();
          ctx.ellipse(cx, cy + h * 0.48, w * 0.38, 3.5, 0, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = 'rgba(226, 232, 240, 0.22)';
          ctx.beginPath();
          ctx.arc(cx, cy - h * 0.36, Math.max(3.5, w * 0.16), 0, Math.PI * 2);
          ctx.fill();

          drawRoundedRectPath(ctx, cx - w * 0.22, cy - h * 0.24, w * 0.44, h * 0.46, 4);
          ctx.fill();
        } else if (ent.className === 'car' || ent.className === 'bus') {
          ctx.fillStyle =
            ent.className === 'bus' ? 'rgba(239, 68, 68, 0.20)' : 'rgba(249, 115, 22, 0.20)';
          drawRoundedRectPath(ctx, cx - w * 0.48, cy - h * 0.42, w * 0.96, h * 0.84, 5);
          ctx.fill();
        } else if (ent.className === 'bicycle') {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.48)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(cx - w * 0.28, cy + h * 0.2, Math.max(4, h * 0.22), 0, Math.PI * 2);
          ctx.arc(cx + w * 0.28, cy + h * 0.2, Math.max(4, h * 0.22), 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // Draw detection overlays
    for (const track of tracks) {
      const palette = CLASS_COLOR_MAP[track.className];
      const { x, y, width, height } = track.screenBox;

      ctx.fillStyle = palette.fill;
      drawRoundedRectPath(ctx, x, y, width, height, 8);
      ctx.fill();

      ctx.strokeStyle = palette.stroke;
      ctx.lineWidth = 2;
      drawRoundedRectPath(ctx, x, y, width, height, 8);
      ctx.stroke();

      const label = `${track.className} #${track.trackId} ${track.confidence.toFixed(2)}`;
      ctx.font = '600 10.5px "JetBrains Mono", monospace';
      const textWidth = ctx.measureText(label).width;
      const pillW = textWidth + 12;
      const pillH = 18;
      const pillX = Math.min(W - pillW - 4, Math.max(2, x));
      const pillY = y >= pillH + 4 ? y - pillH - 2 : y + 2;

      ctx.fillStyle = 'rgba(11, 15, 23, 0.84)';
      drawRoundedRectPath(ctx, pillX, pillY, pillW, pillH, 5);
      ctx.fill();
      ctx.strokeStyle = palette.stroke;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = palette.stroke;
      ctx.fillText(label, pillX + 6, pillY + 12.5);

      if (settings.enablePose && track.className === 'person' && track.keypoints.length === 17) {
        ctx.save();
        ctx.strokeStyle = '#22C55E';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.shadowColor = 'rgba(34, 197, 94, 0.55)';
        ctx.shadowBlur = 4;

        for (const [idxA, idxB] of SKELETON_CONNECTIONS) {
          const kpA = track.keypoints[idxA];
          const kpB = track.keypoints[idxB];
          if (!kpA || !kpB || kpA.score <= 0.5 || kpB.score <= 0.5) continue;
          ctx.beginPath();
          ctx.moveTo(kpA.x, kpA.y);
          ctx.lineTo(kpB.x, kpB.y);
          ctx.stroke();
        }

        ctx.shadowBlur = 0;
        for (const kp of track.keypoints) {
          if (kp.score <= 0.5) continue;
          ctx.fillStyle = '#22C55E';
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 2.6, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#0B0F17';
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    ctx.restore();
  }, [tracks, entities, viewportSize, settings.enablePose, settings.cameraMode]);

  // Draw cumulative blurred spatial Heatmap preview
  useEffect(() => {
    const canvas = heatmapCanvasRef.current;
    if (!canvas) return;
    const W = 180;
    const H = 56;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
    ctx.lineWidth = 1;
    for (let gx = 1; gx < 4; gx++) {
      ctx.beginPath();
      ctx.moveTo((W / 4) * gx, 0);
      ctx.lineTo((W / 4) * gx, H);
      ctx.stroke();
    }
    for (let gy = 1; gy < 3; gy++) {
      ctx.beginPath();
      ctx.moveTo(0, (H / 3) * gy);
      ctx.lineTo(W, (H / 3) * gy);
      ctx.stroke();
    }

    const COLS = 32;
    const ROWS = 18;
    const grid = new Float32Array(COLS * ROWS);
    let maxCell = 1;
    for (const pt of heatmapPoints) {
      const gx = Math.min(COLS - 1, Math.max(0, Math.floor(pt.xNorm * COLS)));
      const gy = Math.min(ROWS - 1, Math.max(0, Math.floor(pt.yNorm * ROWS)));
      const val = (grid[gy * COLS + gx] += pt.weight);
      if (val > maxCell) maxCell = val;
    }

    ctx.save();
    ctx.filter = 'blur(4.5px)';
    for (let gy = 0; gy < ROWS; gy++) {
      for (let gx = 0; gx < COLS; gx++) {
        const count = grid[gy * COLS + gx];
        if (count <= 0) continue;
        const norm = Math.min(1, count / Math.max(2, maxCell * 0.85));
        const px = ((gx + 0.5) / COLS) * W;
        const py = ((gy + 0.5) / ROWS) * H;
        const radius = 4 + norm * 7.5;

        let color = 'rgba(56, 189, 248, 0.45)';
        if (norm > 0.72) color = 'rgba(239, 68, 68, 0.85)';
        else if (norm > 0.45) color = 'rgba(250, 204, 21, 0.72)';
        else if (norm > 0.22) color = 'rgba(34, 197, 94, 0.62)';

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
    ctx.restore();
  }, [heatmapPoints]);

  const maxCount = Math.max(1, stats.persons, stats.vehicles, stats.bicycles, 6);

  const renderDelta = (delta: number) => {
    if (delta > 0) return <span className="text-emerald-400 font-mono text-[10.5px] tabular-nums">▲ +{delta}</span>;
    if (delta < 0) return <span className="text-rose-400 font-mono text-[10.5px] tabular-nums">▼ {delta}</span>;
    return <span className="text-slate-500 font-mono text-[10.5px] tabular-nums">— 0</span>;
  };

  // Smoothed SVG polyline for Processing Speed waveform
  const waveW = 160;
  const waveH = layout.waveHeight;
  const wavePath = (() => {
    if (latencyHistory.length < 2) return `M 0 ${waveH / 2} L ${waveW} ${waveH / 2}`;
    const pts = latencyHistory.map((ms, idx) => {
      const x = (idx / Math.max(1, latencyHistory.length - 1)) * waveW;
      const clampedMs = Math.min(60, Math.max(8, ms));
      const y = waveH - ((clampedMs - 8) / 52) * (waveH - 8) - 4;
      return { x, y };
    });
    let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
    for (let i = 1; i < pts.length; i++) {
      const midX = ((pts[i - 1].x + pts[i].x) * 0.5).toFixed(1);
      d += ` C ${midX} ${pts[i - 1].y.toFixed(1)}, ${midX} ${pts[i].y.toFixed(1)}, ${pts[i].x.toFixed(1)} ${pts[i].y.toFixed(1)}`;
    }
    return d;
  })();

  // Multi-line SVG paths for 20-second Object Tracking Chart
  const chartW = 360;
  const chartH = layout.trackingSvgHeight;
  const maxY = Math.max(
    6,
    ...timeSeries.flatMap((p) => [p.persons, p.vehicles, p.bicycles])
  );

  const buildSeriesPath = (key: 'persons' | 'vehicles' | 'bicycles') => {
    if (timeSeries.length < 2) return '';
    return timeSeries
      .map((pt, i) => {
        const x = (i / 19) * chartW;
        const y = chartH - (pt[key] / maxY) * (chartH - 10) - 5;
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  };

  const densityColor =
    metrics.density === 'High'
      ? 'text-rose-400'
      : metrics.density === 'Moderate'
      ? 'text-amber-400'
      : 'text-emerald-400';

  const backdropSrc =
    settings.cameraMode === 'bike-boulevard' ? BOULEVARD_IMAGE : CROSSWALK_IMAGE;

  return (
    <div
      className="h-dvh w-full bg-[#0B0F17] flex flex-col safe-area-screen overflow-hidden select-none"
      style={{ gap: `${layout.gap}px`, paddingLeft: `${layout.paddingHorizontal}px`, paddingRight: `${layout.paddingHorizontal}px` }}
    >
      {/* 1. Top Status Bar (42px / 36px en compactas) */}
      <header
        className="shrink-0 flex items-center justify-between"
        style={{ height: `${layout.topBarHeight}px` }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`w-2.5 h-2.5 rounded-full shrink-0 ${
              settings.isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
            }`}
          />
          <span className="text-[13px] font-semibold text-slate-100 truncate">
            {settings.isRunning
              ? settings.cameraMode === 'webcam'
                ? 'Camera: Live Feed'
                : 'Camera: Online'
              : 'Camera: Paused'}
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={onToggleCameraMode}
            title="Cambiar lente / fuente de cámara"
            className="h-[34px] px-2.5 rounded-lg bg-[#121826] hover:bg-[#192235] border border-[#1E293B] text-[11px] font-medium text-slate-200 flex items-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap"
          >
            {settings.cameraMode === 'webcam' ? (
              <Video className="w-3.5 h-3.5 text-sky-400" />
            ) : (
              <VideoOff className="w-3.5 h-3.5 text-slate-400" />
            )}
            <span>
              {settings.cameraMode === 'rear-crosswalk'
                ? 'Rear Wide'
                : settings.cameraMode === 'bike-boulevard'
                ? 'Front / Bike'
                : 'Webcam'}
            </span>
          </button>

          <span className="font-mono text-[13px] font-bold text-sky-400 tabular-nums whitespace-nowrap">
            FPS: {metrics.fps.toFixed(1)}
          </span>
        </div>
      </header>

      {/* 2. Visor Principal de Cámara (flex: 1 absorbe el espacio sobrante) */}
      <div
        ref={viewfinderRef}
        className="relative flex-1 min-h-0 w-full rounded-[16px] overflow-hidden bg-[#121826] border border-[#1E293B]"
      >
        {settings.cameraMode === 'webcam' ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover"
            />
            {webcamError && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0B0F17]/90 p-4 text-center gap-1.5">
                <p className="text-xs text-rose-300 font-medium">{webcamError}</p>
                <p className="text-[11px] text-slate-400">
                  Pulsa el selector superior para volver a Rear Wide o Front / Bike.
                </p>
              </div>
            )}
          </>
        ) : (
          <img
            src={backdropSrc}
            alt="Visor de cámara urbana en tiempo real"
            referrerPolicy="no-referrer"
            className="absolute inset-0 w-full h-full object-cover brightness-[0.88] contrast-[1.06]"
          />
        )}

        {/* Contraste inferior sutil dentro del visor */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-black/75 via-black/30 to-transparent" />

        {/* Capa Canvas 60 FPS (Bounding Boxes + Esqueletos Pose) */}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Franja inferior sutil dentro del visor */}
        <div className="absolute bottom-2 left-3 right-3 flex items-center justify-between text-[10.5px] font-mono text-slate-200/90 pointer-events-none">
          <span className="tabular-nums truncate">
            YOLOv8n + Pose · Conf {settings.confidenceThreshold.toFixed(2)}
          </span>
          <span className="tabular-nums shrink-0 text-slate-300">
            {settings.delegate === 'coreml-metal'
              ? 'CoreML / Metal'
              : settings.delegate === 'nnapi-gpu'
              ? 'NNAPI / GPU'
              : 'CPU XNNPACK'}
          </span>
        </div>
      </div>

      {/* 3. Fila de Contadores Rápidos (76px / 64px en compactas) */}
      <div
        className="shrink-0 grid grid-cols-3"
        style={{ height: `${layout.counterRowHeight}px`, gap: `${layout.gap}px` }}
      >
        {[
          {
            label: 'Personas',
            count: stats.persons,
            delta: stats.deltaPersons,
            barColor: 'bg-sky-400',
          },
          {
            label: 'Vehículos',
            count: stats.vehicles,
            delta: stats.deltaVehicles,
            barColor: 'bg-orange-500',
          },
          {
            label: 'Bicicletas',
            count: stats.bicycles,
            delta: stats.deltaBicycles,
            barColor: 'bg-emerald-500',
          },
        ].map((item) => (
          <div
            key={item.label}
            className="bg-[#121826] border border-[#1E293B] rounded-xl px-2.5 flex flex-col justify-between"
            style={{ paddingTop: layout.isCompactScreen ? 6 : 8, paddingBottom: layout.isCompactScreen ? 6 : 8 }}
          >
            <div className="flex items-center justify-between gap-1">
              <span className="text-[11px] font-medium text-slate-300 truncate">{item.label}</span>
              {renderDelta(item.delta)}
            </div>
            <div
              className="font-mono font-bold text-white tabular-nums leading-none"
              style={{ fontSize: `${layout.counterFontSize}px` }}
            >
              {item.count}
            </div>
            <div className="w-full h-1.5 bg-slate-800/90 rounded-full overflow-hidden">
              <div
                className={`h-full ${item.barColor} rounded-full transition-all duration-200`}
                style={{ width: `${Math.min(100, (item.count / maxCount) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* 4. Fila de Métricas y Escena (96px / 82px en compactas) */}
      <div
        className="shrink-0 grid grid-cols-2"
        style={{ height: `${layout.metricsRowHeight}px`, gap: `${layout.gap}px` }}
      >
        {/* Izquierda: Processing Speed */}
        <div
          className="bg-[#121826] border border-[#1E293B] rounded-xl px-3 flex flex-col justify-between"
          style={{ paddingTop: layout.isCompactScreen ? 6 : 8, paddingBottom: layout.isCompactScreen ? 6 : 8 }}
        >
          <div className="flex items-baseline justify-between gap-1">
            <span className="text-[11px] font-medium text-slate-400 truncate">Processing Speed</span>
            <span className="font-mono text-[10.5px] text-slate-400 tabular-nums shrink-0">
              {metrics.timings.totalMs.toFixed(1)} ms
            </span>
          </div>
          <div
            className={`font-mono font-bold text-slate-100 tabular-nums leading-tight ${
              layout.isCompactScreen ? 'text-[15px]' : 'text-[17px]'
            }`}
          >
            {metrics.fps.toFixed(1)} <span className="text-[11px] font-semibold text-sky-400">FPS</span>
          </div>
          <svg
            viewBox={`0 0 ${waveW} ${waveH}`}
            preserveAspectRatio="none"
            className="w-full overflow-visible"
            style={{ height: `${layout.waveHeight}px` }}
            aria-label="Onda suavizada de velocidad de procesamiento"
          >
            <path
              d={wavePath}
              fill="none"
              stroke="#38BDF8"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* Derecha: Analysis + Heatmap difuminado */}
        <div
          className="bg-[#121826] border border-[#1E293B] rounded-xl px-3 flex flex-col justify-between"
          style={{ paddingTop: layout.isCompactScreen ? 6 : 8, paddingBottom: layout.isCompactScreen ? 6 : 8 }}
        >
          <div className="flex items-center justify-between gap-1">
            <span className="text-[11px] font-medium text-slate-400">Analysis</span>
            <span className="text-[11px] font-semibold text-slate-300">{metrics.sceneLabel}</span>
          </div>
          <div className="flex items-baseline justify-between gap-1">
            <span className={`text-[11.5px] font-semibold truncate ${densityColor}`}>
              Density: {metrics.density}
            </span>
            <span className="font-mono text-[10px] text-slate-400 tabular-nums shrink-0">
              n={tracks.length}
            </span>
          </div>
          <div
            className="relative w-full rounded-lg overflow-hidden bg-[#090D16] border border-[#1E293B]"
            style={{ height: `${layout.heatmapHeight}px` }}
          >
            <canvas ref={heatmapCanvasRef} className="w-full h-full block" />
          </div>
        </div>
      </div>

      {/* 5. Gráfico Temporal (Object Tracking 20s: 105px / 88px en compactas) */}
      <div
        className="shrink-0 bg-[#121826] border border-[#1E293B] rounded-xl px-3 flex flex-col justify-between"
        style={{
          height: `${layout.trackingChartHeight}px`,
          paddingTop: layout.isCompactScreen ? 6 : 8,
          paddingBottom: layout.isCompactScreen ? 6 : 8,
        }}
      >
        <div className="flex items-center justify-between">
          <span className="text-[11.5px] font-semibold text-slate-200">Object Tracking (20s)</span>
          <div className="flex items-center gap-2.5 text-[10px] font-mono">
            <span className="text-sky-400">● Per</span>
            <span className="text-orange-400">● Veh</span>
            <span className="text-emerald-400">● Bic</span>
          </div>
        </div>

        <svg
          viewBox={`0 0 ${chartW} ${chartH}`}
          preserveAspectRatio="none"
          className="w-full overflow-visible"
          style={{ height: `${layout.trackingSvgHeight}px` }}
          aria-label="Gráfico de seguimiento de objetos en ventana de 20 segundos"
        >
          {[0.25, 0.5, 0.75].map((ratio) => (
            <line
              key={ratio}
              x1={0}
              y1={chartH * ratio}
              x2={chartW}
              y2={chartH * ratio}
              stroke="rgba(148, 163, 184, 0.12)"
              strokeDasharray="3 3"
            />
          ))}
          <path d={buildSeriesPath('persons')} fill="none" stroke="#38BDF8" strokeWidth="2" />
          <path d={buildSeriesPath('vehicles')} fill="none" stroke="#F97316" strokeWidth="2" />
          <path d={buildSeriesPath('bicycles')} fill="none" stroke="#10B981" strokeWidth="2" />
        </svg>

        <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 tabular-nums leading-none">
          <span>-20s</span>
          <span>-10s</span>
          <span>Now</span>
        </div>
      </div>

      {/* 6. Footer Bar (48px / 42px en compactas) */}
      <footer
        className="shrink-0 flex items-center justify-between border-t border-[#1E293B]"
        style={{ height: `${layout.footerHeight}px` }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`w-2.5 h-2.5 rounded-full shrink-0 ${
              !settings.isRunning
                ? 'bg-amber-500'
                : metrics.thermalMode === 'Thermal Safe'
                ? 'bg-rose-500'
                : metrics.thermalMode === 'Balanced'
                ? 'bg-amber-400'
                : 'bg-emerald-500'
            }`}
          />
          <span className="text-[12px] font-medium text-slate-300 truncate">
            System Status: {settings.isRunning ? `Running [${metrics.thermalMode}]` : 'Paused'}
          </span>
        </div>

        <button
          type="button"
          onClick={onOpenSettings}
          className={`${
            layout.isCompactScreen ? 'h-[32px] px-3' : 'h-[36px] px-3.5'
          } rounded-xl bg-[#121826] hover:bg-[#192235] border border-[#1E293B] text-xs font-semibold text-slate-100 flex items-center gap-2 transition-colors cursor-pointer whitespace-nowrap shrink-0`}
        >
          <SlidersHorizontal className="w-3.5 h-3.5 text-sky-400" />
          <span>Ajustes</span>
        </button>
      </footer>
    </div>
  );
};
