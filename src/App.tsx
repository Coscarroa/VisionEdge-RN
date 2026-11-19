import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  Compass,
  Eye,
  EyeOff,
  Flame,
  Radar,
  RefreshCw,
  SlidersHorizontal,
} from 'lucide-react';
import {
  ARStartOverlay,
  CameraErrorDetails,
  PermissionStepStatus,
  requestMobileInertialSensorsPermission,
  unlockAudioAndHapticsOnGesture,
} from './components/ARStartOverlay';
import {
  WebCameraFacingMode,
  WebCameraStream,
  WebCameraStreamHandle,
} from './components/WebCameraStream';
import { ARWebCanvasOverlay } from './components/ARWebCanvasOverlay';
import {
  OnnxTrackedDetection,
  useOnnxVisionEngine,
} from './hooks/useOnnxVisionEngine';
import { useInertialARTracker } from './hooks/useInertialARTracker';
import { DiagnosticsModal } from './components/DiagnosticsModal';
import {
  AR_THEMES,
  AR_THEME_ORDER,
  ARThemeId,
} from './config/ARThemeConfig';
import {
  GyroStabilizationTelemetry,
  PipelineMetrics,
  PipelineSettings,
  TargetClass,
  TimeSeriesPoint,
  TrackedObject,
} from './engine/cvTypes';
import { AdaptiveThermalController } from './utils/pipelineThermal';

const INITIAL_SETTINGS: PipelineSettings = {
  confidenceThreshold: 0.40,
  nmsIouThreshold: 0.45,
  cameraMode: 'webcam',
  delegate: 'coreml-metal',
  thermalControlMode: 'auto',
  arTheme: 'tactical',
  enablePose: true,
  enableGyro: true,
  crowdBoost: false,
  isRunning: true,
};

function mapCocoNameToTargetClass(name: string): TargetClass {
  const n = name.toLowerCase();
  if (n === 'person') return 'person';
  if (n === 'car') return 'car';
  if (n === 'bus' || n === 'train') return 'bus';
  if (n === 'truck' || n === 'boat' || n === 'airplane') return 'truck';
  if (n === 'bicycle') return 'bicycle';
  if (n === 'motorcycle') return 'motorcycle';
  if (
    n === 'dog' ||
    n === 'cat' ||
    n === 'bird' ||
    n === 'horse' ||
    n === 'sheep' ||
    n === 'cow' ||
    n === 'elephant' ||
    n === 'bear' ||
    n === 'zebra' ||
    n === 'giraffe'
  ) {
    return 'dog';
  }
  return 'backpack';
}

export default function App() {
  const [settings, setSettings] = useState<PipelineSettings>(INITIAL_SETTINGS);
  const [arStarted, setArStarted] = useState(false);
  const [facingMode, setFacingMode] = useState<WebCameraFacingMode>('environment');
  const [isSwitchingCamera, setIsSwitchingCamera] = useState(false);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [cameraError, setCameraError] = useState<CameraErrorDetails | null>(null);
  const [permissionStatus, setPermissionStatus] = useState<PermissionStepStatus>({
    imu: 'idle',
    camera: 'idle',
    engine: 'loading',
  });

  const [viewportSize, setViewportSize] = useState<{ width: number; height: number }>(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 412,
    height: typeof window !== 'undefined' ? window.innerHeight : 840,
  }));

  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [radarExpanded, setRadarExpanded] = useState(true);
  const [showFullHeatmapOverlay, setShowFullHeatmapOverlay] = useState(false);
  const [heatmapPoints, setHeatmapPoints] = useState<
    Array<{ xNorm: number; yNorm: number; weight: number }>
  >([]);
  const [latencyHistory, setLatencyHistory] = useState<number[]>(() =>
    Array.from({ length: 32 }, () => 18.2)
  );
  const [timeSeries, setTimeSeries] = useState<TimeSeriesPoint[]>(() =>
    Array.from({ length: 20 }, (_, idx) => ({
      timestampLabel: `-${20 - idx}s`,
      secondIndex: idx,
      persons: 0,
      vehicles: 0,
      bicycles: 0,
    }))
  );
  const [gyroTelemetry, setGyroTelemetry] = useState<GyroStabilizationTelemetry>({
    omegaX: 0,
    omegaY: 0,
    omegaZ: 0,
    offsetX: 0,
    offsetY: 0,
    activeHz: 60,
    lastResetAgoMs: 0,
  });

  const cameraRef = useRef<WebCameraStreamHandle | null>(null);
  const radarCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const thermalControllerRef = useRef(new AdaptiveThermalController());
  const pointerDragRef = useRef<{ active: boolean; lastX: number; lastY: number }>({
    active: false,
    lastX: 0,
    lastY: 0,
  });

  // 1. Anclaje Espacial Inercial con Filtro de Kalman + LERP (useInertialARTracker.ts)
  const {
    inertialStateRef,
    onNewOnnxInferenceBatch,
    injectPointerDelta,
    computeAnchoredDetectionState,
    decayInertialOffsetOnRenderFrame,
  } = useInertialARTracker({
    enabled: settings.enableGyro,
    screenWidth: viewportSize.width,
    screenHeight: viewportSize.height,
  });

  // Medir cambios de tamaño y orientación del navegador móvil (100vw x 100dvh)
  useEffect(() => {
    const updateViewport = () => {
      setViewportSize({
        width: Math.max(280, window.innerWidth),
        height: Math.max(320, window.innerHeight),
      });
    };
    updateViewport();
    window.addEventListener('resize', updateViewport);
    window.addEventListener('orientationchange', updateViewport);
    return () => {
      window.removeEventListener('resize', updateViewport);
      window.removeEventListener('orientationchange', updateViewport);
    };
  }, []);

  // Callback ejecutado al llegar cada inferencia del motor ONNX
  const handleNewInferenceFrame = useCallback(
    (newDetections: OnnxTrackedDetection[]) => {
      onNewOnnxInferenceBatch();
      if (newDetections.length > 0 && viewportSize.width > 0 && viewportSize.height > 0) {
        setHeatmapPoints((prev) => {
          const additions = newDetections.map((d) => ({
            xNorm: Math.min(1, Math.max(0, d.centerX / viewportSize.width)),
            yNorm: Math.min(1, Math.max(0, d.centerY / viewportSize.height)),
            weight: d.category === 'person' ? 1.0 : 0.78,
          }));
          return [...prev, ...additions].slice(-180);
        });
      }
    },
    [onNewOnnxInferenceBatch, viewportSize.width, viewportSize.height]
  );

  // 2. Motor de Inferencia ONNX Runtime Web (Planar NCHW [1, 3, 640, 640], WebGPU/WASM SIMD, 80 COCO + Pose)
  const {
    isEngineReady,
    engineStatusText,
    activeBackend,
    activeModelName,
    detections,
    counts,
    timings,
    letterboxMeta,
    loadCustomOnnxFile,
  } = useOnnxVisionEngine({
    videoElement,
    isCameraActive: arStarted && Boolean(videoElement),
    isFrontCamera: facingMode === 'user',
    screenWidth: viewportSize.width,
    screenHeight: viewportSize.height,
    confidenceThreshold: settings.confidenceThreshold,
    nmsIouThreshold: settings.nmsIouThreshold,
    enablePose: settings.enablePose,
    isPaused: !settings.isRunning,
    onNewInferenceFrame: handleNewInferenceFrame,
  });

  useEffect(() => {
    setPermissionStatus((prev) => ({
      ...prev,
      engine: isEngineReady ? 'ready' : 'loading',
    }));
  }, [isEngineReady]);

  // Actualizar telemetría de giroscopio Kalman y series temporales de 20s
  useEffect(() => {
    if (!arStarted) return;
    const timer = window.setInterval(() => {
      const g = inertialStateRef.current;
      setGyroTelemetry({
        omegaX: g.omegaX,
        omegaY: g.omegaY,
        omegaZ: g.omegaZ,
        offsetX: g.offsetX,
        offsetY: g.offsetY,
        activeHz: 60,
        lastResetAgoMs: Math.round(timings.totalMs),
      });
      setLatencyHistory((prev) => [...prev.slice(1), timings.totalMs]);
    }, 250);
    return () => window.clearInterval(timer);
  }, [arStarted, inertialStateRef, timings.totalMs]);

  useEffect(() => {
    if (!arStarted) return;
    const secTimer = window.setInterval(() => {
      setTimeSeries((prev) => {
        const nextIdx = (prev[prev.length - 1]?.secondIndex ?? 0) + 1;
        const updated = [
          ...prev.slice(1),
          {
            timestampLabel: 'now',
            secondIndex: nextIdx,
            persons: counts.persons,
            vehicles: counts.vehicles,
            bicycles: counts.animals + counts.objects,
          },
        ];
        return updated.map((pt, idx) => ({
          ...pt,
          timestampLabel: idx === updated.length - 1 ? 'now' : `-${updated.length - 1 - idx}s`,
        }));
      });
    }, 1000);
    return () => window.clearInterval(secTimer);
  }, [arStarted, counts]);

  // Dibujar Mini-Radar espacial en esquina inferior derecha
  useEffect(() => {
    const rCanvas = radarCanvasRef.current;
    if (!rCanvas || !radarExpanded || !arStarted) return;
    const ctx = rCanvas.getContext('2d');
    if (!ctx) return;

    const w = rCanvas.width;
    const h = rCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const activeTheme = AR_THEMES[settings.arTheme] ?? AR_THEMES.tactical;

    ctx.strokeStyle =
      settings.arTheme === 'industrial'
        ? 'rgba(250, 204, 21, 0.24)'
        : 'rgba(56, 189, 248, 0.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.88, w * 0.36, Math.PI, 0);
    ctx.arc(w * 0.5, h * 0.88, w * 0.68, Math.PI, 0);
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.5, h);
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    for (const pt of heatmapPoints.slice(-80)) {
      const px = pt.xNorm * w;
      const py = pt.yNorm * h;
      const rad = 10;
      const grad = ctx.createRadialGradient(px, py, 1, px, py, rad);
      grad.addColorStop(0, 'rgba(249, 115, 22, 0.34)');
      grad.addColorStop(0.6, 'rgba(56, 189, 248, 0.14)');
      grad.addColorStop(1, 'rgba(56, 189, 248, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fill();
    }

    for (const det of detections) {
      const col = activeTheme.categories[det.category]?.stroke ?? '#38BDF8';
      const bx = Math.max(4, Math.min(w - 4, (det.centerX / Math.max(1, viewportSize.width)) * w));
      const by = Math.max(4, Math.min(h - 4, (det.centerY / Math.max(1, viewportSize.height)) * h));
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(bx, by, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [heatmapPoints, detections, radarExpanded, arStarted, settings.arTheme, viewportSize]);

  /**
   * Secuencia obligatoria activada por el gesto onClick de "Iniciar Cámara AR":
   * (1) DeviceMotionEvent.requestPermission() en iOS 13+ Safari
   * (2) navigator.mediaDevices.getUserMedia + await videoElement.play()
   * (3) Desbloqueo de audio/vibración y transición al HUD AR en vivo
   */
  const handleStartARCamera = useCallback(async () => {
    setCameraError(null);

    setPermissionStatus((prev) => ({ ...prev, imu: 'requesting' }));
    const imuResult = await requestMobileInertialSensorsPermission();
    setPermissionStatus((prev) => ({ ...prev, imu: imuResult }));

    setPermissionStatus((prev) => ({ ...prev, camera: 'requesting' }));
    const videoEl = await cameraRef.current?.startStream(facingMode);

    if (!videoEl) {
      setPermissionStatus((prev) => ({ ...prev, camera: 'denied' }));
      return;
    }

    unlockAudioAndHapticsOnGesture();
    setPermissionStatus((prev) => ({ ...prev, camera: 'granted' }));
    setVideoElement(videoEl);
    setArStarted(true);
  }, [facingMode]);

  const handleSwitchCamera = useCallback(async () => {
    if (!cameraRef.current || isSwitchingCamera) return;
    setIsSwitchingCamera(true);
    try {
      const nextFacing = await cameraRef.current.switchCamera();
      setFacingMode(nextFacing);
      setHeatmapPoints([]);
    } finally {
      setIsSwitchingCamera(false);
    }
  }, [isSwitchingCamera]);

  const handleStreamReady = useCallback(
    (video: HTMLVideoElement, actualFacing: WebCameraFacingMode) => {
      setVideoElement(video);
      setFacingMode(actualFacing);
      setCameraError(null);
    },
    []
  );

  const handleStreamError = useCallback((err: CameraErrorDetails) => {
    setCameraError(err);
    setPermissionStatus((prev) => ({ ...prev, camera: 'denied' }));
  }, []);

  const handleCycleARTheme = useCallback(() => {
    setSettings((prev) => {
      const idx = AR_THEME_ORDER.indexOf(prev.arTheme);
      const nextTheme: ARThemeId = AR_THEME_ORDER[(idx + 1) % AR_THEME_ORDER.length];
      return { ...prev, arTheme: nextTheme };
    });
  }, []);

  const handleUpdateSettings = useCallback((patch: Partial<PipelineSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const handleResetHeatmap = useCallback(() => {
    setHeatmapPoints([]);
  }, []);

  // Soporte táctil/arrastre para probar el Filtro de Kalman inercial en escritorio
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('button')) return;
    pointerDragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!pointerDragRef.current.active || !settings.enableGyro) return;
    const dx = e.clientX - pointerDragRef.current.lastX;
    const dy = e.clientY - pointerDragRef.current.lastY;
    pointerDragRef.current.lastX = e.clientX;
    pointerDragRef.current.lastY = e.clientY;
    injectPointerDelta(dx, dy);
  };

  const handlePointerUp = () => {
    pointerDragRef.current.active = false;
  };

  const autoThermal = thermalControllerRef.current.evaluate(timings.totalMs);
  const activeThermalMode =
    settings.thermalControlMode === 'auto' ? autoThermal.mode : settings.thermalControlMode;

  const densityLabel =
    counts.total < 3 ? 'Low' : counts.total <= 7 ? 'Moderate' : 'High';

  const modalMetrics: PipelineMetrics = {
    fps: timings.fps,
    theoreticalFps: Number(Math.min(60, 1000 / Math.max(4, timings.totalMs)).toFixed(1)),
    realFps: timings.fps,
    thermalMode: activeThermalMode,
    rollingAvgMs: Number(autoThermal.rollingAvgMs.toFixed(1)),
    rawProposalsCount: counts.total,
    postNmsCount: counts.total,
    suppressedCount: 0,
    density: densityLabel,
    sceneLabel: facingMode === 'environment' ? 'Rear ONNX AR' : 'Front ONNX AR',
    timings: {
      captureMs: timings.captureMs,
      preprocessMs: timings.preprocessMs,
      yoloMs: timings.inferenceMs,
      poseMs:
        settings.enablePose && counts.persons > 0
          ? Number((timings.postprocessMs * 0.6).toFixed(1))
          : 0,
      nmsTrackMs: timings.postprocessMs,
      renderMs: 1.4,
      totalMs: timings.totalMs,
    },
    letterbox: {
      scale: Number(letterboxMeta.scale.toFixed(3)),
      padX: letterboxMeta.padX,
      padY: letterboxMeta.padY,
    },
  };

  const modalTracks: TrackedObject[] = detections.map((d) => ({
    trackId: d.trackId,
    classId: d.classId,
    className: mapCocoNameToTargetClass(d.className),
    confidence: d.confidence,
    screenBox: d.screenBox,
    keypoints: d.keypoints,
    age: d.age,
    lostFrames: d.lostFrames,
    vx: d.vx,
    vy: d.vy,
    centerX: d.centerX,
    centerY: d.centerY,
  }));

  const activeThemeSpec = AR_THEMES[settings.arTheme] ?? AR_THEMES.tactical;
  const glassStyle: React.CSSProperties = {
    backgroundColor: 'rgba(11, 15, 23, 0.58)',
    borderColor: activeThemeSpec.hudBorderColor,
  };

  return (
    <div
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className="relative w-screen h-dvh overflow-hidden bg-[#06090F] text-[#F8FAFC] select-none"
    >
      {/* 1. Stream de Cámara Física Real (100vw x 100dvh, object-fit: cover, playsInline) */}
      <WebCameraStream
        ref={cameraRef}
        facingMode={facingMode}
        onStreamReady={handleStreamReady}
        onStreamError={handleStreamError}
      />

      {/* 2. HUD AR Avanzado en Canvas HiDPI (Retículas Neón por Categoría + Estelas 15pt + Callout Pins + Kalman) */}
      <ARWebCanvasOverlay
        detections={detections}
        arTheme={settings.arTheme}
        enablePose={settings.enablePose}
        enableGyro={settings.enableGyro}
        showHeatmapOverlay={showFullHeatmapOverlay}
        heatmapPoints={heatmapPoints}
        screenWidth={viewportSize.width}
        screenHeight={viewportSize.height}
        inertialStateRef={inertialStateRef}
        computeAnchoredDetectionState={computeAnchoredDetectionState}
        decayInertialOffsetOnRenderFrame={decayInertialOffsetOnRenderFrame}
      />

      {/* 3. Pantalla Inicial Inmersiva con Botón Flotante "Iniciar Cámara AR" */}
      <ARStartOverlay
        visible={!arStarted || Boolean(cameraError)}
        engineStatusText={engineStatusText}
        activeBackend={activeBackend}
        activeModelName={activeModelName}
        permissionStatus={permissionStatus}
        errorDetails={cameraError}
        onStartARCamera={handleStartARCamera}
        onUploadCustomOnnxFile={(file) => loadCustomOnnxFile(file, 'detector')}
      />

      {/* 4. HUD Flotante Glassmorphism sobre 100dvh */}
      {arStarted && !cameraError && (
        <>
          {/* Barra Superior Flotante */}
          <div className="fixed top-3 left-3 right-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
            <div
              style={glassStyle}
              className="pointer-events-auto min-h-[42px] px-3.5 rounded-2xl backdrop-blur-md border flex items-center gap-2 shadow-lg"
            >
              <span className="w-2 h-2 rounded-full bg-[#22C55E]" />
              <span className="text-xs font-bold tracking-wide text-slate-100 whitespace-nowrap">
                ONNX AR
              </span>
              <span className="text-slate-500" aria-hidden="true">
                ·
              </span>
              <span className="font-mono text-xs font-bold text-sky-400 tabular-nums whitespace-nowrap">
                {timings.fps.toFixed(1)} FPS
              </span>
              <span className="text-slate-500 hidden sm:inline" aria-hidden="true">
                ·
              </span>
              <span className="hidden sm:inline font-mono text-[11px] text-emerald-300 uppercase">
                {activeBackend}
              </span>
            </div>

            <div className="pointer-events-auto flex items-center gap-1.5 flex-wrap">
              <button
                type="button"
                onClick={handleCycleARTheme}
                style={glassStyle}
                className="min-h-[42px] px-3.5 rounded-2xl backdrop-blur-md border text-xs font-semibold text-slate-100 hover:bg-slate-900/75 flex items-center gap-1.5 transition-colors cursor-pointer shadow-lg whitespace-nowrap"
              >
                <span>Estilo: {activeThemeSpec.shortLabel}</span>
              </button>

              <button
                type="button"
                onClick={() => handleUpdateSettings({ enablePose: !settings.enablePose })}
                style={glassStyle}
                className={`min-h-[42px] px-3.5 rounded-2xl backdrop-blur-md border text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer shadow-lg whitespace-nowrap ${
                  settings.enablePose ? 'text-emerald-300' : 'text-slate-400'
                }`}
              >
                {settings.enablePose ? (
                  <Eye className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <EyeOff className="w-3.5 h-3.5 text-slate-400" />
                )}
                <span>Pose {settings.enablePose ? 'ON' : 'OFF'}</span>
              </button>

              <button
                type="button"
                onClick={handleSwitchCamera}
                disabled={isSwitchingCamera}
                style={glassStyle}
                title="Alternar entre cámara trasera y frontal"
                className="min-h-[42px] px-3.5 rounded-2xl backdrop-blur-md border text-xs font-semibold text-sky-300 hover:bg-slate-900/75 disabled:opacity-50 flex items-center gap-1.5 transition-colors cursor-pointer shadow-lg whitespace-nowrap"
              >
                {isSwitchingCamera ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-sky-400" />
                ) : (
                  <Camera className="w-3.5 h-3.5 text-sky-400" />
                )}
                <span>{facingMode === 'environment' ? 'Trasera (1x)' : 'Frontal'}</span>
              </button>
            </div>
          </div>

          {/* Contadores en Vivo por Categoría con Colores Neón Exactos (Cian, Naranja, Esmeralda, Violeta) */}
          <div
            style={glassStyle}
            className="fixed top-16 left-3 z-20 px-3.5 py-2 rounded-2xl backdrop-blur-md border flex flex-wrap items-center gap-3 text-xs shadow-md pointer-events-auto"
          >
            <div className="flex items-center gap-1.5">
              <span className="text-slate-300">Personas</span>
              <span className="font-mono font-bold text-[#38BDF8] tabular-nums">
                {counts.persons}
              </span>
            </div>
            <span className="text-slate-600" aria-hidden="true">
              ·
            </span>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-300">Vehículos</span>
              <span className="font-mono font-bold text-[#F97316] tabular-nums">
                {counts.vehicles}
              </span>
            </div>
            <span className="text-slate-600" aria-hidden="true">
              ·
            </span>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-300">Animales</span>
              <span className="font-mono font-bold text-[#10B981] tabular-nums">
                {counts.animals}
              </span>
            </div>
            <span className="text-slate-600" aria-hidden="true">
              ·
            </span>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-300">Objetos</span>
              <span className="font-mono font-bold text-[#A855F7] tabular-nums">
                {counts.objects}
              </span>
            </div>
          </div>

          {/* Mini-Radar / Heatmap Flotante Colapsable en Esquina Inferior Derecha */}
          <div className="fixed bottom-16 right-3 z-20 flex flex-col items-end gap-1.5 pointer-events-auto">
            {radarExpanded ? (
              <div
                style={glassStyle}
                className="w-[156px] rounded-2xl backdrop-blur-md border p-2.5 shadow-2xl flex flex-col gap-1.5"
              >
                <div className="flex items-center justify-between gap-1">
                  <div className="flex items-center gap-1.5">
                    <Radar className="w-3.5 h-3.5 text-sky-400" />
                    <span className="text-[11px] font-semibold text-slate-200">
                      Radar Espacial
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
                      <Flame className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setRadarExpanded(false)}
                      className="px-1.5 py-0.5 rounded text-xs font-mono text-slate-400 hover:text-white cursor-pointer"
                    >
                      —
                    </button>
                  </div>
                </div>

                <div className="relative w-full h-[82px] rounded-xl bg-[#070A10]/75 border border-white/10 overflow-hidden">
                  <canvas
                    ref={radarCanvasRef}
                    width={136}
                    height={82}
                    className="w-full h-full block"
                  />
                  <div className="absolute bottom-1 left-1.5 right-1.5 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span>n={counts.total}</span>
                    <button
                      type="button"
                      onClick={handleResetHeatmap}
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
                style={glassStyle}
                className="min-h-[42px] px-3.5 rounded-2xl backdrop-blur-md border text-xs font-medium text-slate-200 hover:text-white flex items-center gap-1.5 shadow-lg cursor-pointer"
              >
                <Radar className="w-3.5 h-3.5 text-sky-400" />
                <span>Radar ({counts.total})</span>
              </button>
            )}
          </div>

          {/* Barra Inferior Flotante */}
          <div className="fixed bottom-3 left-3 right-3 z-20 flex items-center justify-between gap-2 pointer-events-none">
            <button
              type="button"
              onClick={() => handleUpdateSettings({ enableGyro: !settings.enableGyro })}
              style={glassStyle}
              className="pointer-events-auto min-h-[44px] px-3.5 rounded-2xl backdrop-blur-md border flex items-center gap-2.5 shadow-xl cursor-pointer transition-colors hover:bg-slate-900/75"
            >
              <Compass
                className={`w-4 h-4 ${
                  settings.enableGyro ? 'text-emerald-400' : 'text-slate-500'
                }`}
              />
              <div className="flex flex-col items-start leading-tight">
                <span className="text-xs font-semibold text-slate-100 whitespace-nowrap">
                  {settings.enableGyro ? 'Kalman IMU + Estelas Activo' : 'Kalman IMU Pausado'}
                </span>
                <span className="text-[10.5px] font-mono text-slate-400 tabular-nums">
                  Δx:{gyroTelemetry.offsetX >= 0 ? '+' : ''}
                  {gyroTelemetry.offsetX.toFixed(1)}px Δy:{gyroTelemetry.offsetY >= 0 ? '+' : ''}
                  {gyroTelemetry.offsetY.toFixed(1)}px · {timings.totalMs.toFixed(1)}ms
                </span>
              </div>
            </button>

            <button
              type="button"
              onClick={() => setDiagnosticsOpen(true)}
              style={glassStyle}
              className="pointer-events-auto min-h-[44px] px-4 rounded-2xl backdrop-blur-md border hover:bg-slate-900/80 text-slate-100 text-xs font-semibold flex items-center gap-2 shadow-xl transition-colors cursor-pointer whitespace-nowrap"
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-sky-400" />
              <span>Ajustes y Telemetría</span>
            </button>
          </div>
        </>
      )}

      {/* 5. Bottom Sheet de Ajustes, Telemetría y Explorador de Código ONNX Web / RN */}
      <DiagnosticsModal
        visible={diagnosticsOpen}
        settings={settings}
        metrics={modalMetrics}
        tracks={modalTracks}
        timeSeries={timeSeries}
        latencyHistory={latencyHistory}
        gyroTelemetry={gyroTelemetry}
        onUpdate={handleUpdateSettings}
        onResetHeatmap={handleResetHeatmap}
        onClose={() => setDiagnosticsOpen(false)}
      />
    </div>
  );
}
