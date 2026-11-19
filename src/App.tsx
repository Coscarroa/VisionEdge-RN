import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ARVisualTheme,
  CategoryStats,
  GyroStabilizationTelemetry,
  PipelineMetrics,
  PipelineSettings,
  TimeSeriesPoint,
  TrackedObject,
} from './engine/cvTypes';
import {
  ByteTrackLite,
  classifySceneDensity,
  countByBucket,
  decodeProposalsToScreen,
} from './engine/postprocess';
import {
  RenderedSimEntity,
  WebcamMotionExtractor,
  generateSimulatedProposals,
} from './engine/sceneSimulation';
import { MobileVisionViewport } from './components/MobileVisionViewport';
import { DiagnosticsModal } from './components/DiagnosticsModal';
import { AdaptiveThermalController, PipelineProfiler } from './utils/pipelineThermal';
import { AR_THEME_ORDER } from './config/ARThemeConfig';

const INITIAL_SETTINGS: PipelineSettings = {
  confidenceThreshold: 0.45,
  nmsIouThreshold: 0.45,
  cameraMode: 'rear-crosswalk',
  delegate: 'coreml-metal',
  thermalControlMode: 'auto',
  arTheme: 'tactical',
  enablePose: true,
  enableGyro: true,
  crowdBoost: false,
  isRunning: true,
};

const INITIAL_METRICS: PipelineMetrics = {
  fps: 38.8,
  theoreticalFps: 38.8,
  realFps: 38.8,
  thermalMode: 'Ultra',
  rollingAvgMs: 25.8,
  rawProposalsCount: 14,
  postNmsCount: 7,
  suppressedCount: 7,
  density: 'Moderate',
  sceneLabel: 'Urban',
  timings: {
    captureMs: 1.6,
    preprocessMs: 1.9,
    yoloMs: 11.4,
    poseMs: 7.7,
    nmsTrackMs: 1.4,
    renderMs: 1.8,
    totalMs: 25.8,
  },
  letterbox: { scale: 0.5, padX: 0, padY: 80 },
};

const INITIAL_GYRO: GyroStabilizationTelemetry = {
  omegaX: 0.0,
  omegaY: 0.0,
  omegaZ: 0.0,
  offsetX: 0.0,
  offsetY: 0.0,
  activeHz: 60,
  lastResetAgoMs: 0,
};

export default function App() {
  const [settings, setSettings] = useState<PipelineSettings>(INITIAL_SETTINGS);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);

  const [tracks, setTracks] = useState<TrackedObject[]>([]);
  const [modalTracks, setModalTracks] = useState<TrackedObject[]>([]);
  const [entities, setEntities] = useState<RenderedSimEntity[]>([]);
  const [stats, setStats] = useState<CategoryStats>({
    persons: 0,
    vehicles: 0,
    bicycles: 0,
    animalsOther: 0,
    deltaPersons: 0,
    deltaVehicles: 0,
    deltaBicycles: 0,
    deltaAnimalsOther: 0,
  });
  const [metrics, setMetrics] = useState<PipelineMetrics>(INITIAL_METRICS);
  const [gyroTelemetry, setGyroTelemetry] = useState<GyroStabilizationTelemetry>(INITIAL_GYRO);
  const [latencyHistory, setLatencyHistory] = useState<number[]>(() =>
    Array.from({ length: 32 }, () => 24.5 + Math.random() * 3.5)
  );
  const [timeSeries, setTimeSeries] = useState<TimeSeriesPoint[]>(() =>
    Array.from({ length: 20 }, (_, idx) => ({
      timestampLabel: `-${20 - idx}s`,
      secondIndex: idx,
      persons: 3,
      vehicles: 3,
      bicycles: 1,
    }))
  );
  const [heatmapPoints, setHeatmapPoints] = useState<
    Array<{ xNorm: number; yNorm: number; weight: number }>
  >([]);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const diagnosticsOpenRef = useRef(diagnosticsOpen);
  diagnosticsOpenRef.current = diagnosticsOpen;

  const viewportSizeRef = useRef({ width: 412, height: 840 });
  const tracksRef = useRef<TrackedObject[]>([]);
  // Diccionario LERP por trackId para suavizar transición al reiniciar offset de giroscopio
  const lerpedTracksRef = useRef<Map<number, TrackedObject>>(new Map());
  const trackerRef = useRef(new ByteTrackLite(18, 0.55, 0.25));
  const profilerRef = useRef(new PipelineProfiler(0.18));
  const thermalRef = useRef(new AdaptiveThermalController());
  const prevCountsRef = useRef({ persons: 3, vehicles: 3, bicycles: 1, animalsOther: 1 });
  const frameIndexRef = useRef(0);
  const lastInferenceTimeRef = useRef(performance.now());
  const lastMetricPushRef = useRef(performance.now());
  const lastSeriesSecondRef = useRef(performance.now());

  // Estado reactivo del giroscopio a 60Hz (inter-frame offset entre inferencias YOLO)
  const gyroStateRef = useRef({
    omegaX: 0, // rad/s Pitch
    omegaY: 0, // rad/s Yaw
    omegaZ: 0, // rad/s Roll
    offsetX: 0, // Píxeles acumulados entre inferencias
    offsetY: 0,
    manualImpulseX: 0,
    manualImpulseY: 0,
    lastSampleMs: performance.now(),
    lastResetMs: performance.now(),
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamExtractorRef = useRef<WebcamMotionExtractor | null>(null);

  // Gestión de Webcam en vivo
  useEffect(() => {
    let activeStream: MediaStream | null = null;
    if (settings.cameraMode === 'webcam') {
      setWebcamError(null);
      if (!webcamExtractorRef.current) {
        webcamExtractorRef.current = new WebcamMotionExtractor();
      }
      navigator.mediaDevices
        ?.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: 'user' },
          audio: false,
        })
        .then((stream) => {
          activeStream = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play().catch(() => {});
          }
        })
        .catch(() => {
          setWebcamError('Permiso denegado — usando simulación sintética en tiempo real.');
        });
    } else if (videoRef.current && videoRef.current.srcObject) {
      const oldStream = videoRef.current.srcObject as MediaStream;
      oldStream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [settings.cameraMode]);

  // 1. Bucle de Giroscopio a 60Hz (16ms) — Estabilización Inercial Inter-Frame + LERP
  useEffect(() => {
    const interval = window.setInterval(() => {
      const now = performance.now();
      const g = gyroStateRef.current;
      const dtSec = Math.min(0.05, Math.max(0.005, (now - g.lastSampleMs) / 1000));
      g.lastSampleMs = now;

      if (!settingsRef.current.isRunning || !settingsRef.current.enableGyro) {
        g.offsetX *= 0.7;
        g.offsetY *= 0.7;
        g.omegaX *= 0.7;
        g.omegaY *= 0.7;
        return;
      }

      // Micro-oscilación fisiológica de mano + impulsos de paneo del usuario en el visor
      const tSec = now * 0.001;
      const handTremorYaw = Math.sin(tSec * 2.4) * 0.09 + Math.cos(tSec * 5.1) * 0.04;
      const handTremorPitch = Math.cos(tSec * 1.9) * 0.07 + Math.sin(tSec * 4.3) * 0.03;

      g.omegaY = handTremorYaw + g.manualImpulseX;
      g.omegaX = handTremorPitch + g.manualImpulseY;
      g.omegaZ = Math.sin(tSec * 1.3) * 0.02;

      // Decaimiento suave del impulso manual
      g.manualImpulseX *= 0.86;
      g.manualImpulseY *= 0.86;

      // Ganancia focal (píxeles por radián para lente 1x ~62° FOV)
      const focalPx = viewportSizeRef.current.width * 0.82;
      g.offsetX = Math.max(-72, Math.min(72, (g.offsetX - g.omegaY * dtSec * focalPx) * 0.94));
      g.offsetY = Math.max(-72, Math.min(72, (g.offsetY + g.omegaX * dtSec * focalPx) * 0.94));

      // Interpolación Lineal (LERP) a 60 FPS hacia las coordenadas reales de YOLO
      const rawTargets = tracksRef.current;
      if (rawTargets.length > 0) {
        const lerpAlpha = 0.36;
        const lerpedList: TrackedObject[] = rawTargets.map((target) => {
          const prev = lerpedTracksRef.current.get(target.trackId);
          if (!prev) {
            lerpedTracksRef.current.set(target.trackId, target);
            return target;
          }

          const smoothBox = {
            x: prev.screenBox.x + (target.screenBox.x - prev.screenBox.x) * lerpAlpha,
            y: prev.screenBox.y + (target.screenBox.y - prev.screenBox.y) * lerpAlpha,
            width: prev.screenBox.width + (target.screenBox.width - prev.screenBox.width) * lerpAlpha,
            height: prev.screenBox.height + (target.screenBox.height - prev.screenBox.height) * lerpAlpha,
          };

          const smoothKeypoints = target.keypoints.map((kp, idx) => {
            const pKp = prev.keypoints[idx];
            if (!pKp) return kp;
            return {
              ...kp,
              x: pKp.x + (kp.x - pKp.x) * lerpAlpha,
              y: pKp.y + (kp.y - pKp.y) * lerpAlpha,
            };
          });

          const updated: TrackedObject = {
            ...target,
            screenBox: smoothBox,
            centerX: smoothBox.x + smoothBox.width * 0.5,
            centerY: smoothBox.y + smoothBox.height * 0.5,
            keypoints: smoothKeypoints,
          };
          lerpedTracksRef.current.set(target.trackId, updated);
          return updated;
        });

        setTracks(lerpedList);
      }

      setGyroTelemetry({
        omegaX: g.omegaX,
        omegaY: g.omegaY,
        omegaZ: g.omegaZ,
        offsetX: g.offsetX,
        offsetY: g.offsetY,
        activeHz: 60,
        lastResetAgoMs: Math.round(now - g.lastResetMs),
      });
    }, 16);

    return () => window.clearInterval(interval);
  }, []);

  // 2. Bucle de Inferencia YOLOv8n + MoveNet + ByteTrack + Gobernador Térmico
  useEffect(() => {
    let rafId: number;

    const tick = (now: number) => {
      const currentSettings = settingsRef.current;

      if (currentSettings.isRunning) {
        const delegateFactor =
          currentSettings.delegate === 'coreml-metal'
            ? 1.0
            : currentSettings.delegate === 'nnapi-gpu'
            ? 1.25
            : 2.35;

        const baseCaptureMs = 1.5;
        const basePreprocessMs = 1.9;
        const baseYoloMs = (10.8 + (currentSettings.crowdBoost ? 2.4 : 0)) * delegateFactor;

        const estWithoutPose = baseCaptureMs + basePreprocessMs + baseYoloMs + 3.0;
        const autoThermal = thermalRef.current.evaluate(
          estWithoutPose + (currentSettings.enablePose ? 7.4 * delegateFactor : 0)
        );

        const activeThermalMode =
          currentSettings.thermalControlMode === 'auto'
            ? autoThermal.mode
            : currentSettings.thermalControlMode;

        const targetIntervalMs = Math.max(24, estWithoutPose);
        if (now - lastInferenceTimeRef.current >= targetIntervalMs) {
          lastInferenceTimeRef.current = now;
          frameIndexRef.current += 1;
          const frameIdx = frameIndexRef.current;

          const shouldRunPose =
            currentSettings.enablePose &&
            (activeThermalMode === 'Ultra' ||
              (activeThermalMode === 'Balanced' && frameIdx % 2 === 0));

          const shouldUpdateHeatmap =
            activeThermalMode !== 'Thermal Safe' || frameIdx % 4 === 0;

          const srcWidth = 640;
          const srcHeight = 360;
          const vpWidth = Math.max(280, viewportSizeRef.current.width);
          const vpHeight = Math.max(220, viewportSizeRef.current.height);

          let rawProposals;
          let renderedSim: RenderedSimEntity[] = [];

          if (
            currentSettings.cameraMode === 'webcam' &&
            videoRef.current &&
            videoRef.current.readyState >= 2 &&
            webcamExtractorRef.current
          ) {
            rawProposals = webcamExtractorRef.current.extractProposals(
              videoRef.current,
              now * 0.001
            );
          } else {
            const simMode =
              currentSettings.cameraMode === 'webcam'
                ? 'rear-crosswalk'
                : currentSettings.cameraMode;
            const simResult = generateSimulatedProposals(
              simMode,
              now * 0.001,
              currentSettings.crowdBoost,
              vpWidth,
              vpHeight
            );
            rawProposals = simResult.proposals;
            renderedSim = simResult.entities;
          }

          const { rawFilteredCount, detections, letterbox } = decodeProposalsToScreen(
            rawProposals,
            currentSettings.confidenceThreshold,
            currentSettings.nmsIouThreshold,
            srcWidth,
            srcHeight,
            vpWidth,
            vpHeight,
            shouldRunPose,
            currentSettings.cameraMode === 'webcam'
          );

          const activeTracks = trackerRef.current.update(detections);
          tracksRef.current = activeTracks;
          setEntities(renderedSim);

          // REINICIO SINCRONIZADO DEL GIROSCOPIO:
          // Al llegar una nueva inferencia de YOLO, las cajas reales ya reflejan la nueva pose
          // de la cámara; amortiguamos el offset inter-frame a 0 mientras el LERP absorbe el cambio.
          gyroStateRef.current.offsetX *= 0.18;
          gyroStateRef.current.offsetY *= 0.18;
          gyroStateRef.current.lastResetMs = now;

          if (shouldUpdateHeatmap && activeTracks.length > 0) {
            setHeatmapPoints((prev) => {
              const additions = activeTracks.map((tr) => ({
                xNorm: Math.min(1, Math.max(0, tr.centerX / vpWidth)),
                yNorm: Math.min(1, Math.max(0, tr.centerY / vpHeight)),
                weight: tr.className === 'person' ? 1.0 : 0.75,
              }));
              return [...prev, ...additions].slice(-200);
            });
          }

          // Throttler de 250 ms hacia estado React del HUD
          if (now - lastMetricPushRef.current >= 250) {
            lastMetricPushRef.current = now;

            const jitter = () => (Math.random() - 0.5) * 0.35;
            const captureMs = Math.max(0.8, baseCaptureMs + jitter() * 0.3);
            const preprocessMs = Math.max(1.1, basePreprocessMs + jitter() * 0.3);
            const yoloMs = Math.max(4.5, baseYoloMs + jitter());
            const poseMs = shouldRunPose
              ? Math.max(2.5, (7.1 + activeTracks.length * 0.3) * delegateFactor + jitter())
              : 0;
            const nmsTrackMs = Math.max(0.5, 1.2 + rawFilteredCount * 0.04 + jitter() * 0.2);
            const renderMs = Math.max(0.9, 1.6 + activeTracks.length * 0.08 + jitter() * 0.2);

            const ema = profilerRef.current.recordFrame(
              captureMs,
              preprocessMs,
              yoloMs,
              poseMs,
              nmsTrackMs,
              renderMs,
              now
            );

            const buckets = countByBucket(activeTracks);
            const density = classifySceneDensity(activeTracks.length);

            setStats({
              persons: buckets.persons,
              vehicles: buckets.vehicles,
              bicycles: buckets.bicycles,
              animalsOther: buckets.animalsOther,
              deltaPersons: buckets.persons - prevCountsRef.current.persons,
              deltaVehicles: buckets.vehicles - prevCountsRef.current.vehicles,
              deltaBicycles: buckets.bicycles - prevCountsRef.current.bicycles,
              deltaAnimalsOther: buckets.animalsOther - prevCountsRef.current.animalsOther,
            });

            if (diagnosticsOpenRef.current) {
              setModalTracks(activeTracks);
            }

            setMetrics({
              fps: Number(ema.realFps.toFixed(1)),
              theoreticalFps: Number(ema.theoreticalFps.toFixed(1)),
              realFps: Number(ema.realFps.toFixed(1)),
              thermalMode: activeThermalMode,
              rollingAvgMs: Number(autoThermal.rollingAvgMs.toFixed(1)),
              rawProposalsCount: rawFilteredCount,
              postNmsCount: detections.length,
              suppressedCount: Math.max(0, rawFilteredCount - detections.length),
              density,
              sceneLabel:
                currentSettings.cameraMode === 'rear-crosswalk'
                  ? 'Urban'
                  : currentSettings.cameraMode === 'bike-boulevard'
                  ? 'Transit'
                  : 'Live Cam',
              timings: {
                captureMs: Number(ema.captureMs.toFixed(1)),
                preprocessMs: Number(ema.resizeMs.toFixed(1)),
                yoloMs: Number(ema.yoloMs.toFixed(1)),
                poseMs: Number(ema.poseMs.toFixed(1)),
                nmsTrackMs: Number(ema.nmsTrackMs.toFixed(1)),
                renderMs: Number(ema.skiaRenderMs.toFixed(1)),
                totalMs: Number(ema.totalMs.toFixed(1)),
              },
              letterbox: {
                scale: Number(letterbox.scale.toFixed(3)),
                padX: Math.round(letterbox.padX),
                padY: Math.round(letterbox.padY),
              },
            });

            setLatencyHistory((prev) => [
              ...prev.slice(1),
              Number(ema.totalMs.toFixed(1)),
            ]);
          }

          if (now - lastSeriesSecondRef.current >= 1000) {
            lastSeriesSecondRef.current = now;
            const buckets = countByBucket(activeTracks);
            prevCountsRef.current = buckets;

            setTimeSeries((prev) => {
              const nextIdx = (prev[prev.length - 1]?.secondIndex ?? 0) + 1;
              const updated = [
                ...prev.slice(1),
                {
                  timestampLabel: 'now',
                  secondIndex: nextIdx,
                  persons: buckets.persons,
                  vehicles: buckets.vehicles,
                  bicycles: buckets.bicycles,
                },
              ];
              return updated.map((pt, idx) => ({
                ...pt,
                timestampLabel: idx === updated.length - 1 ? 'now' : `-${updated.length - 1 - idx}s`,
              }));
            });
          }
        }
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, []);

  const handleUpdateSettings = useCallback((patch: Partial<PipelineSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      if (patch.cameraMode && patch.cameraMode !== prev.cameraMode) {
        trackerRef.current.reset();
        lerpedTracksRef.current.clear();
        setHeatmapPoints([]);
      }
      return next;
    });
  }, []);

  const handleCycleARTheme = useCallback(() => {
    setSettings((prev) => {
      const idx = AR_THEME_ORDER.indexOf(prev.arTheme);
      const nextTheme: ARVisualTheme = AR_THEME_ORDER[(idx + 1) % AR_THEME_ORDER.length];
      return { ...prev, arTheme: nextTheme };
    });
  }, []);

  // Permite simular un giro rápido de cámara (Yaw/Pitch) al arrastrar o mover sobre el visor
  const handleSimulateGyroImpulse = useCallback((deltaYawRadS: number, deltaPitchRadS: number) => {
    gyroStateRef.current.manualImpulseX = Math.max(
      -2.2,
      Math.min(2.2, gyroStateRef.current.manualImpulseX + deltaYawRadS)
    );
    gyroStateRef.current.manualImpulseY = Math.max(
      -2.2,
      Math.min(2.2, gyroStateRef.current.manualImpulseY + deltaPitchRadS)
    );
  }, []);

  const handleViewportResize = useCallback((width: number, height: number) => {
    viewportSizeRef.current = { width, height };
  }, []);

  const handleResetHeatmap = useCallback(() => {
    setHeatmapPoints([]);
  }, []);

  const handleOpenDiagnostics = useCallback(() => {
    setModalTracks(tracksRef.current);
    setDiagnosticsOpen(true);
  }, []);

  const handleCloseDiagnostics = useCallback(() => {
    setDiagnosticsOpen(false);
  }, []);

  return (
    <div className="h-dvh w-full overflow-hidden bg-[#0B0F17] text-[#F8FAFC] select-none relative">
      {/* Visor AR 100% Edge-to-Edge a Pantalla Completa (StyleSheet.absoluteFill) + HUD Flotante Glassmorphism */}
      <MobileVisionViewport
        settings={settings}
        metrics={metrics}
        stats={stats}
        tracks={tracks}
        entities={entities}
        heatmapPoints={heatmapPoints}
        gyroTelemetry={gyroTelemetry}
        videoRef={videoRef}
        webcamError={webcamError}
        onViewportResize={handleViewportResize}
        onUpdateSettings={handleUpdateSettings}
        onCycleARTheme={handleCycleARTheme}
        onSimulateGyroImpulse={handleSimulateGyroImpulse}
        onOpenDiagnostics={handleOpenDiagnostics}
        onResetHeatmap={handleResetHeatmap}
      />

      {/* Bottom Sheet de Ajustes, Telemetría Avanzada (Gráfica 20s, Latencia EMA, Tabla ByteTrack y Código RN) */}
      <DiagnosticsModal
        visible={diagnosticsOpen}
        settings={settings}
        metrics={metrics}
        tracks={modalTracks}
        timeSeries={timeSeries}
        latencyHistory={latencyHistory}
        gyroTelemetry={gyroTelemetry}
        onUpdate={handleUpdateSettings}
        onResetHeatmap={handleResetHeatmap}
        onClose={handleCloseDiagnostics}
      />
    </div>
  );
}
