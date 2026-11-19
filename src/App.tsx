import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  CategoryStats,
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

const INITIAL_SETTINGS: PipelineSettings = {
  confidenceThreshold: 0.45,
  nmsIouThreshold: 0.45,
  cameraMode: 'rear-crosswalk',
  delegate: 'coreml-metal',
  thermalControlMode: 'auto',
  enablePose: true,
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
    deltaPersons: 0,
    deltaVehicles: 0,
    deltaBicycles: 0,
  });
  const [metrics, setMetrics] = useState<PipelineMetrics>(INITIAL_METRICS);
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

  const viewportSizeRef = useRef({ width: 390, height: 300 });
  const tracksRef = useRef<TrackedObject[]>([]);
  const trackerRef = useRef(new ByteTrackLite(18, 0.55, 0.25));
  const profilerRef = useRef(new PipelineProfiler(0.18));
  const thermalRef = useRef(new AdaptiveThermalController());
  const prevPoseByTrackRef = useRef<Map<number, TrackedObject['keypoints']>>(new Map());
  const webcamExtractorRef = useRef(new WebcamMotionExtractor());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const prevBucketRef = useRef({ persons: 0, vehicles: 0, bicycles: 0 });
  const simTimeRef = useRef(0);
  const lastFrameStampRef = useRef(performance.now());
  const lastUiSyncRef = useRef(0);
  const secondCounterRef = useRef(20);
  const frameIndexRef = useRef(0);

  const handleViewportResize = useCallback((width: number, height: number) => {
    viewportSizeRef.current = { width, height };
  }, []);

  const updateSettings = useCallback((patch: Partial<PipelineSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      if (patch.cameraMode && patch.cameraMode !== prev.cameraMode) {
        trackerRef.current.reset();
      }
      return next;
    });
  }, []);

  const handleResetHeatmap = useCallback(() => {
    setHeatmapPoints([]);
  }, []);

  // Manage live Webcam stream lifecycle when cameraMode === 'webcam'
  useEffect(() => {
    let cancelled = false;

    const stopStream = () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };

    if (settings.cameraMode !== 'webcam') {
      stopStream();
      setWebcamError(null);
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setWebcamError('El navegador actual no expone acceso a cámara (getUserMedia).');
      return;
    }

    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setWebcamError(null);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWebcamError('Permiso de cámara denegado o sin webcam disponible en el dispositivo.');
        }
      });

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [settings.cameraMode]);

  // Real-time CV Inference, NMS, Tensor-to-Screen, and ByteTrack loop
  useEffect(() => {
    let rafId = 0;

    const step = (now: number) => {
      const dt = Math.min(0.1, (now - lastFrameStampRef.current) / 1000);
      lastFrameStampRef.current = now;
      const cfg = settingsRef.current;

      if (cfg.isRunning) {
        simTimeRef.current += dt;

        const srcW = 1280;
        const srcH = 960;
        const { width: canvasW, height: canvasH } = viewportSizeRef.current;

        let proposals = [];
        let simEntities: RenderedSimEntity[] = [];

        if (cfg.cameraMode === 'webcam' && videoRef.current && videoRef.current.readyState >= 2) {
          proposals = webcamExtractorRef.current.extractProposals(
            videoRef.current,
            simTimeRef.current
          );
        } else {
          const generated = generateSimulatedProposals(
            cfg.cameraMode,
            simTimeRef.current,
            cfg.crowdBoost,
            srcW,
            srcH
          );
          proposals = generated.proposals;
          simEntities = generated.entities;
        }

        const delegateScale =
          cfg.delegate === 'coreml-metal'
            ? 1.0
            : cfg.delegate === 'nnapi-gpu'
            ? 1.42
            : 2.55;
        const crowdOverhead = cfg.crowdBoost ? 4.5 : 0;
        const yoloMs =
          (10.8 + Math.sin(simTimeRef.current * 2.1) * 1.0) * delegateScale + crowdOverhead * 0.6;
        const estimatedPoseMs =
          (7.6 + Math.cos(simTimeRef.current * 1.7) * 0.8) * delegateScale + crowdOverhead * 0.4;

        frameIndexRef.current += 1;
        const captureMs = 1.6;
        const preprocessMs = 1.9;
        const renderMs = 1.8;
        const preliminaryMs =
          captureMs + preprocessMs + yoloMs + (cfg.enablePose ? estimatedPoseMs : 0) + 1.4 + renderMs;
        const autoDecision = thermalRef.current.evaluate(preliminaryMs);
        const thermalDecision =
          cfg.thermalControlMode === 'auto'
            ? autoDecision
            : {
                mode: cfg.thermalControlMode,
                rollingAvgMs: autoDecision.rollingAvgMs,
                shouldRunPose:
                  cfg.thermalControlMode === 'Ultra'
                    ? true
                    : cfg.thermalControlMode === 'Balanced'
                    ? frameIndexRef.current % 2 === 0
                    : false,
                shouldUpdateHeatmap:
                  cfg.thermalControlMode === 'Thermal Safe'
                    ? frameIndexRef.current % 4 === 0
                    : true,
                poseStride:
                  cfg.thermalControlMode === 'Ultra' ? 1 : cfg.thermalControlMode === 'Balanced' ? 2 : 0,
                heatmapStride: cfg.thermalControlMode === 'Thermal Safe' ? 4 : 1,
              };
        const runPoseThisFrame = cfg.enablePose && thermalDecision.shouldRunPose;

        const tPostStart = performance.now();
        const { rawFilteredCount, detections, letterbox } = decodeProposalsToScreen(
          proposals,
          cfg.confidenceThreshold,
          cfg.nmsIouThreshold,
          cfg.cameraMode === 'webcam' && videoRef.current?.videoWidth
            ? videoRef.current.videoWidth
            : srcW,
          cfg.cameraMode === 'webcam' && videoRef.current?.videoHeight
            ? videoRef.current.videoHeight
            : srcH,
          canvasW,
          canvasH,
          runPoseThisFrame || thermalDecision.mode === 'Balanced',
          cfg.cameraMode === 'webcam'
        );

        const activeTracks = trackerRef.current.update(detections);
        const tPostEnd = performance.now();

        for (const tr of activeTracks) {
          if (tr.className !== 'person') continue;
          if (thermalDecision.mode === 'Thermal Safe' || !cfg.enablePose) {
            tr.keypoints = [];
          } else if (runPoseThisFrame && tr.keypoints.length === 17) {
            prevPoseByTrackRef.current.set(tr.trackId, tr.keypoints);
          } else if (thermalDecision.mode === 'Balanced') {
            const cached = prevPoseByTrackRef.current.get(tr.trackId);
            if (cached) tr.keypoints = cached;
          }
        }

        tracksRef.current = activeTracks;

        const measuredPostMs = Math.max(0.4, tPostEnd - tPostStart);
        const actualPoseMs = runPoseThisFrame ? estimatedPoseMs : 0;
        const emaTimings = profilerRef.current.recordFrame(
          captureMs,
          preprocessMs,
          yoloMs,
          actualPoseMs,
          measuredPostMs,
          renderMs,
          now
        );

        setTracks(activeTracks);
        setEntities(simEntities);

        if (now - lastUiSyncRef.current >= 160) {
          lastUiSyncRef.current = now;
          const buckets = countByBucket(activeTracks);

          // Solo envía la tabla de objetos rastreados hacia el modal si está abierto (visible === true)
          if (diagnosticsOpenRef.current) {
            setModalTracks(activeTracks);
          }

          setStats((prev) => ({
            ...prev,
            persons: buckets.persons,
            vehicles: buckets.vehicles,
            bicycles: buckets.bicycles,
          }));

          setMetrics({
            fps: Math.min(60, emaTimings.theoreticalFps),
            theoreticalFps: emaTimings.theoreticalFps,
            realFps: emaTimings.realFps,
            thermalMode: thermalDecision.mode,
            rollingAvgMs: thermalDecision.rollingAvgMs,
            rawProposalsCount: rawFilteredCount,
            postNmsCount: detections.length,
            suppressedCount: Math.max(0, rawFilteredCount - detections.length),
            density: classifySceneDensity(activeTracks.length),
            sceneLabel: 'Urban',
            timings: {
              captureMs: emaTimings.captureMs,
              preprocessMs: emaTimings.resizeMs,
              yoloMs: emaTimings.yoloMs,
              poseMs: emaTimings.poseMs,
              nmsTrackMs: emaTimings.nmsTrackMs,
              renderMs: emaTimings.skiaRenderMs,
              totalMs: emaTimings.totalMs,
            },
            letterbox: {
              scale: letterbox.scale,
              padX: letterbox.padX,
              padY: letterbox.padY,
            },
          });

          setLatencyHistory((prev) => [...prev.slice(-31), emaTimings.totalMs]);

          if (activeTracks.length > 0 && thermalDecision.shouldUpdateHeatmap) {
            setHeatmapPoints((prev) => {
              const additions = activeTracks.map((tr) => ({
                xNorm: Math.max(0.04, Math.min(0.96, tr.centerX / Math.max(1, canvasW))),
                yNorm: Math.max(0.06, Math.min(0.94, tr.centerY / Math.max(1, canvasH))),
                weight: 1,
              }));
              return [...prev, ...additions].slice(-140);
            });
          }
        }
      }

      rafId = requestAnimationFrame(step);
    };

    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
  }, []);

  // 1-second interval sampler for 20-second sliding window Object Tracking chart & interval deltas
  useEffect(() => {
    const interval = setInterval(() => {
      if (!settingsRef.current.isRunning) return;
      const buckets = countByBucket(tracksRef.current);
      const prev = prevBucketRef.current;

      setStats({
        persons: buckets.persons,
        vehicles: buckets.vehicles,
        bicycles: buckets.bicycles,
        deltaPersons: buckets.persons - prev.persons,
        deltaVehicles: buckets.vehicles - prev.vehicles,
        deltaBicycles: buckets.bicycles - prev.bicycles,
      });

      prevBucketRef.current = buckets;
      secondCounterRef.current += 1;

      setTimeSeries((prevSeries) => [
        ...prevSeries.slice(-19),
        {
          timestampLabel: `${secondCounterRef.current}s`,
          secondIndex: secondCounterRef.current,
          persons: buckets.persons,
          vehicles: buckets.vehicles,
          bicycles: buckets.bicycles,
        },
      ]);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <main className="h-dvh w-full bg-[#0B0F17] text-slate-100 overflow-hidden">
      <MobileVisionViewport
        settings={settings}
        tracks={tracks}
        entities={entities}
        stats={stats}
        metrics={metrics}
        latencyHistory={latencyHistory}
        timeSeries={timeSeries}
        heatmapPoints={heatmapPoints}
        videoRef={videoRef}
        webcamError={webcamError}
        onViewportResize={handleViewportResize}
        onOpenSettings={() => {
          setModalTracks(tracksRef.current);
          setDiagnosticsOpen(true);
        }}
        onToggleCameraMode={() => {
          const order: Array<PipelineSettings['cameraMode']> = [
            'rear-crosswalk',
            'bike-boulevard',
            'webcam',
          ];
          const nextIdx = (order.indexOf(settings.cameraMode) + 1) % order.length;
          updateSettings({ cameraMode: order[nextIdx] });
        }}
      />

      <DiagnosticsModal
        visible={diagnosticsOpen}
        settings={settings}
        metrics={metrics}
        tracks={modalTracks}
        onUpdate={updateSettings}
        onResetHeatmap={handleResetHeatmap}
        onClose={() => setDiagnosticsOpen(false)}
      />
    </main>
  );
}
