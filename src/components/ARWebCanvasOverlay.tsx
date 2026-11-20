/**
 * Componente Principal de Renderizado AR con OffscreenCanvas + Fallback Transparente + Captura Táctil
 * (src/components/ARWebCanvasOverlay.tsx)
 *
 * - Detección segura de soporte:
 *   Verifica 'OffscreenCanvas' in window && typeof canvas.transferControlToOffscreen === 'function'
 *   dentro de un bloque try/catch con protección WeakMap contra React 19 StrictMode.
 * - Modo Worker (Zero-Copy):
 *   Empaqueta todas las detecciones en un Float32Array plano (arBinaryProtocol.ts) y transfiere el
 *   ArrayBuffer mediante Transferable Objects a ARRenderWorker.ts (< 0.05 ms).
 * - Modo Fallback Transparente:
 *   Si OffscreenCanvas no está soportado O si el usuario activa la grabación de video AR (REC)
 *   mediante MediaRecorder, conmuta automáticamente al bucle requestAnimationFrame en el hilo
 *   principal ejecutando exactamente la misma función renderARFrame(ctx, state) sin diferencia visual.
 * - Capa Transparente de Captura Táctil (Touch Loupe & Magnetic Snapping):
 *   Captura gestos onPointerDown / onPointerMove / onPointerUp sobre los vértices del Polígono ROI (V1..V4)
 *   y los nodos de la Línea Virtual de Aforo (A/B), enviando POINTER_DRAG a 60Hz al Worker con Lupa Táctica.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ARThemeId } from '../config/ARThemeConfig';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { InertialARState } from '../hooks/useInertialARTracker';
import {
  globalVirtualCountingLine,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';
import {
  globalSpatialZonesROI,
  RoiSnapGuide,
  RoiVerticesNorm,
} from '../modules/SpatialZonesROI';
import { ProximityHazardLink } from '../modules/HumanSafetyAnalytics';
import {
  ARPointerDragPayload,
  encodeDetectionsToBinaryBuffer,
} from '../rendering/arBinaryProtocol';
import {
  ARCanvasRendererEngine,
  ARCanvasRenderState,
  ARInertialGyroState,
  renderARFrame,
} from '../rendering/arCanvasRenderer';
import {
  ARRenderWorkerInboundMessage,
  ARRenderWorkerOutboundMessage,
} from '../workers/ARRenderWorker';

export interface ARRenderTelemetryStatus {
  mode: 'OffscreenWorker' | 'MainThreadFallback';
  label: string;
  fps: number;
  decodeTimeMs: number;
  renderTimeMs: number;
}

export interface ARWebCanvasOverlayProps {
  detections: OnnxTrackedDetection[];
  arTheme: ARThemeId;
  enablePose: boolean;
  enableGyro: boolean;
  onlyVehiclesMode: boolean;
  showCountingLine: boolean;
  lineEndpointsNorm: VirtualLineEndpointsNorm;
  activeDragHandle: 'a' | 'b' | null;
  /** Polígono ROI de 4 vértices arrastrables (Zona de Exclusión / Estacionamiento Prohibido) */
  enableRoiPolygon?: boolean;
  enableRoiSnapping?: boolean;
  roiVerticesNorm?: RoiVerticesNorm;
  activeRoiVertexIndex?: 0 | 1 | 2 | 3 | null;
  activeRoiViolationInView?: boolean;
  /** Guías magnéticas activas durante el arrastre con Snapping de los vértices V1..V4 */
  roiSnapGuides?: RoiSnapGuide[];
  /** Líneas de advertencia de riesgo de colisión peatón-vehículo */
  proximityLinks?: ProximityHazardLink[];
  showHeatmapOverlay: boolean;
  heatmapPoints: Array<{ xNorm: number; yNorm: number; weight: number }>;
  screenWidth: number;
  screenHeight: number;
  /** Indica si la grabación MediaRecorder AR (REC) está activa para activar el Fallback compatible */
  isRecordingAR?: boolean;
  /** Referencia mutable compartida con useInertialARTracker (Kalman + LERP 60/120Hz) */
  inertialStateRef: React.MutableRefObject<InertialARState>;
  computeAnchoredDetectionState?: (
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
  /** Callback para entregar la referencia del <canvas> al grabador MediaRecorder AR */
  onCanvasReady?: (canvas: HTMLCanvasElement | null) => void;
  /** Callback opcional para sincronizar cambios de vértices ROI desde la capa táctil */
  onRoiVerticesChange?: (
    vertices: RoiVerticesNorm,
    snapGuides: RoiSnapGuide[],
    activeIndex: 0 | 1 | 2 | 3 | null
  ) => void;
  /** Callback opcional para sincronizar cambios de la línea de aforo desde la capa táctil */
  onCountingLineChange?: (
    endpoints: VirtualLineEndpointsNorm,
    activeHandle: 'a' | 'b' | null
  ) => void;
  /** Callback opcional para notificar telemetría del motor de renderizado (OffscreenWorker vs Fallback) */
  onRenderTelemetryChange?: (telemetry: ARRenderTelemetryStatus) => void;
}

/**
 * Registro WeakMap por elemento <canvas> para garantizar que canvas.transferControlToOffscreen()
 * jamás se invoque dos veces sobre el mismo nodo DOM (protección contra React 19 StrictMode).
 */
const OFFSCREEN_WORKER_REGISTRY = new WeakMap<
  HTMLCanvasElement,
  { worker: Worker; initialized: boolean }
>();

export const ARWebCanvasOverlay: React.FC<ARWebCanvasOverlayProps> = ({
  detections,
  arTheme,
  enablePose,
  enableGyro,
  onlyVehiclesMode,
  showCountingLine,
  lineEndpointsNorm,
  activeDragHandle,
  enableRoiPolygon = false,
  enableRoiSnapping = true,
  roiVerticesNorm,
  activeRoiVertexIndex = null,
  activeRoiViolationInView = false,
  roiSnapGuides = [],
  proximityLinks = [],
  showHeatmapOverlay,
  heatmapPoints,
  screenWidth,
  screenHeight,
  isRecordingAR = false,
  decayInertialOffsetOnRenderFrame,
  onCanvasReady,
  onRoiVerticesChange,
  onCountingLineChange,
  onRenderTelemetryChange,
}) => {
  const workerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fallbackCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const [workerSupported, setWorkerSupported] = useState<boolean>(false);
  const [workerTelemetry, setWorkerTelemetry] =
    useState<ARRenderTelemetryStatus>({
      mode: 'OffscreenWorker',
      label: 'Render: OffscreenWorker (60 FPS)',
      fps: 60,
      decodeTimeMs: 0.03,
      renderTimeMs: 0.9,
    });

  // Estado local de la Lupa Táctica de Precisión para el modo Fallback
  const pointerLoupeRef = useRef<ARCanvasRenderState['activePointerLoupe']>(
    null
  );

  // Cuando isRecordingAR === true o el navegador no soporta OffscreenCanvas, el modo Fallback toma el control
  const useFallbackMode = !workerSupported || isRecordingAR;

  const getDevicePixelRatio = () =>
    typeof window !== 'undefined'
      ? Math.min(2.5, window.devicePixelRatio || 1)
      : 1;

  const buildRenderStateSnapshot = useCallback(
    (): ARCanvasRenderState => ({
      detections,
      arTheme,
      enablePose,
      enableGyro,
      onlyVehiclesMode,
      showCountingLine,
      lineEndpointsNorm,
      activeDragHandle,
      countingStats: globalVirtualCountingLine.getStats(),
      enableRoiPolygon,
      roiVerticesNorm,
      activeRoiVertexIndex,
      activeRoiViolationInView,
      roiSnapGuides,
      activePointerLoupe: pointerLoupeRef.current,
      proximityLinks,
      showHeatmapOverlay,
      heatmapPoints,
      screenWidth,
      screenHeight,
      dpr: getDevicePixelRatio(),
    }),
    [
      detections,
      arTheme,
      enablePose,
      enableGyro,
      onlyVehiclesMode,
      showCountingLine,
      lineEndpointsNorm,
      activeDragHandle,
      enableRoiPolygon,
      roiVerticesNorm,
      activeRoiVertexIndex,
      activeRoiViolationInView,
      roiSnapGuides,
      proximityLinks,
      showHeatmapOverlay,
      heatmapPoints,
      screenWidth,
      screenHeight,
    ]
  );

  const latestStateRef = useRef<ARCanvasRenderState>(
    buildRenderStateSnapshot()
  );
  latestStateRef.current = buildRenderStateSnapshot();

  const decayFnRef = useRef(decayInertialOffsetOnRenderFrame);
  decayFnRef.current = decayInertialOffsetOnRenderFrame;

  const telemetryCallbackRef = useRef(onRenderTelemetryChange);
  telemetryCallbackRef.current = onRenderTelemetryChange;

  // Entregar siempre al grabador MediaRecorder AR el <canvas> activo (fallbackCanvas cuando REC está activo)
  useEffect(() => {
    const activeCanvas = useFallbackMode
      ? fallbackCanvasRef.current
      : workerCanvasRef.current ?? fallbackCanvasRef.current;
    onCanvasReady?.(activeCanvas);
    return () => onCanvasReady?.(null);
  }, [onCanvasReady, useFallbackMode]);

  // 1. Inicialización segura de OffscreenCanvas + ARRenderWorker.ts
  useEffect(() => {
    const canvas = workerCanvasRef.current;
    if (!canvas) return;

    const canUseOffscreen =
      typeof window !== 'undefined' &&
      'OffscreenCanvas' in window &&
      typeof Worker !== 'undefined' &&
      typeof canvas.transferControlToOffscreen === 'function';

    if (!canUseOffscreen) {
      setWorkerSupported(false);
      return;
    }

    try {
      let entry = OFFSCREEN_WORKER_REGISTRY.get(canvas);

      if (!entry) {
        const offscreen = canvas.transferControlToOffscreen();
        const worker = new Worker(
          new URL('../workers/ARRenderWorker.ts', import.meta.url),
          { type: 'module' }
        );

        const initialGyro = decayFnRef.current();
        const initMsg: ARRenderWorkerInboundMessage = {
          type: 'INIT_CANVAS',
          canvas: offscreen,
          initialState: latestStateRef.current,
          gyro: {
            offsetX: initialGyro.gx,
            offsetY: initialGyro.gy,
            enabled: latestStateRef.current.enableGyro,
          },
        };
        worker.postMessage(initMsg, [offscreen]);

        entry = { worker, initialized: true };
        OFFSCREEN_WORKER_REGISTRY.set(canvas, entry);
      } else {
        const resumeMsg: ARRenderWorkerInboundMessage = {
          type: 'SET_ACTIVE',
          active: true,
        };
        entry.worker.postMessage(resumeMsg);
      }

      workerRef.current = entry.worker;
      setWorkerSupported(true);

      const handleWorkerMessage = (
        ev: MessageEvent<ARRenderWorkerOutboundMessage>
      ) => {
        if (ev.data?.type === 'WORKER_TELEMETRY') {
          const nextTelemetry: ARRenderTelemetryStatus = {
            mode: 'OffscreenWorker',
            label: `Render: OffscreenWorker (${ev.data.fps} FPS)`,
            fps: ev.data.fps,
            decodeTimeMs: ev.data.decodeTimeMs,
            renderTimeMs: ev.data.renderTimeMs,
          };
          setWorkerTelemetry(nextTelemetry);
          telemetryCallbackRef.current?.(nextTelemetry);
        }
      };

      entry.worker.addEventListener('message', handleWorkerMessage);

      // Sincronizador ligero de offset inercial (IMU / Giroscopio) hacia el Worker
      let gyroRafId: number;
      let lastSentGx = 0;
      let lastSentGy = 0;

      const syncGyroLoop = () => {
        const { gx, gy } = decayFnRef.current();
        const gyroEnabled = latestStateRef.current.enableGyro;

        if (
          Math.abs(gx - lastSentGx) > 0.08 ||
          Math.abs(gy - lastSentGy) > 0.08
        ) {
          lastSentGx = gx;
          lastSentGy = gy;
          const gyroPayload: ARInertialGyroState = {
            offsetX: gx,
            offsetY: gy,
            enabled: gyroEnabled,
          };
          const msg: ARRenderWorkerInboundMessage = {
            type: 'GYRO_UPDATE',
            gyro: gyroPayload,
          };
          entry?.worker.postMessage(msg);
        }

        gyroRafId = window.requestAnimationFrame(syncGyroLoop);
      };

      gyroRafId = window.requestAnimationFrame(syncGyroLoop);

      return () => {
        window.cancelAnimationFrame(gyroRafId);
        entry?.worker.removeEventListener('message', handleWorkerMessage);
        const pauseMsg: ARRenderWorkerInboundMessage = {
          type: 'SET_ACTIVE',
          active: false,
        };
        entry?.worker.postMessage(pauseMsg);
      };
    } catch {
      setWorkerSupported(false);
    }
  }, []);

  // 2. Pausar el Worker cuando el modo Fallback está activo (por ejemplo, durante grabación REC) y reanudarlo después
  useEffect(() => {
    if (!workerSupported || !workerRef.current) return;
    const activeMsg: ARRenderWorkerInboundMessage = {
      type: 'SET_ACTIVE',
      active: !useFallbackMode,
    };
    workerRef.current.postMessage(activeMsg);
  }, [workerSupported, useFallbackMode]);

  // 3. Bucle de Fallback Transparente en Hilo Principal (ejecuta exactamente renderARFrame(ctx, state))
  useEffect(() => {
    if (!useFallbackMode) return;

    const canvas = fallbackCanvasRef.current;
    if (!canvas) return;

    const fallbackEngine = new ARCanvasRendererEngine();
    let fallbackRafId: number;
    let lastTime = performance.now();
    let lastTelemetryTick = performance.now();

    const fallbackLoop = (nowMs: number) => {
      const dtSec = Math.min(0.05, Math.max(0.004, (nowMs - lastTime) * 0.001));
      lastTime = nowMs;

      const ctx = canvas.getContext('2d');
      const state = latestStateRef.current;

      if (ctx && state.screenWidth > 0 && state.screenHeight > 0) {
        const targetW = Math.round(state.screenWidth * state.dpr);
        const targetH = Math.round(state.screenHeight * state.dpr);
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }

        const { gx, gy } = decayFnRef.current();
        fallbackEngine.updateGyroOffset({
          offsetX: gx,
          offsetY: gy,
          enabled: state.enableGyro,
        });

        const t0 = performance.now();
        renderARFrame(ctx, state, nowMs, dtSec, fallbackEngine);
        const renderMs = performance.now() - t0;

        if (nowMs - lastTelemetryTick >= 500) {
          lastTelemetryTick = nowMs;
          const fps = fallbackEngine.getMeasuredRenderFps();
          const fallbackStatus: ARRenderTelemetryStatus = {
            mode: 'MainThreadFallback',
            label: isRecordingAR
              ? `Render: MainThread REC (${fps} FPS)`
              : `Render: MainThread Fallback (${fps} FPS)`,
            fps,
            decodeTimeMs: 0,
            renderTimeMs: Number(renderMs.toFixed(2)),
          };
          setWorkerTelemetry(fallbackStatus);
          telemetryCallbackRef.current?.(fallbackStatus);
        }
      }

      fallbackRafId = window.requestAnimationFrame(fallbackLoop);
    };

    fallbackRafId = window.requestAnimationFrame(fallbackLoop);
    return () => window.cancelAnimationFrame(fallbackRafId);
  }, [useFallbackMode, isRecordingAR]);

  // 4. Envío Zero-Copy de Detecciones Binarias (FRAME_DATA con Transferable ArrayBuffer) al Worker
  useEffect(() => {
    if (useFallbackMode || !workerRef.current) return;

    const { buffer, meta } = encodeDetectionsToBinaryBuffer({
      detections,
      countingStats: globalVirtualCountingLine.getStats(),
      activeRoiViolationInView,
      proximityLinks,
      showHeatmapOverlay,
      heatmapPoints,
    });

    const frameMsg: ARRenderWorkerInboundMessage = {
      type: 'FRAME_DATA',
      buffer,
      meta,
    };
    workerRef.current.postMessage(frameMsg, [buffer]);
  }, [
    useFallbackMode,
    detections,
    activeRoiViolationInView,
    proximityLinks,
    showHeatmapOverlay,
    heatmapPoints,
  ]);

  // 5. Envío de cambios de dimensión (RESIZE) al Worker
  useEffect(() => {
    if (useFallbackMode || !workerRef.current) return;
    const resizeMsg: ARRenderWorkerInboundMessage = {
      type: 'RESIZE',
      screenWidth,
      screenHeight,
      dpr: getDevicePixelRatio(),
    };
    workerRef.current.postMessage(resizeMsg);
  }, [useFallbackMode, screenWidth, screenHeight]);

  // 6. Envío de cambios de Tema y Modo (THEME_CHANGE) al Worker
  useEffect(() => {
    if (useFallbackMode || !workerRef.current) return;
    const themeMsg: ARRenderWorkerInboundMessage = {
      type: 'THEME_CHANGE',
      arTheme,
      enablePose,
      enableGyro,
      onlyVehiclesMode,
    };
    workerRef.current.postMessage(themeMsg);
  }, [useFallbackMode, arTheme, enablePose, enableGyro, onlyVehiclesMode]);

  // 7. Envío de actualización de geometría ROI y Línea de Aforo (ROI_UPDATE) al Worker
  useEffect(() => {
    if (useFallbackMode || !workerRef.current) return;
    const roiMsg: ARRenderWorkerInboundMessage = {
      type: 'ROI_UPDATE',
      enableRoiPolygon,
      roiVerticesNorm,
      activeRoiVertexIndex,
      activeRoiViolationInView,
      roiSnapGuides,
      showCountingLine,
      lineEndpointsNorm,
      activeDragHandle,
    };
    workerRef.current.postMessage(roiMsg);
  }, [
    useFallbackMode,
    enableRoiPolygon,
    roiVerticesNorm,
    activeRoiVertexIndex,
    activeRoiViolationInView,
    roiSnapGuides,
    showCountingLine,
    lineEndpointsNorm,
    activeDragHandle,
  ]);

  // 8. Sincronización de Lupa Táctica de Precisión (POINTER_DRAG a 60Hz) cuando el usuario arrastra nodos V1..V4 o A/B
  useEffect(() => {
    const isDraggingRoi =
      activeRoiVertexIndex !== null &&
      roiVerticesNorm &&
      roiVerticesNorm[activeRoiVertexIndex];
    const isDraggingLine = activeDragHandle !== null;

    if (!isDraggingRoi && !isDraggingLine) {
      pointerLoupeRef.current = null;
      return;
    }

    let pointerX = screenWidth * 0.5;
    let pointerY = screenHeight * 0.5;
    let label = '';
    let accentColor = '#38BDF8';

    if (isDraggingRoi && roiVerticesNorm && activeRoiVertexIndex !== null) {
      const v = roiVerticesNorm[activeRoiVertexIndex];
      pointerX = v.x * screenWidth;
      pointerY = v.y * screenHeight;
      const hasSnap = roiSnapGuides.length > 0;
      label = hasSnap
        ? `SNAP V${activeRoiVertexIndex + 1}`
        : `ROI V${activeRoiVertexIndex + 1}`;
      accentColor = hasSnap ? '#22D3EE' : '#F59E0B';
    } else if (isDraggingLine && activeDragHandle !== null) {
      pointerX =
        (activeDragHandle === 'a'
          ? lineEndpointsNorm.x1
          : lineEndpointsNorm.x2) * screenWidth;
      pointerY =
        (activeDragHandle === 'a'
          ? lineEndpointsNorm.y1
          : lineEndpointsNorm.y2) * screenHeight;
      label = `NODO ${activeDragHandle.toUpperCase()}`;
      accentColor = '#38BDF8';
    }

    pointerLoupeRef.current = {
      active: true,
      x: pointerX,
      y: pointerY,
      label,
      accentColor,
    };

    if (!useFallbackMode && workerRef.current) {
      const dragPayload: ARPointerDragPayload = {
        active: true,
        pointerX,
        pointerY,
        activeDragHandle,
        activeRoiVertexIndex,
        lineEndpointsNorm,
        roiVerticesNorm,
        roiSnapGuides,
      };
      const dragMsg: ARRenderWorkerInboundMessage = {
        type: 'POINTER_DRAG',
        payload: dragPayload,
      };
      workerRef.current.postMessage(dragMsg);
    }
  }, [
    useFallbackMode,
    activeRoiVertexIndex,
    activeDragHandle,
    roiVerticesNorm,
    roiSnapGuides,
    lineEndpointsNorm,
    screenWidth,
    screenHeight,
  ]);

  // 9. Capa Transparente de Captura Táctil Dedicada (nodos ROI V1..V4 y Aforo A/B con setPointerCapture a 60Hz)
  const localDragRef = useRef<{
    active: boolean;
    roiIdx: 0 | 1 | 2 | 3 | null;
    lineHandle: 'a' | 'b' | null;
  }>({
    active: false,
    roiIdx: null,
    lineHandle: null,
  });

  const handleTouchNodePointerDown = (
    e: React.PointerEvent<HTMLButtonElement>,
    kind: 'ROI' | 'LINE',
    indexOrHandle: 0 | 1 | 2 | 3 | 'a' | 'b'
  ) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);

    if (kind === 'ROI') {
      const roiIdx = indexOrHandle as 0 | 1 | 2 | 3;
      localDragRef.current = { active: true, roiIdx, lineHandle: null };
      if (roiVerticesNorm) {
        onRoiVerticesChange?.(roiVerticesNorm, roiSnapGuides, roiIdx);
      }
    } else {
      const handle = indexOrHandle as 'a' | 'b';
      localDragRef.current = { active: true, roiIdx: null, lineHandle: handle };
      onCountingLineChange?.(lineEndpointsNorm, handle);
    }
  };

  const handleTouchNodePointerMove = (
    e: React.PointerEvent<HTMLButtonElement>
  ) => {
    if (!localDragRef.current.active) return;
    if (screenWidth <= 0 || screenHeight <= 0) return;
    e.stopPropagation();

    const rawXNorm = Math.max(0.02, Math.min(0.98, e.clientX / screenWidth));
    const rawYNorm = Math.max(0.05, Math.min(0.95, e.clientY / screenHeight));

    if (localDragRef.current.roiIdx !== null) {
      const { verticesNorm: snappedVertices, snapGuides } =
        globalSpatialZonesROI.setVertexNormWithSnapping({
          index: localDragRef.current.roiIdx,
          rawXNorm,
          rawYNorm,
          screenWidth,
          screenHeight,
          enableSnapping: enableRoiSnapping,
          countingLineNorm: showCountingLine ? lineEndpointsNorm : undefined,
        });

      const snapPt = snappedVertices[localDragRef.current.roiIdx];
      const px = snapPt.x * screenWidth;
      const py = snapPt.y * screenHeight;

      if (!useFallbackMode && workerRef.current) {
        const msg: ARRenderWorkerInboundMessage = {
          type: 'POINTER_DRAG',
          payload: {
            active: true,
            pointerX: px,
            pointerY: py,
            activeDragHandle: null,
            activeRoiVertexIndex: localDragRef.current.roiIdx,
            lineEndpointsNorm,
            roiVerticesNorm: snappedVertices,
            roiSnapGuides: snapGuides,
          },
        };
        workerRef.current.postMessage(msg);
      }

      onRoiVerticesChange?.(
        [...snappedVertices],
        snapGuides,
        localDragRef.current.roiIdx
      );
      return;
    }

    if (localDragRef.current.lineHandle !== null) {
      globalVirtualCountingLine.setEndpointNorm(
        localDragRef.current.lineHandle,
        rawXNorm,
        rawYNorm
      );
      const nextEndpoints = {
        ...globalVirtualCountingLine.getEndpointsNorm(),
      };
      const px = rawXNorm * screenWidth;
      const py = rawYNorm * screenHeight;

      if (!useFallbackMode && workerRef.current) {
        const msg: ARRenderWorkerInboundMessage = {
          type: 'POINTER_DRAG',
          payload: {
            active: true,
            pointerX: px,
            pointerY: py,
            activeDragHandle: localDragRef.current.lineHandle,
            activeRoiVertexIndex: null,
            lineEndpointsNorm: nextEndpoints,
            roiVerticesNorm,
            roiSnapGuides: [],
          },
        };
        workerRef.current.postMessage(msg);
      }

      onCountingLineChange?.(nextEndpoints, localDragRef.current.lineHandle);
    }
  };

  const handleTouchNodePointerUp = (
    e: React.PointerEvent<HTMLButtonElement>
  ) => {
    if (!localDragRef.current.active) return;
    e.stopPropagation();
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Ignorar si el puntero ya fue liberado
    }

    const wasRoi = localDragRef.current.roiIdx !== null;
    const wasLine = localDragRef.current.lineHandle !== null;
    localDragRef.current = { active: false, roiIdx: null, lineHandle: null };
    pointerLoupeRef.current = null;
    globalSpatialZonesROI.clearActiveSnapGuides();

    if (!useFallbackMode && workerRef.current) {
      const msg: ARRenderWorkerInboundMessage = {
        type: 'POINTER_DRAG',
        payload: {
          active: false,
          pointerX: 0,
          pointerY: 0,
          activeDragHandle: null,
          activeRoiVertexIndex: null,
          lineEndpointsNorm,
          roiVerticesNorm,
          roiSnapGuides: [],
        },
      };
      workerRef.current.postMessage(msg);
    }

    if (wasRoi && roiVerticesNorm) {
      onRoiVerticesChange?.(roiVerticesNorm, [], null);
    }
    if (wasLine) {
      onCountingLineChange?.(lineEndpointsNorm, null);
    }
  };

  return (
    <>
      {/* 1. Lienzo Principal transferido al Dedicated Web Worker (OffscreenCanvas 60 FPS) */}
      <canvas
        ref={workerCanvasRef}
        className={`fixed inset-0 w-screen h-dvh pointer-events-none select-none z-10 transition-opacity duration-150 ${
          useFallbackMode ? 'opacity-0' : 'opacity-100'
        }`}
      />

      {/* 2. Lienzo de Fallback Transparente (Activo sólo cuando OffscreenCanvas no existe o durante grabación REC) */}
      <canvas
        ref={fallbackCanvasRef}
        className={`fixed inset-0 w-screen h-dvh pointer-events-none select-none z-10 transition-opacity duration-150 ${
          useFallbackMode ? 'opacity-100' : 'opacity-0'
        }`}
      />

      {/* 3. Capa Transparente de Captura Táctil (pointer-events-none en contenedor, pointer-events-auto en vértices V1..V4 y A/B) */}
      <div
        aria-hidden="true"
        className="fixed inset-0 w-screen h-dvh pointer-events-none select-none z-15"
      >
        {enableRoiPolygon &&
          roiVerticesNorm &&
          roiVerticesNorm.length === 4 &&
          screenWidth > 0 &&
          screenHeight > 0 &&
          roiVerticesNorm.map((v, idx) => (
            <button
              key={`roi-touch-vertex-${idx}`}
              type="button"
              aria-label={`Arrastrar vértice ROI V${idx + 1}`}
              onPointerDown={(e) =>
                handleTouchNodePointerDown(e, 'ROI', idx as 0 | 1 | 2 | 3)
              }
              onPointerMove={handleTouchNodePointerMove}
              onPointerUp={handleTouchNodePointerUp}
              onPointerCancel={handleTouchNodePointerUp}
              style={{
                left: `${v.x * 100}%`,
                top: `${v.y * 100}%`,
              }}
              className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-transparent cursor-grab active:cursor-grabbing touch-none"
            />
          ))}

        {showCountingLine && screenWidth > 0 && screenHeight > 0 && (
          <>
            <button
              type="button"
              aria-label="Arrastrar nodo A de línea virtual de aforo"
              onPointerDown={(e) => handleTouchNodePointerDown(e, 'LINE', 'a')}
              onPointerMove={handleTouchNodePointerMove}
              onPointerUp={handleTouchNodePointerUp}
              onPointerCancel={handleTouchNodePointerUp}
              style={{
                left: `${lineEndpointsNorm.x1 * 100}%`,
                top: `${lineEndpointsNorm.y1 * 100}%`,
              }}
              className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-transparent cursor-grab active:cursor-grabbing touch-none"
            />
            <button
              type="button"
              aria-label="Arrastrar nodo B de línea virtual de aforo"
              onPointerDown={(e) => handleTouchNodePointerDown(e, 'LINE', 'b')}
              onPointerMove={handleTouchNodePointerMove}
              onPointerUp={handleTouchNodePointerUp}
              onPointerCancel={handleTouchNodePointerUp}
              style={{
                left: `${lineEndpointsNorm.x2 * 100}%`,
                top: `${lineEndpointsNorm.y2 * 100}%`,
              }}
              className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-transparent cursor-grab active:cursor-grabbing touch-none"
            />
          </>
        )}
      </div>

      {/* 4. Píldora Discreta de Telemetría del Motor de Renderizado (OffscreenWorker 60 FPS vs Fallback) */}
      <div className="fixed bottom-16 left-3 z-20 pointer-events-none hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-[#090D16]/75 backdrop-blur-md border border-slate-700/70 text-[10.5px] font-mono text-slate-200 shadow-md">
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            workerTelemetry.mode === 'OffscreenWorker'
              ? 'bg-emerald-400 animate-pulse'
              : 'bg-amber-400'
          }`}
        />
        <span>{workerTelemetry.label}</span>
      </div>
    </>
  );
};
