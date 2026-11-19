import React, { useEffect, useRef } from 'react';
import { ARThemeId } from '../config/ARThemeConfig';
import { OnnxKeypoint2D, OnnxTrackedDetection } from '../hooks/useOnnxVisionEngine';
import { InertialARState } from '../hooks/useInertialARTracker';
import {
  globalVirtualCountingLine,
  VirtualLineEndpointsNorm,
} from '../utils/VirtualCountingLine';
import { RoiSnapGuide, RoiVerticesNorm } from '../modules/SpatialZonesROI';
import { ProximityHazardLink } from '../modules/HumanSafetyAnalytics';
import {
  ARGyroWorkerState,
  AROverlayRendererEngine,
  ARSceneWorkerState,
} from '../workers/arOverlayRendererCore';
import { ARRenderWorkerMessage } from '../workers/arCanvasRenderWorker';

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
  roiVerticesNorm,
  activeRoiVertexIndex = null,
  activeRoiViolationInView = false,
  roiSnapGuides = [],
  proximityLinks = [],
  showHeatmapOverlay,
  heatmapPoints,
  screenWidth,
  screenHeight,
  decayInertialOffsetOnRenderFrame,
  onCanvasReady,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const usingOffscreenWorkerRef = useRef<boolean>(false);

  const getDevicePixelRatio = () =>
    typeof window !== 'undefined'
      ? Math.min(2.5, window.devicePixelRatio || 1)
      : 1;

  const buildSceneSnapshot = (): ARSceneWorkerState => ({
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
    proximityLinks,
    showHeatmapOverlay,
    heatmapPoints,
    screenWidth,
    screenHeight,
    dpr: getDevicePixelRatio(),
  });

  const latestSceneRef = useRef<ARSceneWorkerState>(buildSceneSnapshot());
  latestSceneRef.current = buildSceneSnapshot();

  const decayFnRef = useRef(decayInertialOffsetOnRenderFrame);
  decayFnRef.current = decayInertialOffsetOnRenderFrame;

  // Entregar la referencia del <canvas> al compositor MediaRecorder AR
  useEffect(() => {
    onCanvasReady?.(canvasRef.current);
    return () => onCanvasReady?.(null);
  }, [onCanvasReady]);

  // 1. Inicialización de OffscreenCanvas + Dedicated Render Worker (con fallback automático a Canvas 2D)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const supportsOffscreen =
      typeof window !== 'undefined' &&
      typeof Worker !== 'undefined' &&
      typeof canvas.transferControlToOffscreen === 'function';

    if (supportsOffscreen) {
      try {
        let entry = OFFSCREEN_WORKER_REGISTRY.get(canvas);

        if (!entry) {
          const offscreen = canvas.transferControlToOffscreen();
          const worker = new Worker(
            new URL('../workers/arCanvasRenderWorker.ts', import.meta.url),
            { type: 'module' }
          );

          const initialGyro = decayFnRef.current();
          const initMsg: ARRenderWorkerMessage = {
            type: 'INIT_CANVAS',
            canvas: offscreen,
            scene: latestSceneRef.current,
            gyro: {
              offsetX: initialGyro.gx,
              offsetY: initialGyro.gy,
              enabled: latestSceneRef.current.enableGyro,
            },
          };
          worker.postMessage(initMsg, [offscreen]);

          entry = { worker, initialized: true };
          OFFSCREEN_WORKER_REGISTRY.set(canvas, entry);
        } else {
          // React StrictMode re-mount sobre el mismo nodo <canvas>: reactivamos el Worker existente
          const resumeMsg: ARRenderWorkerMessage = {
            type: 'SET_ACTIVE',
            active: true,
          };
          entry.worker.postMessage(resumeMsg);
          const syncMsg: ARRenderWorkerMessage = {
            type: 'UPDATE_SCENE',
            scene: latestSceneRef.current,
          };
          entry.worker.postMessage(syncMsg);
        }

        workerRef.current = entry.worker;
        usingOffscreenWorkerRef.current = true;

        // Sincronizador ligero de offset inercial (Kalman IMU) hacia el Worker
        let gyroSyncRafId: number;
        let lastSentGx = 0;
        let lastSentGy = 0;

        const syncGyroToWorkerLoop = () => {
          const { gx, gy } = decayFnRef.current();
          const gyroEnabled = latestSceneRef.current.enableGyro;

          if (
            Math.abs(gx - lastSentGx) > 0.08 ||
            Math.abs(gy - lastSentGy) > 0.08
          ) {
            lastSentGx = gx;
            lastSentGy = gy;
            const gyroPayload: ARGyroWorkerState = {
              offsetX: gx,
              offsetY: gy,
              enabled: gyroEnabled,
            };
            const msg: ARRenderWorkerMessage = {
              type: 'UPDATE_GYRO',
              gyro: gyroPayload,
            };
            entry?.worker.postMessage(msg);
          }

          gyroSyncRafId = window.requestAnimationFrame(syncGyroToWorkerLoop);
        };

        gyroSyncRafId = window.requestAnimationFrame(syncGyroToWorkerLoop);

        return () => {
          window.cancelAnimationFrame(gyroSyncRafId);
          const pauseMsg: ARRenderWorkerMessage = {
            type: 'SET_ACTIVE',
            active: false,
          };
          entry?.worker.postMessage(pauseMsg);
        };
      } catch {
        // Si el navegador rechaza transferControlToOffscreen, usamos fallback 2D en hilo principal
        usingOffscreenWorkerRef.current = false;
      }
    }

    // Fallback para navegadores sin OffscreenCanvas: ejecuta AROverlayRendererEngine en Canvas 2D
    usingOffscreenWorkerRef.current = false;
    const fallbackEngine = new AROverlayRendererEngine();
    let fallbackRafId: number;
    let lastTime = performance.now();

    const fallbackLoop = (now: number) => {
      const dtSec = Math.min(0.05, Math.max(0.004, (now - lastTime) * 0.001));
      lastTime = now;

      const ctx = canvas.getContext('2d');
      const scene = latestSceneRef.current;
      if (ctx && scene.screenWidth > 0 && scene.screenHeight > 0) {
        const targetW = Math.round(scene.screenWidth * scene.dpr);
        const targetH = Math.round(scene.screenHeight * scene.dpr);
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }
        const { gx, gy } = decayFnRef.current();
        fallbackEngine.updateGyroOffset({
          offsetX: gx,
          offsetY: gy,
          enabled: scene.enableGyro,
        });
        fallbackEngine.renderFrame(ctx, scene, now, dtSec);
      }

      fallbackRafId = window.requestAnimationFrame(fallbackLoop);
    };

    fallbackRafId = window.requestAnimationFrame(fallbackLoop);
    return () => window.cancelAnimationFrame(fallbackRafId);
  }, []);

  // 2. Envío reactivo de cambios de dimensión (RESIZE) al Worker dedicado
  useEffect(() => {
    if (!usingOffscreenWorkerRef.current || !workerRef.current) return;
    const resizeMsg: ARRenderWorkerMessage = {
      type: 'RESIZE',
      screenWidth,
      screenHeight,
      dpr: getDevicePixelRatio(),
    };
    workerRef.current.postMessage(resizeMsg);
  }, [screenWidth, screenHeight]);

  // 3. Envío asíncrono del estado de la escena (Detecciones, Polígono ROI, Aforo, Proximidad, Tema) al Worker
  useEffect(() => {
    if (!usingOffscreenWorkerRef.current || !workerRef.current) return;
    const updateMsg: ARRenderWorkerMessage = {
      type: 'UPDATE_SCENE',
      scene: latestSceneRef.current,
    };
    workerRef.current.postMessage(updateMsg);
  }, [
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
  ]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-screen h-dvh pointer-events-none select-none z-10"
    />
  );
};
