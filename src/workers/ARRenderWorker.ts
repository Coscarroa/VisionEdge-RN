/**
 * Dedicated Web Worker de Renderizado AR a 60 FPS (OffscreenCanvas + Transferable Objects)
 * (src/workers/ARRenderWorker.ts)
 *
 * - Recibe el OffscreenCanvas mediante Transferable Objects (INIT_CANVAS).
 * - Decodifica los buffers binarios Zero-Copy (Float32Array) enviados por el hilo principal (FRAME_DATA).
 * - Mantiene su propio bucle requestAnimationFrame autónomo a 60 FPS dentro del Worker,
 *   desacoplado de la inferencia ONNX (~25-30 FPS) y de los re-renders de React.
 * - Soporta eventos de control en tiempo real:
 *   INIT_CANVAS, RESIZE, FRAME_DATA, ROI_UPDATE, POINTER_DRAG, THEME_CHANGE, GYRO_UPDATE y SET_ACTIVE.
 */

import { ARThemeId } from '../config/ARThemeConfig';
import {
  ARFrameMetadataPayload,
  ARPointerDragPayload,
  decodeDetectionsFromBinaryBuffer,
} from '../rendering/arBinaryProtocol';
import {
  ARCanvasRendererEngine,
  ARCanvasRenderState,
  ARInertialGyroState,
  renderARFrame,
} from '../rendering/arCanvasRenderer';
import { RoiSnapGuide, RoiVerticesNorm } from '../modules/SpatialZonesROI';
import { VirtualLineEndpointsNorm } from '../utils/VirtualCountingLine';

export type ARRenderWorkerInboundMessage =
  | {
      type: 'INIT_CANVAS';
      canvas: OffscreenCanvas;
      initialState: ARCanvasRenderState;
      gyro: ARInertialGyroState;
    }
  | {
      type: 'RESIZE';
      screenWidth: number;
      screenHeight: number;
      dpr: number;
    }
  | {
      type: 'FRAME_DATA';
      buffer: ArrayBuffer;
      meta: ARFrameMetadataPayload;
    }
  | {
      type: 'ROI_UPDATE';
      enableRoiPolygon: boolean;
      roiVerticesNorm?: RoiVerticesNorm;
      activeRoiVertexIndex: 0 | 1 | 2 | 3 | null;
      activeRoiViolationInView: boolean;
      roiSnapGuides: RoiSnapGuide[];
      showCountingLine: boolean;
      lineEndpointsNorm: VirtualLineEndpointsNorm;
      activeDragHandle: 'a' | 'b' | null;
    }
  | {
      type: 'POINTER_DRAG';
      payload: ARPointerDragPayload;
    }
  | {
      type: 'THEME_CHANGE';
      arTheme: ARThemeId;
      enablePose: boolean;
      enableGyro: boolean;
      onlyVehiclesMode: boolean;
    }
  | {
      type: 'GYRO_UPDATE';
      gyro: ARInertialGyroState;
    }
  | {
      type: 'SET_ACTIVE';
      active: boolean;
    };

export type ARRenderWorkerOutboundMessage = {
  type: 'WORKER_TELEMETRY';
  fps: number;
  decodeTimeMs: number;
  renderTimeMs: number;
  activeObjects: number;
};

let offscreenCanvas: OffscreenCanvas | null = null;
let offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
let currentState: ARCanvasRenderState | null = null;

const rendererEngine = new ARCanvasRendererEngine();
let isLoopRunning = false;
let rafHandle: number | null = null;
let lastFrameTimeMs = performance.now();
let lastTelemetryBroadcastMs = performance.now();
let lastDecodeTimeMs = 0;

function syncOffscreenDimensions(
  sw: number,
  sh: number,
  dpr: number
): void {
  if (!offscreenCanvas) return;
  const safeDpr = Math.min(2.5, Math.max(1, dpr || 1));
  const targetW = Math.max(1, Math.round(sw * safeDpr));
  const targetH = Math.max(1, Math.round(sh * safeDpr));
  if (
    offscreenCanvas.width !== targetW ||
    offscreenCanvas.height !== targetH
  ) {
    offscreenCanvas.width = targetW;
    offscreenCanvas.height = targetH;
  }
}

function scheduleNextWorkerFrame(cb: (now: number) => void): number {
  if (typeof self.requestAnimationFrame === 'function') {
    return self.requestAnimationFrame(cb);
  }
  return self.setTimeout(
    () => cb(performance.now()),
    16.666
  ) as unknown as number;
}

function cancelScheduledWorkerFrame(id: number | null): void {
  if (id === null) return;
  if (typeof self.cancelAnimationFrame === 'function') {
    self.cancelAnimationFrame(id);
  } else {
    self.clearTimeout(id);
  }
}

function workerRenderTick(nowMs: number): void {
  if (!isLoopRunning) {
    rafHandle = null;
    return;
  }

  const dtSec = Math.min(
    0.05,
    Math.max(0.004, (nowMs - lastFrameTimeMs) * 0.001)
  );
  lastFrameTimeMs = nowMs;

  if (offscreenCtx && currentState) {
    syncOffscreenDimensions(
      currentState.screenWidth,
      currentState.screenHeight,
      currentState.dpr
    );

    const t0 = performance.now();
    renderARFrame(offscreenCtx, currentState, nowMs, dtSec, rendererEngine);
    const renderTimeMs = performance.now() - t0;

    if (nowMs - lastTelemetryBroadcastMs >= 500) {
      lastTelemetryBroadcastMs = nowMs;
      const telemetryMsg: ARRenderWorkerOutboundMessage = {
        type: 'WORKER_TELEMETRY',
        fps: rendererEngine.getMeasuredRenderFps(),
        decodeTimeMs: Number(lastDecodeTimeMs.toFixed(3)),
        renderTimeMs: Number(renderTimeMs.toFixed(2)),
        activeObjects: currentState.detections.length,
      };
      self.postMessage(telemetryMsg);
    }
  }

  rafHandle = scheduleNextWorkerFrame(workerRenderTick);
}

function startWorkerLoop(): void {
  if (isLoopRunning) return;
  isLoopRunning = true;
  lastFrameTimeMs = performance.now();
  rafHandle = scheduleNextWorkerFrame(workerRenderTick);
}

function stopWorkerLoop(): void {
  isLoopRunning = false;
  cancelScheduledWorkerFrame(rafHandle);
  rafHandle = null;
}

self.onmessage = (event: MessageEvent<ARRenderWorkerInboundMessage>) => {
  const msg = event.data;
  if (!msg) return;

  switch (msg.type) {
    case 'INIT_CANVAS': {
      offscreenCanvas = msg.canvas;
      offscreenCtx = offscreenCanvas.getContext('2d', {
        alpha: true,
        desynchronized: true,
      });
      currentState = msg.initialState;
      rendererEngine.updateGyroOffset(msg.gyro);
      syncOffscreenDimensions(
        msg.initialState.screenWidth,
        msg.initialState.screenHeight,
        msg.initialState.dpr
      );
      startWorkerLoop();
      break;
    }

    case 'RESIZE': {
      if (currentState) {
        currentState.screenWidth = msg.screenWidth;
        currentState.screenHeight = msg.screenHeight;
        currentState.dpr = msg.dpr;
      }
      syncOffscreenDimensions(msg.screenWidth, msg.screenHeight, msg.dpr);
      break;
    }

    case 'FRAME_DATA': {
      if (!currentState) break;
      const t0 = performance.now();
      const decodedDetections = decodeDetectionsFromBinaryBuffer(
        msg.buffer,
        msg.meta
      );
      lastDecodeTimeMs = performance.now() - t0;

      currentState.detections = decodedDetections;
      currentState.countingStats = msg.meta.countingStats;
      currentState.activeRoiViolationInView = msg.meta.activeRoiViolationInView;
      currentState.proximityLinks = msg.meta.proximityLinks;
      currentState.showHeatmapOverlay = msg.meta.showHeatmapOverlay;
      currentState.heatmapPoints = msg.meta.heatmapPoints;
      break;
    }

    case 'ROI_UPDATE': {
      if (!currentState) break;
      currentState.enableRoiPolygon = msg.enableRoiPolygon;
      currentState.roiVerticesNorm = msg.roiVerticesNorm;
      currentState.activeRoiVertexIndex = msg.activeRoiVertexIndex;
      currentState.activeRoiViolationInView = msg.activeRoiViolationInView;
      currentState.roiSnapGuides = msg.roiSnapGuides;
      currentState.showCountingLine = msg.showCountingLine;
      currentState.lineEndpointsNorm = msg.lineEndpointsNorm;
      currentState.activeDragHandle = msg.activeDragHandle;
      if (
        msg.activeRoiVertexIndex === null &&
        msg.activeDragHandle === null
      ) {
        currentState.activePointerLoupe = null;
      }
      break;
    }

    case 'POINTER_DRAG': {
      if (!currentState) break;
      const { payload } = msg;
      currentState.activeDragHandle = payload.activeDragHandle;
      currentState.activeRoiVertexIndex = payload.activeRoiVertexIndex;
      currentState.lineEndpointsNorm = payload.lineEndpointsNorm;
      if (payload.roiVerticesNorm) {
        currentState.roiVerticesNorm = payload.roiVerticesNorm;
      }
      currentState.roiSnapGuides = payload.roiSnapGuides;

      if (!payload.active) {
        currentState.activePointerLoupe = null;
      } else if (payload.activeRoiVertexIndex !== null) {
        const hasSnap = payload.roiSnapGuides.length > 0;
        currentState.activePointerLoupe = {
          active: true,
          x: payload.pointerX,
          y: payload.pointerY,
          label: hasSnap
            ? `SNAP V${payload.activeRoiVertexIndex + 1}`
            : `ROI V${payload.activeRoiVertexIndex + 1}`,
          accentColor: hasSnap ? '#22D3EE' : '#F59E0B',
        };
      } else if (payload.activeDragHandle !== null) {
        currentState.activePointerLoupe = {
          active: true,
          x: payload.pointerX,
          y: payload.pointerY,
          label: `NODO ${payload.activeDragHandle.toUpperCase()}`,
          accentColor: '#38BDF8',
        };
      } else {
        currentState.activePointerLoupe = null;
      }
      break;
    }

    case 'THEME_CHANGE': {
      if (!currentState) break;
      currentState.arTheme = msg.arTheme;
      currentState.enablePose = msg.enablePose;
      currentState.enableGyro = msg.enableGyro;
      currentState.onlyVehiclesMode = msg.onlyVehiclesMode;
      break;
    }

    case 'GYRO_UPDATE': {
      rendererEngine.updateGyroOffset(msg.gyro);
      break;
    }

    case 'SET_ACTIVE': {
      if (msg.active) {
        startWorkerLoop();
      } else {
        stopWorkerLoop();
      }
      break;
    }
  }
};
