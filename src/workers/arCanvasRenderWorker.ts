/**
 * Dedicated Web Worker de Renderizado AR con OffscreenCanvas (60/120 FPS)
 * (src/workers/arCanvasRenderWorker.ts)
 *
 * - Recibe la transferencia directa del <canvas> principal mediante canvas.transferControlToOffscreen().
 * - Ejecuta su propio bucle de animación independiente (self.requestAnimationFrame con fallback a 16ms)
 *   para liberar por completo al hilo principal (Main Thread) del dibujo de retículas, estelas,
 *   esqueletos de 17 keypoints, polígonos ROI complejos y vectores de proximidad peatón-vehículo.
 */

import {
  ARGyroWorkerState,
  AROverlayRendererEngine,
  ARSceneWorkerState,
} from './arOverlayRendererCore';

export type ARRenderWorkerMessage =
  | {
      type: 'INIT_CANVAS';
      canvas: OffscreenCanvas;
      scene: ARSceneWorkerState;
      gyro: ARGyroWorkerState;
    }
  | {
      type: 'UPDATE_SCENE';
      scene: ARSceneWorkerState;
    }
  | {
      type: 'UPDATE_GYRO';
      gyro: ARGyroWorkerState;
    }
  | {
      type: 'RESIZE';
      screenWidth: number;
      screenHeight: number;
      dpr: number;
    }
  | {
      type: 'SET_ACTIVE';
      active: boolean;
    };

let offscreenCanvas: OffscreenCanvas | null = null;
let offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
let currentScene: ARSceneWorkerState | null = null;
let isLoopActive = false;
let rafHandle: number | null = null;
let timerHandle: ReturnType<typeof setTimeout> | null = null;
let lastFrameTime = performance.now();

const rendererEngine = new AROverlayRendererEngine();

function ensureCanvasDimensions(sw: number, sh: number, dpr: number): void {
  if (!offscreenCanvas || sw <= 0 || sh <= 0) return;
  const targetPixelW = Math.round(sw * dpr);
  const targetPixelH = Math.round(sh * dpr);
  if (
    offscreenCanvas.width !== targetPixelW ||
    offscreenCanvas.height !== targetPixelH
  ) {
    offscreenCanvas.width = targetPixelW;
    offscreenCanvas.height = targetPixelH;
  }
}

function scheduleNextTick(): void {
  if (!isLoopActive) return;
  if (typeof self.requestAnimationFrame === 'function') {
    rafHandle = self.requestAnimationFrame(workerRenderLoop);
  } else {
    timerHandle = setTimeout(() => workerRenderLoop(performance.now()), 16);
  }
}

function cancelScheduledTick(): void {
  if (rafHandle !== null && typeof self.cancelAnimationFrame === 'function') {
    self.cancelAnimationFrame(rafHandle);
    rafHandle = null;
  }
  if (timerHandle !== null) {
    clearTimeout(timerHandle);
    timerHandle = null;
  }
}

function workerRenderLoop(now: number): void {
  if (!isLoopActive) return;

  const dtSec = Math.min(0.05, Math.max(0.004, (now - lastFrameTime) * 0.001));
  lastFrameTime = now;

  if (offscreenCanvas && offscreenCtx && currentScene) {
    ensureCanvasDimensions(
      currentScene.screenWidth,
      currentScene.screenHeight,
      currentScene.dpr
    );
    rendererEngine.renderFrame(offscreenCtx, currentScene, now, dtSec);
  }

  scheduleNextTick();
}

self.onmessage = (event: MessageEvent<ARRenderWorkerMessage>) => {
  const msg = event.data;
  if (!msg) return;

  switch (msg.type) {
    case 'INIT_CANVAS': {
      offscreenCanvas = msg.canvas;
      offscreenCtx = offscreenCanvas.getContext('2d', {
        alpha: true,
        desynchronized: true,
      });
      currentScene = msg.scene;
      rendererEngine.updateGyroOffset(msg.gyro);
      ensureCanvasDimensions(
        msg.scene.screenWidth,
        msg.scene.screenHeight,
        msg.scene.dpr
      );

      if (!isLoopActive) {
        isLoopActive = true;
        lastFrameTime = performance.now();
        scheduleNextTick();
      }
      break;
    }

    case 'UPDATE_SCENE': {
      currentScene = msg.scene;
      break;
    }

    case 'UPDATE_GYRO': {
      rendererEngine.updateGyroOffset(msg.gyro);
      break;
    }

    case 'RESIZE': {
      if (currentScene) {
        currentScene.screenWidth = msg.screenWidth;
        currentScene.screenHeight = msg.screenHeight;
        currentScene.dpr = msg.dpr;
      }
      ensureCanvasDimensions(msg.screenWidth, msg.screenHeight, msg.dpr);
      break;
    }

    case 'SET_ACTIVE': {
      if (msg.active && !isLoopActive) {
        isLoopActive = true;
        lastFrameTime = performance.now();
        scheduleNextTick();
      } else if (!msg.active && isLoopActive) {
        isLoopActive = false;
        cancelScheduledTick();
      }
      break;
    }
  }
};
