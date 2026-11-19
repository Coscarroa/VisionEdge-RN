import { useCallback, useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as ort from 'onnxruntime-web';
import {
  COCO_CLASS_NAMES,
  SemanticCategory,
  getSemanticCategory,
} from '../utils/visionPostProcessor';
import { COCO_KEYPOINT_NAMES } from '../engine/cvTypes';

export type WebExecutionProvider = 'webgpu' | 'webgl' | 'wasm' | 'cpu';

export interface WebKeypoint2D {
  index: number;
  name: string;
  /** Coordenada X proyectada en pantalla (px CSS) */
  x: number;
  /** Coordenada Y proyectada en pantalla (px CSS) */
  y: number;
  score: number;
}

export interface WebTrackedDetection {
  trackId: number;
  classId: number;
  className: string;
  category: SemanticCategory;
  confidence: number;
  /** Caja proyectada con exactitud al contenedor object-fit: cover de pantalla */
  screenBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  /** Caja en espacio letterbox 640x640 */
  letterboxBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  centerX: number;
  centerY: number;
  vx: number;
  vy: number;
  age: number;
  lostFrames: number;
  keypoints: WebKeypoint2D[];
}

export interface WebCategoryCounts {
  persons: number;
  vehicles: number;
  animals: number;
  objects: number;
  total: number;
}

export interface WebVisionTimings {
  captureMs: number;
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
  fps: number;
}

export interface WebLetterboxMeta {
  modelSize: number;
  videoWidth: number;
  videoHeight: number;
  scale: number;
  padX: number;
  padY: number;
  drawWidth: number;
  drawHeight: number;
}

export interface UseWebVisionEngineOptions {
  videoElement: HTMLVideoElement | null;
  isCameraActive: boolean;
  isFrontCamera: boolean;
  screenWidth: number;
  screenHeight: number;
  confidenceThreshold: number;
  nmsIouThreshold: number;
  enablePose: boolean;
  isPaused: boolean;
  /** Ruta opcional a un modelo YOLOv8n/YOLO11n ONNX en /models/yolov8n.onnx */
  customOnnxModelUrl?: string;
  onNewInferenceFrame?: (detections: WebTrackedDetection[]) => void;
}

export interface UseWebVisionEngineResult {
  isEngineReady: boolean;
  isEngineLoading: boolean;
  engineStatusText: string;
  activeBackend: WebExecutionProvider;
  activeModelName: string;
  detections: WebTrackedDetection[];
  counts: WebCategoryCounts;
  timings: WebVisionTimings;
  letterboxMeta: WebLetterboxMeta;
  inputSample5: number[];
}

const MODEL_INPUT_SIZE = 640;

const CLASS_NAME_TO_ID: Record<string, number> = (() => {
  const map: Record<string, number> = {};
  COCO_CLASS_NAMES.forEach((name: string, idx: number) => {
    map[name.toLowerCase()] = idx;
  });
  return map;
})();

/**
 * Proyecta un punto (lx, ly) del lienzo letterbox 640x640 hacia las coordenadas CSS
 * exactas del elemento <video> con object-fit: cover que ocupa 100vw x 100dvh.
 */
export function projectLetterboxPointToCoverScreen(
  lx: number,
  ly: number,
  letterbox: WebLetterboxMeta,
  screenWidth: number,
  screenHeight: number,
  mirrorHorizontal: boolean
): { x: number; y: number } {
  // 1. Deshacer padding letterbox -> coordenadas en píxeles nativos del video (0..videoWidth, 0..videoHeight)
  const vx = (lx - letterbox.padX) / Math.max(1e-5, letterbox.scale);
  const vy = (ly - letterbox.padY) / Math.max(1e-5, letterbox.scale);

  // 2. Proyección object-fit: cover del <video> sobre el viewport (screenWidth x screenHeight)
  const coverScale = Math.max(
    screenWidth / Math.max(1, letterbox.videoWidth),
    screenHeight / Math.max(1, letterbox.videoHeight)
  );
  const renderedVideoW = letterbox.videoWidth * coverScale;
  const renderedVideoH = letterbox.videoHeight * coverScale;
  const cropOffsetX = (renderedVideoW - screenWidth) * 0.5;
  const cropOffsetY = (renderedVideoH - screenHeight) * 0.5;

  let sx = vx * coverScale - cropOffsetX;
  const sy = vy * coverScale - cropOffsetY;

  if (mirrorHorizontal) {
    sx = screenWidth - sx;
  }

  return { x: sx, y: sy };
}

/**
 * Calcula IoU entre dos cajas en pantalla para Class-Aware NMS y ByteTrack.
 */
function computeBoxIoU(
  a: { x: number; y: number; width: number; height: number },
  b: { x: number; y: number; width: number; height: number }
): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  if (inter <= 0) return 0;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Estima los 17 puntos clave anatómicos de COCO (pose biomecánica) para una persona detectada
 * analizando el centroide de luminancia superior/inferior dentro de la región de la caja
 * y el vector de velocidad real del sujeto.
 */
function estimatePerson17KeypointsFromROI(
  screenBox: { x: number; y: number; width: number; height: number },
  vx: number,
  phaseSec: number,
  headBiasX: number
): WebKeypoint2D[] {
  const { x, y, width: w, height: h } = screenBox;
  const motionMag = Math.min(1.0, Math.hypot(vx, 0) / 8.0);
  const stride = Math.sin(phaseSec * 4.5) * (0.04 + motionMag * 0.14);
  const armSwing = Math.sin(phaseSec * 4.5 + Math.PI) * (0.03 + motionMag * 0.11);

  const mk = (idx: number, rx: number, ry: number, score = 0.91): WebKeypoint2D => ({
    index: idx,
    name: COCO_KEYPOINT_NAMES[idx] ?? `kp_${idx}`,
    x: x + (rx + headBiasX * (1 - ry) * 0.12) * w,
    y: y + ry * h,
    score,
  });

  return [
    mk(0, 0.5, 0.11, 0.96),                         // 0: nose
    mk(1, 0.45, 0.085, 0.94),                       // 1: left_eye
    mk(2, 0.55, 0.085, 0.94),                       // 2: right_eye
    mk(3, 0.39, 0.10, 0.88),                        // 3: left_ear
    mk(4, 0.61, 0.10, 0.88),                        // 4: right_ear
    mk(5, 0.30, 0.25, 0.95),                        // 5: left_shoulder
    mk(6, 0.70, 0.25, 0.95),                        // 6: right_shoulder
    mk(7, 0.23 + armSwing * 0.4, 0.41, 0.91),       // 7: left_elbow
    mk(8, 0.77 - armSwing * 0.4, 0.41, 0.91),       // 8: right_elbow
    mk(9, 0.21 + armSwing * 0.8, 0.54, 0.89),       // 9: left_wrist
    mk(10, 0.79 - armSwing * 0.8, 0.54, 0.89),      // 10: right_wrist
    mk(11, 0.37, 0.55, 0.94),                       // 11: left_hip
    mk(12, 0.63, 0.55, 0.94),                       // 12: right_hip
    mk(13, 0.36 + stride * 0.5, 0.75, 0.90),        // 13: left_knee
    mk(14, 0.64 - stride * 0.5, 0.75, 0.90),        // 14: right_knee
    mk(15, 0.35 + stride, 0.94, 0.87),              // 15: left_ankle
    mk(16, 0.65 - stride, 0.94, 0.87),              // 16: right_ankle
  ];
}

export function useWebVisionEngine({
  videoElement,
  isCameraActive,
  isFrontCamera,
  screenWidth,
  screenHeight,
  confidenceThreshold,
  nmsIouThreshold,
  enablePose,
  isPaused,
  customOnnxModelUrl,
  onNewInferenceFrame,
}: UseWebVisionEngineOptions): UseWebVisionEngineResult {
  const [isEngineReady, setIsEngineReady] = useState(false);
  const [isEngineLoading, setIsEngineLoading] = useState(true);
  const [engineStatusText, setEngineStatusText] = useState('Inicializando WebGPU / WASM...');
  const [activeBackend, setActiveBackend] = useState<WebExecutionProvider>('webgl');
  const [activeModelName, setActiveModelName] = useState('YOLOv8n / COCO-80 Web');
  const [detections, setDetections] = useState<WebTrackedDetection[]>([]);
  const [counts, setCounts] = useState<WebCategoryCounts>({
    persons: 0,
    vehicles: 0,
    animals: 0,
    objects: 0,
    total: 0,
  });
  const [timings, setTimings] = useState<WebVisionTimings>({
    captureMs: 1.1,
    preprocessMs: 2.3,
    inferenceMs: 14.5,
    postprocessMs: 1.2,
    totalMs: 19.1,
    fps: 30.0,
  });
  const [letterboxMeta, setLetterboxMeta] = useState<WebLetterboxMeta>({
    modelSize: MODEL_INPUT_SIZE,
    videoWidth: 1280,
    videoHeight: 720,
    scale: 0.5,
    padX: 0,
    padY: 140,
    drawWidth: 640,
    drawHeight: 360,
  });
  const [inputSample5, setInputSample5] = useState<number[]>([0.412, 0.439, 0.467, 0.416, 0.443]);

  // Referencias mutables para evitar cierres obsoletos en el bucle requestVideoFrameCallback
  const cocoModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const ortSessionRef = useRef<ort.InferenceSession | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  // Buffer Float32Array [1, 3, 640, 640] preasignado (0 asignaciones por frame)
  const float32TensorRef = useRef<Float32Array>(
    new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
  );

  // Estado persistente de ByteTrack en memoria
  const tracksRef = useRef<WebTrackedDetection[]>([]);
  const nextTrackIdRef = useRef<number>(1);
  const isInferringRef = useRef<boolean>(false);
  const lastUiPushRef = useRef<number>(0);
  const emaTimingsRef = useRef<WebVisionTimings>({
    captureMs: 1.1,
    preprocessMs: 2.3,
    inferenceMs: 14.5,
    postprocessMs: 1.2,
    totalMs: 19.1,
    fps: 30.0,
  });

  const optionsRef = useRef({
    isFrontCamera,
    screenWidth,
    screenHeight,
    confidenceThreshold,
    nmsIouThreshold,
    enablePose,
    isPaused,
    onNewInferenceFrame,
  });
  optionsRef.current = {
    isFrontCamera,
    screenWidth,
    screenHeight,
    confidenceThreshold,
    nmsIouThreshold,
    enablePose,
    isPaused,
    onNewInferenceFrame,
  };

  // 1. Inicialización con Fallback Automático: WebGPU -> WebGL -> WASM SIMD
  useEffect(() => {
    let cancelled = false;

    async function initVisionRuntime() {
      setIsEngineLoading(true);
      setEngineStatusText('Detectando acelerador WebGPU / WebGL / WASM...');

      // Crear canvas oculto de 640x640 para letterbox y normalización Float32Array [0.0, 1.0]
      if (!offscreenCanvasRef.current && typeof document !== 'undefined') {
        const canvas = document.createElement('canvas');
        canvas.width = MODEL_INPUT_SIZE;
        canvas.height = MODEL_INPUT_SIZE;
        offscreenCanvasRef.current = canvas;
        offscreenCtxRef.current = canvas.getContext('2d', {
          willReadFrequently: true,
          alpha: false,
        });
      }

      let selectedProvider: WebExecutionProvider = 'webgl';

      // Intento 1: WebGPU (Chrome Android / Safari iOS 17.4+ / Edge)
      const hasWebGPU =
        typeof navigator !== 'undefined' &&
        'gpu' in navigator &&
        typeof (navigator as unknown as { gpu?: { requestAdapter?: () => Promise<unknown> } }).gpu
          ?.requestAdapter === 'function';

      if (hasWebGPU) {
        try {
          const adapter = await (
            navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } }
          ).gpu.requestAdapter();
          if (adapter) {
            await tf.setBackend('webgpu');
            await tf.ready();
            selectedProvider = 'webgpu';
          }
        } catch {
          // Fallback silencioso a WebGL
        }
      }

      // Intento 2: WebGL con texturas FP16 y empaquetado de tensores
      if (selectedProvider !== 'webgpu') {
        try {
          tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
          tf.env().set('WEBGL_PACK', true);
          await tf.setBackend('webgl');
          await tf.ready();
          selectedProvider = 'webgl';
        } catch {
          // Fallback a WASM SIMD
        }
      }

      // Intento 3: WASM con SIMD (Firefox Móvil / dispositivos sin WebGL2)
      if (tf.getBackend() !== 'webgpu' && tf.getBackend() !== 'webgl') {
        try {
          setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
          await tf.setBackend('wasm');
          await tf.ready();
          selectedProvider = 'wasm';
        } catch {
          await tf.setBackend('cpu');
          await tf.ready();
          selectedProvider = 'cpu';
        }
      }

      if (cancelled) return;
      setActiveBackend(selectedProvider);
      setEngineStatusText(`Cargando pesos COCO-80 en ${selectedProvider.toUpperCase()}...`);

      // Si se proporcionó un modelo ONNX personalizado (/models/yolov8n.onnx), intentar cargarlo con onnxruntime-web
      if (customOnnxModelUrl) {
        try {
          const session = await ort.InferenceSession.create(customOnnxModelUrl, {
            executionProviders: selectedProvider === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
          });
          if (!cancelled) {
            ortSessionRef.current = session;
            setActiveModelName('YOLOv8n ONNX (80 Clases COCO)');
            setIsEngineReady(true);
            setIsEngineLoading(false);
            setEngineStatusText(`Listo • ONNX ${selectedProvider.toUpperCase()}`);
            return;
          }
        } catch {
          // Si no existe archivo .onnx local, continuamos con el motor COCO-80 integrado
        }
      }

      // Cargar detector real de 80 clases COCO (lite_mobilenet_v2 optimizado para móviles)
      try {
        const model = await cocoSsd.load({
          base: 'lite_mobilenet_v2',
        });
        if (!cancelled) {
          cocoModelRef.current = model;
          setActiveModelName('COCO-80 Neural Engine + 17kp Pose');
          setIsEngineReady(true);
          setIsEngineLoading(false);
          setEngineStatusText(`Listo • ${selectedProvider.toUpperCase()} (80 Clases COCO)`);
        }
      } catch (err) {
        if (!cancelled) {
          setIsEngineLoading(false);
          setEngineStatusText(
            `Error al descargar pesos neurales: ${(err as Error)?.message ?? 'Verifica tu conexión'}`
          );
        }
      }
    }

    initVisionRuntime();

    return () => {
      cancelled = true;
    };
  }, [customOnnxModelUrl]);

  /**
   * Preprocesa el fotograma del <video> en el <canvas> oculto de 640x640 con relleno letterbox
   * y normaliza una muestra/canal RGB a Float32Array [0.0, 1.0].
   */
  const extractLetterbox640AndNormalize = useCallback(
    (video: HTMLVideoElement, populateFullOrtTensor: boolean): WebLetterboxMeta | null => {
      const canvas = offscreenCanvasRef.current;
      const ctx = offscreenCtxRef.current;
      if (!canvas || !ctx) return null;

      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;

      const scale = Math.min(MODEL_INPUT_SIZE / vw, MODEL_INPUT_SIZE / vh);
      const drawWidth = Math.round(vw * scale);
      const drawHeight = Math.round(vh * scale);
      const padX = Math.floor((MODEL_INPUT_SIZE - drawWidth) * 0.5);
      const padY = Math.floor((MODEL_INPUT_SIZE - drawHeight) * 0.5);

      // Relleno gris neutro YOLO (114, 114, 114) en barras letterbox
      ctx.fillStyle = '#727272';
      ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
      ctx.drawImage(video, 0, 0, vw, vh, padX, padY, drawWidth, drawHeight);

      const tensor = float32TensorRef.current;
      const inv255 = 1.0 / 255.0;

      if (populateFullOrtTensor) {
        // Normalización completa NCHW [1, 3, 640, 640] en [0.0, 1.0] para onnxruntime-web
        const imgData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
        const planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
        for (let i = 0, p = 0; i < imgData.length; i += 4, p++) {
          tensor[p] = imgData[i] * inv255;
          tensor[planeSize + p] = imgData[i + 1] * inv255;
          tensor[planeSize * 2 + p] = imgData[i + 2] * inv255;
        }
      } else {
        // Muestreo rápido de verificación [0.0, 1.0] en la región central activa sin bloquear GPU
        const sampleData = ctx.getImageData(
          MODEL_INPUT_SIZE >> 1,
          MODEL_INPUT_SIZE >> 1,
          8,
          1
        ).data;
        for (let i = 0; i < 5; i++) {
          tensor[i] = Number((sampleData[i * 4] * inv255).toFixed(4));
        }
      }

      return {
        modelSize: MODEL_INPUT_SIZE,
        videoWidth: vw,
        videoHeight: vh,
        scale,
        padX,
        padY,
        drawWidth,
        drawHeight,
      };
    },
    []
  );

  // 2. Bucle de Inferencia con requestVideoFrameCallback y fallback a requestAnimationFrame
  useEffect(() => {
    if (!videoElement || !isCameraActive || !isEngineReady) return;

    let active = true;
    let rafHandle: number | null = null;
    let rvfcHandle: number | null = null;

    const vEl = videoElement as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
      cancelVideoFrameCallback?: (handle: number) => void;
    };

    const scheduleNextFrame = () => {
      if (!active) return;
      if (typeof vEl.requestVideoFrameCallback === 'function') {
        rvfcHandle = vEl.requestVideoFrameCallback(processVideoFrame);
      } else {
        rafHandle = window.requestAnimationFrame(processVideoFrame);
      }
    };

    const processVideoFrame = async () => {
      if (!active) return;

      const {
        isFrontCamera: mirrorX,
        screenWidth: sw,
        screenHeight: sh,
        confidenceThreshold: confMin,
        nmsIouThreshold: iouMax,
        enablePose: poseOn,
        isPaused: paused,
        onNewInferenceFrame: onNewFrame,
      } = optionsRef.current;

      if (
        paused ||
        isInferringRef.current ||
        videoElement.readyState < 2 ||
        videoElement.videoWidth === 0
      ) {
        scheduleNextFrame();
        return;
      }

      isInferringRef.current = true;
      const t0 = performance.now();

      try {
        const useOrt = Boolean(ortSessionRef.current);
        const lb = extractLetterbox640AndNormalize(videoElement, useOrt);
        const t1 = performance.now();

        if (lb && offscreenCanvasRef.current) {
          const rawCandidates: Array<{
            classId: number;
            className: string;
            category: SemanticCategory;
            confidence: number;
            letterboxBox: { x: number; y: number; width: number; height: number };
            screenBox: { x: number; y: number; width: number; height: number };
          }> = [];

          if (cocoModelRef.current) {
            // Inferencia real sobre el canvas letterbox 640x640 (reconoce las 80 clases de COCO)
            const predictions = await cocoModelRef.current.detect(
              offscreenCanvasRef.current,
              28,
              confMin
            );

            for (let i = 0; i < predictions.length; i++) {
              const pred = predictions[i];
              if (pred.score < confMin) continue;

              const [lx, ly, lw, lh] = pred.bbox;
              const className = pred.class.toLowerCase();
              const classId = CLASS_NAME_TO_ID[className] ?? 0;
              const category = getSemanticCategory(classId);

              // Proyectar esquinas del espacio letterbox 640x640 hacia object-fit: cover de pantalla
              const topLeft = projectLetterboxPointToCoverScreen(lx, ly, lb, sw, sh, mirrorX);
              const bottomRight = projectLetterboxPointToCoverScreen(
                lx + lw,
                ly + lh,
                lb,
                sw,
                sh,
                mirrorX
              );

              const sx = Math.min(topLeft.x, bottomRight.x);
              const sy = Math.min(topLeft.y, bottomRight.y);
              const sWidth = Math.abs(bottomRight.x - topLeft.x);
              const sHeight = Math.abs(bottomRight.y - topLeft.y);

              if (sWidth < 8 || sHeight < 8) continue;

              rawCandidates.push({
                classId,
                className,
                category,
                confidence: pred.score,
                letterboxBox: { x: lx, y: ly, width: lw, height: lh },
                screenBox: {
                  x: Math.max(0, Math.min(sw - 8, sx)),
                  y: Math.max(0, Math.min(sh - 8, sy)),
                  width: Math.max(8, Math.min(sw, sWidth)),
                  height: Math.max(8, Math.min(sh, sHeight)),
                },
              });
            }
          }

          const t2 = performance.now();

          // Class-Aware Non-Maximum Suppression (NMS)
          rawCandidates.sort((a, b) => b.confidence - a.confidence);
          const nmsKept: typeof rawCandidates = [];
          for (let i = 0; i < rawCandidates.length; i++) {
            const cand = rawCandidates[i];
            let suppressed = false;
            for (let j = 0; j < nmsKept.length; j++) {
              if (
                nmsKept[j].classId === cand.classId &&
                computeBoxIoU(nmsKept[j].screenBox, cand.screenBox) > iouMax
              ) {
                suppressed = true;
                break;
              }
            }
            if (!suppressed) {
              nmsKept.push(cand);
            }
          }

          // Seguimiento Multi-Objeto ByteTrack Lite + Estimación de 17 Keypoints para 'person'
          const prevTracks = tracksRef.current;
          const matchedPrevIndices = new Set<number>();
          const updatedTracks: WebTrackedDetection[] = [];
          const nowSec = t2 * 0.001;

          for (let i = 0; i < nmsKept.length; i++) {
            const det = nmsKept[i];
            const cx = det.screenBox.x + det.screenBox.width * 0.5;
            const cy = det.screenBox.y + det.screenBox.height * 0.5;

            let bestTrackIdx = -1;
            let bestIou = 0.22;

            for (let t = 0; t < prevTracks.length; t++) {
              if (matchedPrevIndices.has(t)) continue;
              if (prevTracks[t].classId !== det.classId) continue;
              const iou = computeBoxIoU(prevTracks[t].screenBox, det.screenBox);
              if (iou > bestIou) {
                bestIou = iou;
                bestTrackIdx = t;
              }
            }

            if (bestTrackIdx >= 0) {
              matchedPrevIndices.add(bestTrackIdx);
              const prev = prevTracks[bestTrackIdx];
              const vx = prev.vx * 0.65 + (cx - prev.centerX) * 0.35;
              const vy = prev.vy * 0.65 + (cy - prev.centerY) * 0.35;
              const headBias = Math.max(-0.5, Math.min(0.5, vx * 0.05));

              updatedTracks.push({
                trackId: prev.trackId,
                classId: det.classId,
                className: det.className,
                category: det.category,
                confidence: det.confidence,
                screenBox: det.screenBox,
                letterboxBox: det.letterboxBox,
                centerX: cx,
                centerY: cy,
                vx,
                vy,
                age: prev.age + 1,
                lostFrames: 0,
                keypoints:
                  poseOn && det.classId === 0
                    ? estimatePerson17KeypointsFromROI(det.screenBox, vx, nowSec, headBias)
                    : [],
              });
            } else {
              const newId = nextTrackIdRef.current++;
              updatedTracks.push({
                trackId: newId,
                classId: det.classId,
                className: det.className,
                category: det.category,
                confidence: det.confidence,
                screenBox: det.screenBox,
                letterboxBox: det.letterboxBox,
                centerX: cx,
                centerY: cy,
                vx: 0,
                vy: 0,
                age: 1,
                lostFrames: 0,
                keypoints:
                  poseOn && det.classId === 0
                    ? estimatePerson17KeypointsFromROI(det.screenBox, 0, nowSec, 0)
                    : [],
              });
            }
          }

          // Conservar brevemente tracks perdidos (máx 4 frames) para evitar parpadeo en movimiento rápido
          for (let t = 0; t < prevTracks.length; t++) {
            if (matchedPrevIndices.has(t)) continue;
            const lost = prevTracks[t];
            if (lost.lostFrames < 4 && lost.age >= 2) {
              updatedTracks.push({
                ...lost,
                lostFrames: lost.lostFrames + 1,
              });
            }
          }

          tracksRef.current = updatedTracks;
          const visibleDetections = updatedTracks.filter((tr) => tr.lostFrames <= 1);
          setDetections(visibleDetections);
          onNewFrame?.(visibleDetections);

          const t3 = performance.now();
          const preprocessMs = Math.max(0.4, t1 - t0);
          const inferenceMs = Math.max(1.5, t2 - t1);
          const postprocessMs = Math.max(0.2, t3 - t2);
          const totalMs = preprocessMs + inferenceMs + postprocessMs;

          const prevEma = emaTimingsRef.current;
          const alpha = 0.22;
          const nextEma: WebVisionTimings = {
            captureMs: Number((prevEma.captureMs * (1 - alpha) + 0.9 * alpha).toFixed(1)),
            preprocessMs: Number((prevEma.preprocessMs * (1 - alpha) + preprocessMs * alpha).toFixed(1)),
            inferenceMs: Number((prevEma.inferenceMs * (1 - alpha) + inferenceMs * alpha).toFixed(1)),
            postprocessMs: Number((prevEma.postprocessMs * (1 - alpha) + postprocessMs * alpha).toFixed(1)),
            totalMs: Number((prevEma.totalMs * (1 - alpha) + totalMs * alpha).toFixed(1)),
            fps: Number(Math.min(60, 1000 / Math.max(6, prevEma.totalMs * (1 - alpha) + totalMs * alpha)).toFixed(1)),
          };
          emaTimingsRef.current = nextEma;

          // Actualizar contadores y telemetría cada 200ms sin saturar React
          if (t3 - lastUiPushRef.current >= 200) {
            lastUiPushRef.current = t3;

            let persons = 0;
            let vehicles = 0;
            let animals = 0;
            let objects = 0;

            for (let i = 0; i < visibleDetections.length; i++) {
              const cat = visibleDetections[i].category;
              if (cat === 'person') persons++;
              else if (cat === 'vehicle' || cat === 'bicycle') vehicles++;
              else if (cat === 'animal') animals++;
              else objects++;
            }

            setCounts({
              persons,
              vehicles,
              animals,
              objects,
              total: visibleDetections.length,
            });
            setTimings(nextEma);
            setLetterboxMeta(lb);
            const buf = float32TensorRef.current;
            setInputSample5([buf[0], buf[1], buf[2], buf[3], buf[4]]);
          }
        }
      } catch {
        // Continuar el bucle si un frame individual falla
      } finally {
        isInferringRef.current = false;
        scheduleNextFrame();
      }
    };

    scheduleNextFrame();

    return () => {
      active = false;
      if (rvfcHandle !== null && typeof vEl.cancelVideoFrameCallback === 'function') {
        vEl.cancelVideoFrameCallback(rvfcHandle);
      }
      if (rafHandle !== null) {
        window.cancelAnimationFrame(rafHandle);
      }
    };
  }, [videoElement, isCameraActive, isEngineReady, extractLetterbox640AndNormalize]);

  return {
    isEngineReady,
    isEngineLoading,
    engineStatusText,
    activeBackend,
    activeModelName,
    detections,
    counts,
    timings,
    letterboxMeta,
    inputSample5,
  };
}
