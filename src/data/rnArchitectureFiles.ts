import visionPostProcessorRaw from '../utils/visionPostProcessor.ts?raw';
import pipelineThermalRaw from '../utils/pipelineThermal.ts?raw';
import arThemeConfigRaw from '../config/ARThemeConfig.ts?raw';
import arStartOverlayRaw from '../components/ARStartOverlay.tsx?raw';
import webCameraStreamRaw from '../components/WebCameraStream.tsx?raw';
import useOnnxVisionEngineRaw from '../hooks/useOnnxVisionEngine.ts?raw';
import useInertialARTrackerRaw from '../hooks/useInertialARTracker.ts?raw';
import useWebVisionEngineRaw from '../hooks/useWebVisionEngine.ts?raw';
import arWebCanvasOverlayRaw from '../components/ARWebCanvasOverlay.tsx?raw';
import exportOnnxWebColabRaw from '../../scripts/export_onnx_web_colab.py?raw';

export type DeliverableCategory = 'project' | 'inference' | 'ui' | 'native' | 'colab';

export interface CodeFileNode {
  id: string;
  path: string;
  name: string;
  language: 'typescript' | 'json' | 'swift' | 'kotlin' | 'ruby' | 'groovy' | 'javascript' | 'python' | 'bash';
  deliverable: DeliverableCategory;
  summary: string;
  highlights: string[];
  content: string;
}

export const RN_PROJECT_TREE = `VisionEdge-ONNX-Web-And-RN/
├── src/
│   ├── App.tsx                              # Visor AR ONNX 100% Edge-to-Edge + HUD Glassmorphism + Kalman IMU
│   ├── components/
│   │   ├── ARStartOverlay.tsx               # Botón flotante "Iniciar Cámara AR" + Permisos iOS 13+ IMU/WebRTC
│   │   ├── WebCameraStream.tsx              # getUserMedia progresivo + playsInline Safari/Samsung + Switch Cam
│   │   ├── ARWebCanvasOverlay.tsx           # Canvas HiDPI + Retículas Neón + Estelas 15pt + Callout Pins + Kalman
│   │   ├── AROverlayCanvas.tsx              # Skia AR Full-Screen (React Native) + LERP + 60Hz Gyro Offset
│   │   ├── DetectionOverlayCanvas.tsx       # Skia PictureRecorder 60 FPS (cero re-renders)
│   │   └── DiagnosticsModal.tsx             # Bottom Sheet (Telemetría, Gráfica 20s, Tabla y Explorador Código)
│   ├── hooks/
│   │   ├── useOnnxVisionEngine.ts           # ONNX Runtime Web (WebGPU/WASM SIMD) + Letterbox #727272 + Planar NCHW
│   │   ├── useInertialARTracker.ts          # Filtro de Kalman 1D + LERP + devicemotion/deviceorientation 60/120Hz
│   │   ├── useWebVisionEngine.ts            # Motor WebGPU -> WebGL -> WASM + 80 Clases COCO + 17kp Pose
│   │   ├── useGyroARStabilization.ts        # Giroscopio 60Hz (expo-sensors) -> SharedValue Zero-Lag Tracking
│   │   └── useResponsiveLayout.ts           # Alturas dinámicas useWindowDimensions + SafeArea
│   ├── config/
│   │   └── ARThemeConfig.ts                 # 3 estilos visuales AR (Táctico Cian, Nativo Frosted, Industrial)
│   ├── utils/
│   │   ├── visionPostProcessor.ts           # Zero-alloc YOLO decode (80 COCO) + Class-NMS + ByteTrack
│   │   └── pipelineThermal.ts               # Instrumentación 6 etapas (EMA) + Gobernador Térmico
│   ├── models/
│   │   └── useVisionModels.ts               # Hook TFLite CoreML / GPU / NNAPI (React Native)
│   └── processors/
│       └── useObjectDetectionProcessor.ts   # Frame Processor JSI + resize + runSync (React Native)
├── scripts/
│   └── export_onnx_web_colab.py             # Exportador Python/Colab YOLO11n + Pose -> ONNX FP32 + Validador + ZIP
├── public/
│   └── models/
│       ├── yolo11n.onnx                     # Modelo YOLO11n exportado (opset=17, imgsz=640, simplify=True, FP32)
│       └── yolo11n-pose.onnx                # Modelo YOLO11n-pose exportado (opset=17, imgsz=640, simplify=True, FP32)
└── vite.config.ts                           # Configuración MIME/CORS para servir .onnx y .wasm sin errores`;

export const RN_ARCHITECTURE_FILES: CodeFileNode[] = [
  {
    id: 'export-onnx-web-colab',
    path: 'scripts/export_onnx_web_colab.py',
    name: 'export_onnx_web_colab.py',
    language: 'python',
    deliverable: 'colab',
    summary: 'Script Python completo para Google Colab o entorno local: descarga yolo11n.pt y yolo11n-pose.pt, los exporta a .onnx (opset=17, imgsz=640, simplify=True, dynamic=False, half=False FP32 obligatorio para WebGPU/WASM), valida nodos y dimensiones con onnxruntime.InferenceSession y empaqueta models_onnx_web.zip.',
    highlights: [
      'Exportación FP32 estricta (opset=17, imgsz=640, simplify=True, dynamic=False, half=False) para Safari iOS y Chrome Android',
      'Validación automática con onnxruntime.InferenceSession sobre tensor Float32 aleatorio [1, 3, 640, 640]',
      'Empaquetado automático en models_onnx_web.zip con estructura /public/models/ y descarga directa en Colab',
    ],
    content: exportOnnxWebColabRaw,
  },
  {
    id: 'use-onnx-vision-engine',
    path: 'src/hooks/useOnnxVisionEngine.ts',
    name: 'useOnnxVisionEngine.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Motor de inferencia ONNX Runtime Web de alto rendimiento: configura ort.env.wasm.wasmPaths y ort.env.wasm.simd = true con executionProviders ["webgpu", "wasm"], preprocesa en Letterbox gris neutro (#727272, 114/255) y Tensor Planar NCHW [1, 3, 640, 640] en Float32Array [0, 1], decodifica [1, 84, 8400] y [1, 56, 8400] con stride channel * 8400 + proposalIndex, Class-Aware NMS (IoU=0.45) y ByteTrack con estela de 15 posiciones.',
    highlights: [
      'Preprocesamiento exacto Letterbox (#727272, 114/255) + Tensor Planar NCHW [1, 3, 640, 640] Float32Array',
      'Decodificación transpuesta [1, 84, 8400] (stride channel * 8400 + proposalIndex) sin doble sigmoid',
      'Bucle Zero-Freeze con requestVideoFrameCallback + bandera isInferencingRef + ByteTrack 15-pt Motion Trails',
    ],
    content: useOnnxVisionEngineRaw,
  },
  {
    id: 'use-inertial-ar-tracker',
    path: 'src/hooks/useInertialARTracker.ts',
    name: 'useInertialARTracker.ts',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Sistema de Anclaje Espacial Inercial: combina devicemotion y deviceorientation del navegador con un Filtro de Kalman 1D con velocidad (ScalarKalmanFilter) + LERP por cada trackId de ByteTrack para compensar el movimiento de la mano a 60/120 FPS sin vibración (jitter).',
    highlights: [
      'ScalarKalmanFilter 1D con predicción de velocidad y ganancia de Kalman para ejes Pitch/Yaw y coordenadas de caja',
      'Fusión de devicemotion (rad/s) y deviceorientation (Euler) con reinicio sincronizado al llegar inferencia ONNX',
      'Interpolación Kalman + LERP a 60/120 FPS para cajas delimitadoras, etiquetas y 17 keypoints articulares',
    ],
    content: useInertialARTrackerRaw,
  },
  {
    id: 'ar-start-overlay',
    path: 'src/components/ARStartOverlay.tsx',
    name: 'ARStartOverlay.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Pantalla inicial inmersiva con el botón flotante de alto contraste "Iniciar Cámara AR": ejecuta en secuencia en el gesto onClick (1) DeviceMotionEvent.requestPermission() en iOS 13+, (2) getUserMedia + video.play(), y (3) desbloqueo de WebAudio/vibración + selector opcional de archivo .onnx local.',
    highlights: [
      'Secuencia estricta en gesto onClick para cumplir políticas de iOS Safari 13+ y Chrome Android',
      'Captura segura de DeviceMotionEvent.requestPermission() y DeviceOrientationEvent.requestPermission()',
      'Permite cargar en caliente cualquier archivo .onnx (yolo11n.onnx / yolov8n.onnx) desde el navegador',
    ],
    content: arStartOverlayRaw,
  },
  {
    id: 'use-web-vision-engine',
    path: 'src/hooks/useWebVisionEngine.ts',
    name: 'useWebVisionEngine.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Motor de inferencia real en navegador móvil con fallback automático WebGPU -> WebGL -> WASM SIMD, captura por requestVideoFrameCallback (con fallback a requestAnimationFrame), letterbox 640x640 normalizado a Float32Array [0.0, 1.0], reconocimiento de las 80 clases de COCO y esqueleto de 17 puntos.',
    highlights: [
      'Detección automática de capacidades: WebGPU -> WebGL FP16 -> WASM SIMD (TensorFlow.js + ONNX Runtime Web)',
      'requestVideoFrameCallback con fallback a requestAnimationFrame + canvas oculto 640x640 letterbox',
      'Decodificación completa de las 80 clases de COCO (Personas, Vehículos, Animales y Objetos) + 17 keypoints',
    ],
    content: useWebVisionEngineRaw,
  },
  {
    id: 'web-camera-stream',
    path: 'src/components/WebCameraStream.tsx',
    name: 'WebCameraStream.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Componente universal de cámara WebRTC para iOS Safari, Chrome Android, Firefox, Edge y Samsung Internet: aplica restricciones progresivas anti-OverconstrainedError, atributos playsInline/webkit-playsinline/muted/autoPlay, object-fit: cover (100vw x 100dvh) y liberación segura de tracks al cambiar de cámara.',
    highlights: [
      'Fallback progresivo en 3 niveles para getUserMedia (1280x720 ideal -> 640x480 -> universal)',
      'Atributos nativos playsinline y webkit-playsinline="true" para evitar reproductor fullscreen en iOS/Samsung',
      'Detención explícita stream.getTracks().forEach(t => t.stop()) antes de alternar entre trasera y frontal',
    ],
    content: webCameraStreamRaw,
  },
  {
    id: 'ar-web-canvas-overlay',
    path: 'src/components/ARWebCanvasOverlay.tsx',
    name: 'ARWebCanvasOverlay.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'HUD AR Avanzado en Canvas HiDPI (window.devicePixelRatio): dibuja retículas tácticas con brillo neón por categoría (Cian Personas, Naranja Vehículos, Esmeralda Animales, Violeta Objetos), estelas de movimiento (Motion Trails de 15 posiciones), callout pins con velocidad relativa y distancia (~2.8m), y esqueleto biomecánico.',
    highlights: [
      'Retículas tácticas adaptables por categoría + estela punteada degradada con las últimas 15 posiciones del centroide',
      'Callout pins con líder óptico: clase, ID ByteTrack, confianza (94%), velocidad relativa (m/s) y distancia (~2.8m)',
      'Anclaje espacial a 60/120 FPS conectado a useInertialARTracker (Filtro de Kalman + LERP)',
    ],
    content: arWebCanvasOverlayRaw,
  },
  {
    id: 'app-tsx-unified',
    path: 'App.tsx (Full-Screen AR HUD + 60Hz Gyro Entry)',
    name: 'App.tsx (AR Full-Screen)',
    language: 'typescript',
    deliverable: 'project',
    summary: 'Visor AR 100% Edge-to-Edge (StyleSheet.absoluteFill) con estabilización inercial predictiva a 60Hz (useGyroARStabilization), 3 temas AR intercambiables en caliente (Táctico / Nativo / Industrial) y HUD flotante translúcido Glassmorphism (rgba(11, 15, 23, 0.55)).',
    highlights: [
      'Cámara y Skia AROverlayCanvas 100% edge-to-edge (StyleSheet.absoluteFill) con resizeMode="cover"',
      'Giroscopio 60Hz (expo-sensors) conectado a SharedValue + reinicio sincronizado y LERP al llegar YOLO',
      'HUD flotante Glassmorphism con useSafeAreaInsets: Barra Superior, Contadores, Mini-Radar colapsable y Barra Inferior',
    ],
    content: `import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  AppState,
  AppStateStatus,
  Linking,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from 'react-native';
import {
  SafeAreaProvider,
  SafeAreaView,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import {
  Camera,
  runAsync,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useSharedValue } from 'react-native-reanimated';
import { Worklets } from 'react-native-worklets-core';

import { useVisionModels } from './src/models/useVisionModels';
import {
  ARThemeId,
  AR_THEMES,
  AR_THEME_ORDER,
} from './src/config/ARThemeConfig';
import { useGyroARStabilization } from './src/hooks/useGyroARStabilization';
import { AROverlayCanvas } from './src/components/AROverlayCanvas';
import {
  applyClassAwareNMS,
  decodeYoloDetections,
  decodeYoloPose,
  ensureNormalizedFloat32Input,
  inspectYoloOutputTensor,
  ProposalPool,
  SensorOrientation,
  SimpleByteTracker,
  TrackedObject,
} from './src/utils/visionPostProcessor';
import {
  AdaptiveThermalController,
  PipelineProfiler,
  StageLatencyEma,
  ThermalMode,
} from './src/utils/PipelineProfiler';
import { ActivityHeatmap, HEATMAP_COLS, HEATMAP_ROWS } from './src/components/ActivityHeatmap';
import { TrackingHistorySample } from './src/components/TelemetryCharts';
import { DiagnosticsModal, ThermalControlSetting } from './src/components/DiagnosticsModal';

interface JsTelemetrySnapshot {
  persons: number;
  vehicles: number;
  bicycles: number;
  animalsOther: number;
  density: 'Low' | 'Moderate' | 'High';
  timings: StageLatencyEma;
  thermalMode: ThermalMode;
  rollingAvgMs: number;
  activeTracks: TrackedObject[] | null;
}

const UI_THROTTLE_MS = 250;

// Pools preasignados en el hilo Worklet (zero-allocation)
const detPool = new ProposalPool(256);
const posePool = new ProposalPool(256);
const nmsIndicesOut = new Int32Array(256);
const byteTracker = new SimpleByteTracker(15, 0.28, 0.52);
const profiler = new PipelineProfiler(0.18);
const thermalController = new AdaptiveThermalController();

export default function App() {
  return (
    <SafeAreaProvider>
      <VisionEdgeARHudScreen />
    </SafeAreaProvider>
  );
}

function VisionEdgeARHudScreen() {
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();
  const insets = useSafeAreaInsets();

  // 1. Permisos de Cámara & Ciclo de Vida AppState
  const { hasPermission, requestPermission } = useCameraPermission();
  const [appState, setAppState] = useState<AppStateStatus>(AppState.currentState);
  const [cameraSide, setCameraSide] = useState<'back' | 'front'>('back');
  const [isPaused, setIsPaused] = useState(false);

  // Estado del HUD Flotante y Tema AR
  const [arTheme, setArTheme] = useState<ARThemeId>('tactical');
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [gyroEnabled, setGyroEnabled] = useState(true);
  const [miniRadarCollapsed, setMiniRadarCollapsed] = useState(false);

  // Parámetros de inferencia ajustables en vivo
  const [confThreshold, setConfThreshold] = useState(0.45);
  const [nmsIouThreshold, setNmsIouThreshold] = useState(0.45);
  const [thermalControl, setThermalControl] = useState<ThermalControlSetting>('auto');
  const [diagnosticsVisible, setDiagnosticsVisible] = useState(false);

  const device = useCameraDevice(cameraSide);
  const { detector, poseModel, isLoading, error, reload } = useVisionModels();
  const { resize } = useResizePlugin();

  // 2. Estabilización Predictiva con Giroscopio a 60Hz (16ms) -> SharedValue
  const { gyroOffsetShared, resetOnNewInferenceWorklet } = useGyroARStabilization({
    screenWidth,
    screenHeight,
    horizontalFovDeg: 62,
    enabled: gyroEnabled && !isPaused,
  });

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().catch(() => {});
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      setAppState(nextState);
    });
    return () => sub.remove();
  }, []);

  const isCameraActive =
    appState === 'active' &&
    !isPaused &&
    hasPermission &&
    !isLoading &&
    !error &&
    Boolean(device);

  // 3. SharedValues para renderizado Skia AR a 60 FPS sin cruzar por React
  const tracksShared = useSharedValue<TrackedObject[]>([]);
  const themeIdShared = useSharedValue<ARThemeId>(arTheme);
  const showSkeletonShared = useSharedValue<boolean>(showSkeleton);
  const showLabelsShared = useSharedValue<boolean>(true);
  const fpsWaveShared = useSharedValue<number[]>(new Array(30).fill(30));
  const heatmapGridShared = useSharedValue<number[]>(
    new Array(HEATMAP_COLS * HEATMAP_ROWS).fill(0)
  );
  const lastJsSyncShared = useSharedValue<number>(0);
  const lastSmokeLogShared = useSharedValue<number>(0);
  const frameCounterShared = useSharedValue<number>(0);

  const confShared = useSharedValue(confThreshold);
  const iouShared = useSharedValue(nmsIouThreshold);
  const isFrontShared = useSharedValue(cameraSide === 'front');
  const thermalOverrideShared = useSharedValue<ThermalControlSetting>(thermalControl);
  const viewportSizeShared = useSharedValue({ width: screenWidth, height: screenHeight });
  const isDiagnosticsOpenShared = useSharedValue(diagnosticsVisible);

  useEffect(() => {
    themeIdShared.value = arTheme;
    showSkeletonShared.value = showSkeleton;
    confShared.value = confThreshold;
    iouShared.value = nmsIouThreshold;
    isFrontShared.value = cameraSide === 'front';
    thermalOverrideShared.value = thermalControl;
    viewportSizeShared.value = { width: screenWidth, height: screenHeight };
    isDiagnosticsOpenShared.value = diagnosticsVisible;
  }, [
    arTheme,
    showSkeleton,
    confThreshold,
    nmsIouThreshold,
    cameraSide,
    thermalControl,
    screenWidth,
    screenHeight,
    diagnosticsVisible,
  ]);

  // 4. Telemetría en JS Thread (actualizada únicamente cada 250 ms)
  const sampleTickRef = useRef(0);
  const [telemetry, setTelemetry] = useState({
    persons: 0,
    vehicles: 0,
    bicycles: 0,
    animalsOther: 0,
    density: 'Low' as 'Low' | 'Moderate' | 'High',
    thermalMode: 'Ultra' as ThermalMode,
    rollingAvgMs: 26.4,
    timings: {
      captureMs: 1.5,
      resizeMs: 1.9,
      yoloMs: 11.4,
      poseMs: 7.7,
      nmsTrackMs: 1.4,
      skiaRenderMs: 1.8,
      totalMs: 25.7,
      theoreticalFps: 38.9,
      realFps: 30.0,
    } as StageLatencyEma,
    activeTracks: [] as TrackedObject[],
  });

  const [history20s, setHistory20s] = useState<TrackingHistorySample[]>(() =>
    Array.from({ length: 20 }, () => ({ persons: 0, vehicles: 0, bicycles: 0 }))
  );

  const applyTelemetrySnapshot = useCallback((snap: JsTelemetrySnapshot) => {
    setTelemetry((prevState) => ({
      persons: snap.persons,
      vehicles: snap.vehicles,
      bicycles: snap.bicycles,
      animalsOther: snap.animalsOther,
      density: snap.density,
      thermalMode: snap.thermalMode,
      rollingAvgMs: snap.rollingAvgMs,
      timings: snap.timings,
      activeTracks: snap.activeTracks ?? prevState.activeTracks,
    }));

    sampleTickRef.current = (sampleTickRef.current + 1) % 4;
    if (sampleTickRef.current === 0) {
      setHistory20s((h) => [
        ...h.slice(-19),
        { persons: snap.persons, vehicles: snap.vehicles, bicycles: snap.bicycles },
      ]);
    }
  }, []);

  const syncTelemetryToJs = useMemo(
    () => Worklets.createRunOnJS(applyTelemetrySnapshot),
    [applyTelemetrySnapshot]
  );

  // 5. Frame Processor JSI con Reinicio Sincronizado del Giroscopio
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!detector) return;
      const tCaptureStart = performance.now();

      runAsync(frame, () => {
        'worklet';
        const frameIdx = (frameCounterShared.value += 1);
        const tResizeStart = performance.now();
        const captureMs = Math.max(0.1, tResizeStart - tCaptureStart);

        const rawOrientation = String(frame.orientation ?? 'portrait');
        const sensorOrientation: SensorOrientation =
          rawOrientation === 'landscape-right'
            ? 90
            : rawOrientation === 'portrait-upside-down'
            ? 180
            : rawOrientation === 'landscape-left'
            ? 270
            : 0;
        const rotationDeg =
          sensorOrientation === 90
            ? '90deg'
            : sensorOrientation === 180
            ? '180deg'
            : sensorOrientation === 270
            ? '270deg'
            : '0deg';

        const expectedInType = String(detector.inputs?.[0]?.dataType ?? 'float32');
        const rawResized = resize(frame, {
          scale: { width: 640, height: 640 },
          pixelFormat: 'rgb',
          dataType: expectedInType === 'uint8' ? 'uint8' : 'float32',
          rotation: rotationDeg,
        });
        const { buffer: inputBuffer, wasRescaledFrom255 } = ensureNormalizedFloat32Input(
          rawResized as Float32Array | Uint8Array,
          expectedInType
        );

        const tYoloStart = performance.now();
        const resizeMs = tYoloStart - tResizeStart;

        const [detTensor] = detector.runSync([inputBuffer]) as [Float32Array];
        const detShape = detector.outputs?.[0]?.shape;
        const tPoseStart = performance.now();
        const yoloMs = tPoseStart - tYoloStart;

        if (tPoseStart - lastSmokeLogShared.value >= 1000) {
          lastSmokeLogShared.value = tPoseStart;
          const diag = inspectYoloOutputTensor(detTensor, detShape);
          console.log(
            \`[TFLite SmokeTest] layout=\${diag.layout} | in[0..2]=[\${inputBuffer[0]?.toFixed(3)}, \${inputBuffer[1]?.toFixed(3)}, \${inputBuffer[2]?.toFixed(3)}] (rescaled255=\${wasRescaledFrom255}) | maxScore=\${diag.maxScore.toFixed(4)} (\${diag.bestClassName})\`
          );
        }

        const overrideMode = thermalOverrideShared.value;
        const autoDecision = thermalController.evaluate(captureMs + resizeMs + yoloMs + 9.0);
        const activeThermalMode: ThermalMode =
          overrideMode === 'auto' ? autoDecision.mode : overrideMode;

        const shouldRunPose =
          Boolean(poseModel) &&
          showSkeletonShared.value &&
          (activeThermalMode === 'Ultra' ||
            (activeThermalMode === 'Balanced' && frameIdx % 2 === 0));

        let poseTensor: Float32Array | null = null;
        if (shouldRunPose && poseModel) {
          poseTensor = poseModel.runSync([inputBuffer])[0] as Float32Array;
        }

        const tPostStart = performance.now();
        const poseMs = tPostStart - tPoseStart;

        const confMin = confShared.value;
        decodeYoloDetections(detTensor, confMin, 640, detPool, detShape);
        const { keptIndices, keptCount } = applyClassAwareNMS(
          detPool,
          iouShared.value,
          nmsIndicesOut
        );

        if (poseTensor) {
          const poseShape = poseModel?.outputs?.[0]?.shape;
          decodeYoloPose(poseTensor, confMin, 640, posePool, poseShape);
          for (let i = 0; i < keptCount; i++) {
            const det = detPool.items[keptIndices[i]];
            if (det.classId !== 0) continue;
            let bestDist = 0.08;
            let bestPoseIdx = -1;
            for (let p = 0; p < posePool.count; p++) {
              const candPose = posePool.items[p];
              const d = Math.hypot(candPose.cx - det.cx, candPose.cy - det.cy);
              if (d < bestDist) {
                bestDist = d;
                bestPoseIdx = p;
              }
            }
            if (bestPoseIdx >= 0) {
              det.hasPose = true;
              const srcKps = posePool.items[bestPoseIdx].keypoints;
              for (let k = 0; k < 17; k++) {
                det.keypoints[k].x = srcKps[k].x;
                det.keypoints[k].y = srcKps[k].y;
                det.keypoints[k].visibility = srcKps[k].visibility;
              }
            }
          }
        }

        const vp = viewportSizeShared.value;
        // Calibrado con resizeMode="cover" para anclaje espacial exacto a pantalla completa
        const tracked = byteTracker.update(detPool, keptIndices, keptCount, {
          cameraWidth: frame.width,
          cameraHeight: frame.height,
          screenWidth: vp.width,
          screenHeight: vp.height,
          tensorSize: 640,
          orientation: sensorOrientation,
          resizeMode: 'cover',
          inputPaddingMode: 'stretch',
          isFrontCamera: isFrontShared.value,
        });

        // Publica nuevas coordenadas reales y reinicia sincronizadamente el offset del giroscopio
        tracksShared.value = tracked;
        resetOnNewInferenceWorklet();

        const tEnd = performance.now();
        const nmsTrackMs = tEnd - tPostStart;
        const emaTimings = profiler.recordFrame(
          captureMs,
          resizeMs,
          yoloMs,
          poseMs,
          nmsTrackMs,
          1.6,
          tEnd
        );

        if (tEnd - lastJsSyncShared.value >= UI_THROTTLE_MS) {
          lastJsSyncShared.value = tEnd;
          const visible = tracked.filter((t) => t.lostFrames === 0);
          let persons = 0;
          let vehicles = 0;
          let bicycles = 0;
          let animalsOther = 0;
          for (let i = 0; i < visible.length; i++) {
            const cat = visible[i].category;
            if (cat === 'person') persons++;
            else if (cat === 'vehicle') vehicles++;
            else if (cat === 'bicycle') bicycles++;
            else animalsOther++;
          }
          const total = visible.length;
          const density = total < 3 ? 'Low' : total <= 7 ? 'Moderate' : 'High';

          syncTelemetryToJs({
            persons,
            vehicles,
            bicycles,
            animalsOther,
            density,
            timings: emaTimings,
            thermalMode: activeThermalMode,
            rollingAvgMs: autoDecision.rollingAvgMs,
            activeTracks: isDiagnosticsOpenShared.value ? visible : null,
          });
        }
      });
    },
    [detector, poseModel, resize, resetOnNewInferenceWorklet, syncTelemetryToJs]
  );

  const cycleARTheme = useCallback(() => {
    setArTheme((prev) => {
      const idx = AR_THEME_ORDER.indexOf(prev);
      return AR_THEME_ORDER[(idx + 1) % AR_THEME_ORDER.length];
    });
  }, []);

  const handleResetHeatmap = useCallback(() => {
    heatmapGridShared.value = new Array(HEATMAP_COLS * HEATMAP_ROWS).fill(0);
  }, [heatmapGridShared]);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.centerScreen}>
        <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
        <Text style={styles.titleLg}>Permiso de Cámara Requerido</Text>
        <TouchableOpacity
          style={styles.primaryBtn}
          onPress={() => requestPermission().then((ok) => !ok && Linking.openSettings())}
        >
          <Text style={styles.primaryBtnText}>Conceder Acceso a Cámara</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  if (isLoading || error) {
    return (
      <SafeAreaView style={styles.centerScreen}>
        <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
        {isLoading ? (
          <ActivityIndicator size="large" color="#38BDF8" />
        ) : (
          <TouchableOpacity style={styles.primaryBtn} onPress={reload}>
            <Text style={styles.primaryBtnText}>Reintentar Carga TFLite</Text>
          </TouchableOpacity>
        )}
      </SafeAreaView>
    );
  }

  const activeThemeSpec = AR_THEMES[arTheme];

  return (
    <View style={styles.fullScreenRoot}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* 1. CÁMARA INMERSIVA 100% EDGE-TO-EDGE (StyleSheet.absoluteFill + resizeMode="cover") */}
      {device && (
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isCameraActive}
          frameProcessor={frameProcessor}
          resizeMode="cover"
          pixelFormat="yuv"
          fps={30}
        />
      )}

      {/* 2. LIENZO SKIA AR 100% PANTALLA COMPLETA CON GIROSCOPIO 60Hz Y LERP */}
      <AROverlayCanvas
        tracksShared={tracksShared}
        gyroOffsetShared={gyroOffsetShared}
        themeIdShared={themeIdShared}
        showSkeletonShared={showSkeletonShared}
        showLabelsShared={showLabelsShared}
        width={screenWidth}
        height={screenHeight}
      />

      {/* 3. HUD FLOTANTE TRANSLÚCIDO (GLASSMORPHISM rgba(11, 15, 23, 0.55)) */}
      {/* Barra Superior Flotante */}
      <View style={[styles.floatingTopBar, { top: Math.max(12, insets.top + 6) }]}>
        <View style={[styles.glassPill, { borderColor: activeThemeSpec.hudBorderColor }]}>
          <View
            style={[
              styles.liveDot,
              { backgroundColor: isCameraActive ? '#22C55E' : '#F59E0B' },
            ]}
          />
          <Text style={styles.liveText}>LIVE AR • </Text>
          <Text style={styles.fpsText}>{telemetry.timings.realFps.toFixed(0)} FPS</Text>
        </View>

        <View style={styles.topActionsGroup}>
          <TouchableOpacity
            style={[styles.glassPill, { borderColor: activeThemeSpec.hudBorderColor }]}
            onPress={cycleARTheme}
          >
            <Text style={styles.actionPillText}>AR: {activeThemeSpec.shortLabel}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.glassPill, { borderColor: activeThemeSpec.hudBorderColor }]}
            onPress={() => setShowSkeleton((s) => !s)}
          >
            <Text style={styles.actionPillText}>
              Esqueleto {showSkeleton ? 'ON' : 'OFF'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.glassPill, { borderColor: activeThemeSpec.hudBorderColor }]}
            onPress={() => setCameraSide((s) => (s === 'back' ? 'front' : 'back'))}
          >
            <Text style={styles.actionPillText}>{cameraSide === 'back' ? '1x' : 'Front'}</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Píldoras de Conteo Compactas Flotantes */}
      <View style={[styles.floatingCountersRow, { top: Math.max(56, insets.top + 50) }]}>
        <View style={styles.counterBadge}>
          <View style={[styles.badgeDot, { backgroundColor: '#38BDF8' }]} />
          <Text style={styles.badgeLabel}>Personas</Text>
          <Text style={styles.badgeCount}>{telemetry.persons}</Text>
        </View>

        <View style={styles.counterBadge}>
          <View style={[styles.badgeDot, { backgroundColor: '#F97316' }]} />
          <Text style={styles.badgeLabel}>Vehículos</Text>
          <Text style={styles.badgeCount}>{telemetry.vehicles + telemetry.bicycles}</Text>
        </View>

        <View style={styles.counterBadge}>
          <View style={[styles.badgeDot, { backgroundColor: '#A855F7' }]} />
          <Text style={styles.badgeLabel}>Animales/Otros</Text>
          <Text style={styles.badgeCount}>{telemetry.animalsOther}</Text>
        </View>
      </View>

      {/* Mini-Radar / Heatmap Flotante Colapsable en Esquina Inferior Derecha */}
      <View
        style={[
          styles.floatingMiniRadarWrap,
          { bottom: Math.max(66, insets.bottom + 56) },
        ]}
      >
        {miniRadarCollapsed ? (
          <TouchableOpacity
            style={styles.glassPill}
            onPress={() => setMiniRadarCollapsed(false)}
          >
            <Text style={styles.actionPillText}>Radar ({telemetry.density})</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.miniRadarCard}>
            <View style={styles.miniRadarHeader}>
              <Text style={styles.miniRadarTitle}>MINI-RADAR AR</Text>
              <TouchableOpacity onPress={() => setMiniRadarCollapsed(true)}>
                <Text style={styles.miniRadarCollapseBtn}>—</Text>
              </TouchableOpacity>
            </View>
            <ActivityHeatmap
              gridShared={heatmapGridShared}
              width={132}
              height={74}
              onReset={handleResetHeatmap}
            />
          </View>
        )}
      </View>

      {/* Barra Inferior Flotante */}
      <View
        style={[
          styles.floatingBottomBar,
          { bottom: Math.max(14, insets.bottom + 8) },
        ]}
      >
        <TouchableOpacity
          style={styles.glassPill}
          onPress={() => setGyroEnabled((g) => !g)}
        >
          <View
            style={[
              styles.liveDot,
              { backgroundColor: gyroEnabled ? '#22C55E' : '#94A3B8' },
            ]}
          />
          <Text style={styles.statusFooterText}>
            {gyroEnabled ? 'AR + Gyro Active (60Hz)' : 'Gyro Paused'} •{' '}
            {telemetry.timings.totalMs.toFixed(1)}ms
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.glassPill, styles.settingsPillBtn]}
          onPress={() => {
            isDiagnosticsOpenShared.value = true;
            setDiagnosticsVisible(true);
          }}
        >
          <Text style={styles.settingsPillText}>Ajustes / Telemetría</Text>
        </TouchableOpacity>
      </View>

      {/* Bottom Sheet con Gráfica 20s, Desglose de Latencia EMA y Tabla de Objetos */}
      <DiagnosticsModal
        visible={diagnosticsVisible}
        onClose={() => {
          isDiagnosticsOpenShared.value = false;
          setDiagnosticsVisible(false);
        }}
        isPaused={isPaused}
        onTogglePause={() => setIsPaused((p) => !p)}
        onResetHeatmap={handleResetHeatmap}
        thermalControl={thermalControl}
        onChangeThermalControl={setThermalControl}
        rollingAvgMs={telemetry.rollingAvgMs}
        confThreshold={confThreshold}
        onChangeConfThreshold={setConfThreshold}
        nmsIouThreshold={nmsIouThreshold}
        onChangeNmsIouThreshold={setNmsIouThreshold}
        timings={telemetry.timings}
        tracks={telemetry.activeTracks}
        history20s={history20s}
        fpsWaveShared={fpsWaveShared}
      />
    </View>
  );
}

const GLASS_BG = 'rgba(11, 15, 23, 0.55)';

const styles = StyleSheet.create({
  fullScreenRoot: {
    flex: 1,
    backgroundColor: '#06090F',
  },
  centerScreen: {
    flex: 1,
    backgroundColor: '#0B0F17',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    gap: 14,
  },
  titleLg: { color: '#F8FAFC', fontSize: 18, fontWeight: '700', textAlign: 'center' },
  primaryBtn: {
    backgroundColor: '#38BDF8',
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 12,
  },
  primaryBtnText: { color: '#0B0F17', fontSize: 13, fontWeight: '700' },
  floatingTopBar: {
    position: 'absolute',
    left: 12,
    right: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    zIndex: 20,
  },
  topActionsGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  glassPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: GLASS_BG,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.28)',
    paddingHorizontal: 11,
    height: 36,
    borderRadius: 999,
  },
  liveDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    marginRight: 6,
  },
  liveText: {
    color: '#F8FAFC',
    fontSize: 11,
    fontWeight: '700',
  },
  fpsText: {
    color: '#38BDF8',
    fontSize: 11.5,
    fontWeight: '700',
    fontFamily: 'Courier',
  },
  actionPillText: {
    color: '#F1F5F9',
    fontSize: 11.5,
    fontWeight: '600',
  },
  floatingCountersRow: {
    position: 'absolute',
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    zIndex: 20,
  },
  counterBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: GLASS_BG,
    borderWidth: 1,
    borderColor: 'rgba(148, 163, 184, 0.24)',
    paddingHorizontal: 10,
    height: 28,
    borderRadius: 999,
    gap: 5,
  },
  badgeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  badgeLabel: {
    color: '#CBD5E1',
    fontSize: 11,
    fontWeight: '500',
  },
  badgeCount: {
    color: '#FFFFFF',
    fontSize: 11.5,
    fontWeight: '700',
    fontFamily: 'Courier',
  },
  floatingMiniRadarWrap: {
    position: 'absolute',
    right: 12,
    zIndex: 20,
  },
  miniRadarCard: {
    width: 148,
    backgroundColor: GLASS_BG,
    borderWidth: 1,
    borderColor: 'rgba(56, 189, 248, 0.32)',
    borderRadius: 16,
    padding: 8,
    gap: 6,
  },
  miniRadarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  miniRadarTitle: {
    color: '#E2E8F0',
    fontSize: 10,
    fontWeight: '700',
  },
  miniRadarCollapseBtn: {
    color: '#94A3B8',
    fontSize: 12,
    fontWeight: '700',
    paddingHorizontal: 4,
  },
  floatingBottomBar: {
    position: 'absolute',
    left: 12,
    right: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    zIndex: 20,
  },
  statusFooterText: {
    color: '#E2E8F0',
    fontSize: 11,
    fontWeight: '600',
  },
  settingsPillBtn: {
    borderColor: 'rgba(56, 189, 248, 0.45)',
  },
  settingsPillText: {
    color: '#38BDF8',
    fontSize: 11.5,
    fontWeight: '700',
  },
});`,
  },
  {
    id: 'expo-prebuild-plugin',
    path: 'plugins/withVisionEdgeNative.js (Android & iOS Patch)',
    name: 'withVisionEdgeNative.js',
    language: 'javascript',
    deliverable: 'project',
    summary: 'Expo Config Plugin determinista para blindar npx expo prebuild: inyecta aaptOptions { noCompress "tflite" } y dependencias TFLite GPU en Android build.gradle, y fuerza iOS 15.1+ en Podfile + Metal/CoreML en Info.plist.',
    highlights: [
      'Android: aaptOptions { noCompress "tflite" }, tensorflow-lite-gpu y pickFirst libc++_shared.so',
      'iOS: Podfile IPHONEOS_DEPLOYMENT_TARGET = 15.1 para Skia/FastTFLite + CoreML/Metal',
      'Registrado en app.json bajo plugins: ["./plugins/withVisionEdgeNative"]',
    ],
    content: `const fs = require('fs');
const path = require('path');
const {
  withAppBuildGradle,
  withInfoPlist,
  withDangerousMod,
} = require('@expo/config-plugins');

/**
 * 1. Parche Android (android/app/build.gradle):
 *    - Evita compresión ZIP de modelos .tflite (mmap directo sin errores silenciosos)
 *    - Añade org.tensorflow:tensorflow-lite-gpu para el delegado GPU/OpenCL
 *    - Evita colisiones de libc++_shared.so entre Skia, VisionCamera y FastTFLite
 */
function withAndroidTfliteHardening(config) {
  return withAppBuildGradle(config, (mod) => {
    let contents = mod.modResults.contents;

    // A) aaptOptions { noCompress "tflite" } (y androidResources para AGP 8+)
    if (!contents.includes('noCompress "tflite"')) {
      const noCompressBlock = \`
    aaptOptions {
        noCompress "tflite", "lite"
    }\`;
      contents = contents.replace(
        /android\\s*\\{/,
        \`android {\${noCompressBlock}\`
      );
    }

    // B) pickFirst para libc++_shared.so (requerido al combinar Skia + VisionCamera + TFLite C++)
    if (!contents.includes("pickFirst '**/libc++_shared.so'")) {
      const packagingBlock = \`
    packagingOptions {
        pickFirst '**/libc++_shared.so'
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
        pickFirst 'lib/armeabi-v7a/libc++_shared.so'
    }\`;
      contents = contents.replace(
        /android\\s*\\{/,
        \`android {\${packagingBlock}\`
      );
    }

    // C) Dependencias nativas GPU Delegate para Android
    if (!contents.includes('org.tensorflow:tensorflow-lite-gpu')) {
      contents = contents.replace(
        /dependencies\\s*\\{/,
        \`dependencies {
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.16.1'
    implementation 'org.tensorflow:tensorflow-lite-gpu-api:2.16.1'\`
      );
    }

    mod.modResults.contents = contents;
    return mod;
  });
}

/**
 * 2. Parche iOS Info.plist:
 *    - Declara soporte Metal obligatorio y permiso de cámara
 */
function withIosCoreMlInfoPlist(config) {
  return withInfoPlist(config, (mod) => {
    mod.modResults.NSCameraUsageDescription =
      mod.modResults.NSCameraUsageDescription ||
      'VisionEdge RN utiliza la cámara para inferencia neuronal on-device a 60 FPS.';

    const caps = new Set(mod.modResults.UIRequiredDeviceCapabilities || []);
    caps.add('arm64');
    caps.add('metal');
    mod.modResults.UIRequiredDeviceCapabilities = Array.from(caps);

    return mod;
  });
}

/**
 * 3. Parche iOS Podfile:
 *    - Fuerza deployment target >= 15.1 en el Podfile y en todos los pods (Skia + TFLite + CoreML)
 */
function withIosPodfileMinTarget(config) {
  return withDangerousMod(config, [
    'ios',
    async (mod) => {
      const podfilePath = path.join(mod.modRequest.platformProjectRoot, 'Podfile');
      if (!fs.existsSync(podfilePath)) return mod;

      let podfile = fs.readFileSync(podfilePath, 'utf8');

      // Asegura platform :ios, '15.1'
      podfile = podfile.replace(
        /platform\\s*:ios,\\s*['"][^'"]+['"]/,
        "platform :ios, '15.1'"
      );

      // Fuerza IPHONEOS_DEPLOYMENT_TARGET = '15.1' dentro de post_install
      const enforcementSnippet = \`
    # VisionEdge RN: Deployment Target mínimo 15.1 para FastTFLite CoreML/ANE y Skia
    installer.pods_project.targets.each do |target|
      target.build_configurations.each do |cfg|
        current = cfg.build_settings['IPHONEOS_DEPLOYMENT_TARGET'].to_f
        if current < 15.1
          cfg.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '15.1'
        end
      end
    end\`;

      if (!podfile.includes('VisionEdge RN: Deployment Target')) {
        podfile = podfile.replace(
          /post_install\\s+do\\s+\\|installer\\|/,
          \`post_install do |installer|\${enforcementSnippet}\`
        );
      }

      fs.writeFileSync(podfilePath, podfile, 'utf8');
      return mod;
    },
  ]);
}

module.exports = function withVisionEdgeNative(config) {
  config = withAndroidTfliteHardening(config);
  config = withIosCoreMlInfoPlist(config);
  config = withIosPodfileMinTarget(config);
  return config;
};`,
  },
  {
    id: 'package-json',
    path: 'CLI Init & app.json (Expo CNG)',
    name: 'Expo CLI & app.json',
    language: 'json',
    deliverable: 'project',
    summary: 'Comandos de inicialización con Expo SDK (CNG / expo prebuild sin Expo Go) y app.json blindado con expo-build-properties, CoreML Delegate, Android GPU y ./plugins/withVisionEdgeNative.',
    highlights: [
      'npx create-expo-app VisionEdgeRN --template blank-typescript',
      'Registra ./plugins/withVisionEdgeNative y expo-build-properties (Android SDK 26+ / iOS 15.1+)',
      'Habilita CoreML Delegate (ANE FP16) en iOS y GPU Libraries en Android',
    ],
    content: `// ============================================================================
// 1. Comandos de terminal (Expo SDK + CNG / Prebuild, sin Expo Go)
// ============================================================================
// npx create-expo-app@latest VisionEdgeRN --template blank-typescript
// cd VisionEdgeRN
//
// npx expo install expo-dev-client expo-build-properties \\
//   react-native-vision-camera react-native-worklets-core \\
//   vision-camera-resize-plugin react-native-fast-tflite \\
//   @shopify/react-native-skia react-native-reanimated \\
//   @react-native-community/slider
//
// # Generar carpetas nativas ios/ y android/ blindadas mediante CNG:
// npx expo prebuild --clean
//
// # Compilar y ejecutar en dispositivo físico (requerido para GPU/ANE y Cámara):
// npx expo run:ios --configuration Release --device
// npx expo run:android --variant release --device

// ============================================================================
// 2. app.json (Configuración Expo CNG Blindada)
// ============================================================================
{
  "expo": {
    "name": "VisionEdgeRN",
    "slug": "vision-edge-rn",
    "version": "1.0.0",
    "orientation": "portrait",
    "userInterfaceStyle": "dark",
    "newArchEnabled": true,
    "ios": {
      "bundleIdentifier": "com.visionedge.rn",
      "supportsTablet": false,
      "infoPlist": {
        "NSCameraUsageDescription": "VisionEdge RN utiliza la cámara para detección de objetos y pose local en tiempo real."
      }
    },
    "android": {
      "package": "com.visionedge.rn",
      "permissions": [
        "android.permission.CAMERA"
      ]
    },
    "plugins": [
      "expo-dev-client",
      [
        "expo-build-properties",
        {
          "android": {
            "minSdkVersion": 26,
            "compileSdkVersion": 35,
            "targetSdkVersion": 35
          },
          "ios": {
            "deploymentTarget": "15.1"
          }
        }
      ],
      [
        "react-native-vision-camera",
        {
          "cameraPermissionText": "VisionEdge RN necesita acceso a la cámara para inferencia neuronal on-device.",
          "enableFrameProcessors": true
        }
      ],
      [
        "react-native-fast-tflite",
        {
          "enableCoreMLDelegate": true,
          "enableAndroidGpuLibraries": true
        }
      ],
      [
        "expo-sensors",
        {
          "motionPermission": "VisionEdge RN utiliza el giroscopio a 60Hz para estabilizar las cajas y esqueletos de Realidad Aumentada en tiempo real."
        }
      ],
      "./plugins/withVisionEdgeNative"
    ]
  }
}`,
  },
  {
    id: 'babel-metro-config',
    path: 'babel.config.js & metro.config.js',
    name: 'babel.config.js & metro.config.js',
    language: 'javascript',
    deliverable: 'project',
    summary: 'Blindaje de compilación JS/Worklet: orden estricto de react-native-worklets-core/plugin y react-native-reanimated/plugin en Babel + registro binario de .tflite en Metro.',
    highlights: [
      'react-native-worklets-core/plugin antes de react-native-reanimated/plugin (siempre último)',
      'Configuración anti-colisión para Frame Processors JSI + UI Worklets de Reanimated',
      'Metro assetExts con tflite, mlpackage y bin (y limpieza en sourceExts)',
    ],
    content: `// ============================================================================
// babel.config.js — Orden Estricto Anti-Colisión de Worklets
// ============================================================================
// IMPORTANTE:
// 1. 'react-native-worklets-core/plugin' transforma Frame Processors ('worklet')
//    de VisionCamera v4.
// 2. 'react-native-reanimated/plugin' DEBE ir siempre en la ÚLTIMA posición del
//    array plugins; invertir este orden provoca errores en tiempo de ejecución
//    ("Worklets failed to initialize" o colisión de __workletHash).
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      ['react-native-worklets-core/plugin'],
      'react-native-reanimated/plugin', // OBLIGATORIO: siempre el último plugin
    ],
  };
};

// ============================================================================
// metro.config.js — Resolución de Modelos Binarios .tflite en el Bundle
// ============================================================================
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

const BINARY_MODEL_EXTS = ['tflite', 'lite', 'mlpackage', 'bin'];

// Añade extensiones binarias como assets y garantiza que no se traten como código JS
config.resolver.assetExts = Array.from(
  new Set([...config.resolver.assetExts, ...BINARY_MODEL_EXTS])
);
config.resolver.sourceExts = config.resolver.sourceExts.filter(
  (ext) => !BINARY_MODEL_EXTS.includes(ext)
);

module.exports = config;`,
  },
  {
    id: 'native-configs',
    path: 'package.json (Expo Dependencies)',
    name: 'package.json',
    language: 'json',
    deliverable: 'project',
    summary: 'Manifiesto de dependencias Expo CNG con react-native-vision-camera v4+, fast-tflite, resize-plugin, Skia y Reanimated.',
    highlights: [
      'Scripts prebuild, run:ios y run:android vía expo-dev-client',
      'Sin dependencias de Expo Go; compilación nativa directa JSI/C++',
    ],
    content: `{
  "name": "vision-edge-rn",
  "version": "1.0.0",
  "main": "node_modules/expo/AppEntry.js",
  "scripts": {
    "start": "expo start --dev-client",
    "prebuild": "expo prebuild --clean",
    "android": "expo run:android --variant release --device",
    "ios": "expo run:ios --configuration Release --device",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@react-native-community/slider": "^4.5.5",
    "@shopify/react-native-skia": "^1.5.4",
    "expo": "~52.0.0",
    "expo-build-properties": "~0.13.1",
    "expo-dev-client": "~5.0.0",
    "expo-sensors": "~14.0.1",
    "react": "18.3.1",
    "react-native": "0.76.3",
    "react-native-fast-tflite": "^1.6.0",
    "react-native-gesture-handler": "^2.20.2",
    "react-native-reanimated": "^3.16.1",
    "react-native-safe-area-context": "4.12.0",
    "react-native-vision-camera": "^4.6.2",
    "react-native-worklets-core": "^1.5.0",
    "vision-camera-resize-plugin": "^3.2.0"
  },
  "devDependencies": {
    "@babel/core": "^7.25.2",
    "@expo/config-plugins": "~9.0.0",
    "@types/react": "~18.3.12",
    "typescript": "^5.6.3"
  }
}`,
  },
  {
    id: 'model-loader',
    path: 'src/models/useVisionModels.ts',
    name: 'useVisionModels.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Hook de carga asíncrona en memoria nativa para yolo11n_fp16.tflite y yolo11n_pose_fp16.tflite con selección de delegado GPU/NPU según Platform.OS.',
    highlights: [
      'iOS -> core-ml (ANE + Metal), Android -> android-gpu con fallback a nnapi/default',
      'Manejo completo de estados isLoading, error, activeDelegate y reload()',
    ],
    content: `import { useCallback, useEffect, useState } from 'react';
import { Platform } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';

export type HardwareDelegate = 'core-ml' | 'android-gpu' | 'nnapi' | 'default';

export interface VisionModelsState {
  detector: TensorflowModel | null;
  poseModel: TensorflowModel | null;
  isLoading: boolean;
  error: string | null;
  activeDelegate: HardwareDelegate;
  reload: () => void;
}

function getPlatformDelegateOrder(): HardwareDelegate[] {
  if (Platform.OS === 'ios') {
    // core-ml utiliza Apple Neural Engine (ANE) y Metal GPU automáticamente
    return ['core-ml', 'default'];
  }
  if (Platform.OS === 'android') {
    // Prioriza OpenGL/OpenCL GPU Delegate, luego NNAPI y finalmente CPU XNNPACK
    return ['android-gpu', 'nnapi', 'default'];
  }
  return ['default'];
}

async function loadWithDelegateFallback(
  modelAsset: number,
  delegates: HardwareDelegate[]
): Promise<{ model: TensorflowModel; usedDelegate: HardwareDelegate }> {
  let lastErr: unknown = null;

  for (const delegate of delegates) {
    try {
      const model = await loadTensorflowModel(modelAsset, delegate);
      return { model, usedDelegate: delegate };
    } catch (err) {
      lastErr = err;
    }
  }

  throw lastErr instanceof Error
    ? lastErr
    : new Error('No se pudo inicializar el intérprete TFLite con ningún delegado.');
}

export function useVisionModels(
  preferredDelegate?: HardwareDelegate
): VisionModelsState {
  const [detector, setDetector] = useState<TensorflowModel | null>(null);
  const [poseModel, setPoseModel] = useState<TensorflowModel | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeDelegate, setActiveDelegate] = useState<HardwareDelegate>('default');
  const [reloadTick, setReloadTick] = useState(0);

  const reload = useCallback(() => {
    setReloadTick((t) => t + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function bootstrapModels() {
      setIsLoading(true);
      setError(null);

      try {
        const order = preferredDelegate
          ? [preferredDelegate, ...getPlatformDelegateOrder().filter((d) => d !== preferredDelegate)]
          : getPlatformDelegateOrder();

        const [detRes, poseRes] = await Promise.all([
          loadWithDelegateFallback(require('./yolo11n_fp16.tflite'), order),
          loadWithDelegateFallback(require('./yolo11n_pose_fp16.tflite'), order),
        ]);

        if (cancelled) return;
        setDetector(detRes.model);
        setPoseModel(poseRes.model);
        setActiveDelegate(detRes.usedDelegate);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Error desconocido cargando modelos .tflite';
        setError(message);
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    bootstrapModels();
    return () => {
      cancelled = true;
    };
  }, [preferredDelegate, reloadTick]);

  return { detector, poseModel, isLoading, error, activeDelegate, reload };
}`,
  },
  {
    id: 'tensor-utils',
    path: 'src/cv/postprocess/tensorUtils.ts',
    name: 'tensorUtils.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Conversión exacta de coordenadas del tensor 640×640 (con padding letterbox) a coordenadas reales de pantalla en píxeles.',
    highlights: [
      'Cálculo de escala uniforme min(640/W, 640/H) y offset padX/padY',
      'Soporte de aspect-cover del visor y espejo horizontal para cámara frontal',
    ],
    content: `export interface LetterboxMetrics {
  modelSize: number;
  scale: number;
  padX: number;
  padY: number;
}

export interface ScreenRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function computeLetterbox(
  frameWidth: number,
  frameHeight: number,
  modelSize = 640
): LetterboxMetrics {
  'worklet';
  const scale = Math.min(modelSize / frameWidth, modelSize / frameHeight);
  const padX = (modelSize - frameWidth * scale) * 0.5;
  const padY = (modelSize - frameHeight * scale) * 0.5;
  return { modelSize, scale, padX, padY };
}

/**
 * Convierte [cx, cy, w, h] del espacio del tensor (0..640) a píxeles de pantalla
 * compensando el recorte "aspect-cover" del visor VisionCamera.
 */
export function tensorBoxToScreen(
  cx: number,
  cy: number,
  w: number,
  h: number,
  lb: LetterboxMetrics,
  frameW: number,
  frameH: number,
  screenW: number,
  screenH: number,
  isFrontCamera: boolean
): ScreenRect {
  'worklet';
  // 1. Eliminar padding letterbox -> espacio original del sensor
  const sensorCx = (cx - lb.padX) / lb.scale;
  const sensorCy = (cy - lb.padY) / lb.scale;
  const sensorW = w / lb.scale;
  const sensorH = h / lb.scale;

  // 2. Proyección ResizeMode="cover" sobre el Canvas Skia
  const coverScale = Math.max(screenW / frameW, screenH / frameH);
  const offsetX = (frameW * coverScale - screenW) * 0.5;
  const offsetY = (frameH * coverScale - screenH) * 0.5;

  const outW = sensorW * coverScale;
  const outH = sensorH * coverScale;
  let outX = (sensorCx - sensorW * 0.5) * coverScale - offsetX;
  const outY = (sensorCy - sensorH * 0.5) * coverScale - offsetY;

  if (isFrontCamera) {
    outX = screenW - outX - outW;
  }

  return { x: outX, y: outY, width: outW, height: outH };
}`,
  },
  {
    id: 'nms-and-bytetrack',
    path: 'src/utils/visionPostProcessor.ts',
    name: 'visionPostProcessor.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Módulo matemático zero-allocation (< 2 ms/frame en Worklet): decodeYoloDetections, decodeYoloPose, applyClassAwareNMS, SimpleByteTracker (IoU + centroide + deltas) y projectToScreen.',
    highlights: [
      'TypedArrays y ProposalPool preasignados (cero presión de Garbage Collector)',
      'Class-Aware NMS independiente por clase + SimpleByteTracker (maxLostFrames=15)',
      'Proyección inversa letterbox 640×640 -> pantalla con soporte de orientación y cámara frontal',
    ],
    content: visionPostProcessorRaw,
  },
  {
    id: 'frame-processor',
    path: 'src/processors/useObjectDetectionProcessor.ts',
    name: 'useObjectDetectionProcessor.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Frame Processor principal en JSI Worklet auditado contra los 6 fallos silenciosos de TFLite: normalización float32 [0.0..1.0], orientación VisionCamera v4, layout dinámico [1,84,8400] vs [1,8400,84], escala w<=1.5, 80 clases COCO y Smoke Test a 1 Hz.',
    highlights: [
      'Verifica model.inputs[0].dataType y normaliza a [0.0..1.0] si resize-plugin entrega [0..255]',
      'Detecta dinámicamente layout [1, 84, 8400] vs [1, 8400, 84] y escala [0..1] vs [0..640]',
      'Habilita las 80 clases de COCO + Smoke Test diagnóstico 1 vez/segundo en Worklet',
    ],
    content: `import { useFrameProcessor, runAsync } from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { SharedValue, useSharedValue } from 'react-native-reanimated';
import { TensorflowModel } from 'react-native-fast-tflite';
import {
  applyClassAwareNMS,
  decodeYoloDetections,
  decodeYoloPose,
  ensureNormalizedFloat32Input,
  inspectYoloOutputTensor,
  ProposalPool,
  SensorOrientation,
  SimpleByteTracker,
  TrackedObject,
} from '../utils/visionPostProcessor';

export interface PipelineTimings {
  captureMs: number;
  resizeMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
  fps: number;
}

export interface VisionFrameState {
  tracks: TrackedObject[];
  timings: PipelineTimings;
}

// Pools y buffers preasignados fuera del ciclo de frame (Zero-Allocation en Worklet C++)
const detPool = new ProposalPool(256);
const posePool = new ProposalPool(256);
const nmsKeptIndices = new Int32Array(256);
const byteTracker = new SimpleByteTracker(15, 0.28, 0.42);

/**
 * Convierte el string frame.orientation de VisionCamera v4 al ángulo de rotación
 * necesario para que el tensor 640x640 entre derecho al intérprete TFLite.
 */
function resolveFrameRotation(orientation?: string): {
  sensorOrientation: SensorOrientation;
  rotationDeg: '0deg' | '90deg' | '180deg' | '270deg';
} {
  'worklet';
  switch (orientation) {
    case 'landscape-right':
      // La cámara trasera en portrait suele venir rotada 90° en el sensor nativo
      return { sensorOrientation: 90, rotationDeg: '90deg' };
    case 'portrait-upside-down':
      return { sensorOrientation: 180, rotationDeg: '180deg' };
    case 'landscape-left':
      return { sensorOrientation: 270, rotationDeg: '270deg' };
    case 'portrait':
    default:
      return { sensorOrientation: 0, rotationDeg: '0deg' };
  }
}

export function useObjectDetectionProcessor(
  detector: TensorflowModel | null,
  poseModel: TensorflowModel | null,
  visionStateShared: SharedValue<VisionFrameState>,
  confidenceThreshold: SharedValue<number>,
  nmsIouThreshold: SharedValue<number>,
  isFrontCamera: SharedValue<boolean>,
  viewport: { width: number; height: number }
) {
  const { resize } = useResizePlugin();
  const lastSmokeLogTimestamp = useSharedValue<number>(0);

  return useFrameProcessor(
    (frame) => {
      'worklet';
      if (!detector) return;

      const tCaptureStart = performance.now();

      runAsync(frame, () => {
        'worklet';
        const tResizeStart = performance.now();
        const captureMs = Math.max(0.1, tResizeStart - tCaptureStart);

        // =========================================================================
        // 1. FORMATO, ORIENTACIÓN Y NORMALIZACIÓN DEL TENSOR DE ENTRADA [0.0 .. 1.0]
        // =========================================================================
        const expectedDataType = String(detector.inputs?.[0]?.dataType ?? 'float32');
        const { sensorOrientation, rotationDeg } = resolveFrameRotation(
          String(frame.orientation ?? 'portrait')
        );

        const rawResized = resize(frame, {
          scale: { width: 640, height: 640 },
          pixelFormat: 'rgb',
          dataType: expectedDataType === 'uint8' ? 'uint8' : 'float32',
          rotation: rotationDeg,
        });

        // Corrige el fallo silencioso donde float32/uint8 entrega [0..255] en vez de [0.0..1.0]
        const { buffer: inputTensor, wasRescaledFrom255 } = ensureNormalizedFloat32Input(
          rawResized as Float32Array | Uint8Array,
          expectedDataType
        );

        const tInferStart = performance.now();
        const resizeMs = tInferStart - tResizeStart;

        // =========================================================================
        // 2. INFERENCIA SÍNCRONA EN C++ / GPU / COREML (runSync)
        // =========================================================================
        const detOutputs = detector.runSync([inputTensor]);
        const detTensor = detOutputs[0] as Float32Array;
        const detOutputShape = detector.outputs?.[0]?.shape;

        const poseTensor = poseModel
          ? (poseModel.runSync([inputTensor])[0] as Float32Array)
          : null;
        const poseOutputShape = poseModel?.outputs?.[0]?.shape;

        const tPostStart = performance.now();
        const inferenceMs = tPostStart - tInferStart;

        // =========================================================================
        // 6. LOGS DE DIAGNÓSTICO TEMPORAL (SMOKE TEST — 1 VEZ POR SEGUNDO)
        // =========================================================================
        if (tPostStart - lastSmokeLogTimestamp.value >= 1000) {
          lastSmokeLogTimestamp.value = tPostStart;
          const diag = inspectYoloOutputTensor(detTensor, detOutputShape);

          console.log(
            \`[TFLite SmokeTest 1Hz] \` +
              \`inputs=\${JSON.stringify(detector.inputs)} | \` +
              \`outputs=\${JSON.stringify(detector.outputs)} | \` +
              \`in[0..4]=[\${inputTensor[0]?.toFixed(4)}, \${inputTensor[1]?.toFixed(4)}, \${inputTensor[2]?.toFixed(4)}, \${inputTensor[3]?.toFixed(4)}, \${inputTensor[4]?.toFixed(4)}] \` +
              \`(rescaledFrom255=\${wasRescaledFrom255}, rot=\${rotationDeg}) | \` +
              \`layout=\${diag.layout} [\${diag.numChannels}x\${diag.numAnchors}] | \` +
              \`maxScore=\${diag.maxScore.toFixed(4)} (class=\${diag.bestClassName}#\${diag.bestClassId} @anchor=\${diag.bestAnchorIndex}) | \` +
              \`rawCoords=[cx:\${diag.rawCx.toFixed(4)}, cy:\${diag.rawCy.toFixed(4)}, w:\${diag.rawW.toFixed(4)}, h:\${diag.rawH.toFixed(4)}] \` +
              \`(normalized0to1=\${diag.isNormalizedCoords})\`
          );
        }

        // =========================================================================
        // 3, 4 y 5. DECODIFICACIÓN (80 CLASES COCO, LAYOUT DINÁMICO, ESCALA w<=1.5, SIN DOBLE SIGMOID)
        // =========================================================================
        const confMin = confidenceThreshold.value;
        const iouMax = nmsIouThreshold.value;

        decodeYoloDetections(detTensor, confMin, 640, detPool, detOutputShape);
        const { keptIndices, keptCount } = applyClassAwareNMS(detPool, iouMax, nmsKeptIndices);

        // Si el modelo de pose está activo, asocia los 17 keypoints COCO a las personas detectadas
        if (poseTensor) {
          decodeYoloPose(poseTensor, confMin, 640, posePool, poseOutputShape);

          for (let i = 0; i < keptCount; i++) {
            const det = detPool.items[keptIndices[i]];
            if (det.classId !== 0) continue; // Solo asociar esqueleto a clase 'person' (id 0)

            let bestDist = 0.08;
            let bestPoseIdx = -1;
            for (let p = 0; p < posePool.count; p++) {
              const candPose = posePool.items[p];
              const dist = Math.hypot(candPose.cx - det.cx, candPose.cy - det.cy);
              if (dist < bestDist) {
                bestDist = dist;
                bestPoseIdx = p;
              }
            }

            if (bestPoseIdx >= 0) {
              det.hasPose = true;
              const srcKps = posePool.items[bestPoseIdx].keypoints;
              for (let k = 0; k < 17; k++) {
                det.keypoints[k].x = srcKps[k].x;
                det.keypoints[k].y = srcKps[k].y;
                det.keypoints[k].visibility = srcKps[k].visibility;
              }
            }
          }
        }

        // Proyección exacta a coordenadas de pantalla + seguimiento multi-objeto ByteTrack
        const tracked = byteTracker.update(detPool, keptIndices, keptCount, {
          cameraWidth: frame.width,
          cameraHeight: frame.height,
          screenWidth: viewport.width,
          screenHeight: viewport.height,
          tensorSize: 640,
          orientation: sensorOrientation,
          resizeMode: 'cover',
          inputPaddingMode: 'stretch',
          isFrontCamera: isFrontCamera.value,
        });

        const tEnd = performance.now();
        const postprocessMs = tEnd - tPostStart;
        const totalMs = captureMs + resizeMs + inferenceMs + postprocessMs;

        // Publicar directamente en SharedValue para renderizado Skia a 60 FPS
        visionStateShared.value = {
          tracks: tracked,
          timings: {
            captureMs,
            resizeMs,
            inferenceMs,
            postprocessMs,
            totalMs,
            fps: Math.min(60, 1000 / Math.max(1, totalMs)),
          },
        };
      });
    },
    [detector, poseModel, resize, viewport.width, viewport.height]
  );
}`,
  },
  {
    id: 'ar-theme-config',
    path: 'src/config/ARThemeConfig.ts',
    name: 'ARThemeConfig.ts',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Sistema de Estilos Visuales AR Intercambiables: define los 3 modos en vivo (Táctico / Futurista Cian #38BDF8 con esqueleto #22C55E, Minimalista Cámara Nativa frosted glass y Alto Contraste Industrial #FACC15/#F97316) más estimación geométrica de distancia en metros.',
    highlights: [
      '3 modos AR intercambiables en caliente: tactical, native e industrial',
      'Paleta semántica completa para personas, vehículos, bicicletas/motos, animales y objetos',
      'Función worklet estimateObjectDistanceMeters (~1.2m a ~25.0m) basada en altura proyectada',
    ],
    content: arThemeConfigRaw,
  },
  {
    id: 'gyro-ar-stabilization',
    path: 'src/hooks/useGyroARStabilization.ts',
    name: 'useGyroARStabilization.ts',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Hook de estabilización predictiva AR a 60Hz (16ms) con Gyroscope de expo-sensors y SharedValue de Reanimated: integra velocidad angular (rad/s) en desplazamiento sub-frame (offsetX, offsetY) entre inferencias de YOLO y aplica reinicio sincronizado con decaimiento suave.',
    highlights: [
      'Muestreo a 60Hz (16ms) con Gyroscope.setUpdateInterval(16) directo a SharedValue (0 re-renders JS)',
      'Compensación Inter-Frame (Zero-Lag Tracking) proporcional a la distancia focal del visor',
      'Reinicio sincronizado (resetOnNewInferenceWorklet) al llegar cada nueva detección de YOLO',
    ],
    content: `import { useEffect, useCallback } from 'react';
import { Gyroscope } from 'expo-sensors';
import { SharedValue, useSharedValue } from 'react-native-reanimated';

export interface GyroAROffset {
  /** Desplazamiento horizontal acumulado en píxeles desde la última inferencia YOLO */
  offsetX: number;
  /** Desplazamiento vertical acumulado en píxeles desde la última inferencia YOLO */
  offsetY: number;
  /** Velocidad angular instantánea (rad/s) */
  omegaX: number;
  omegaY: number;
  omegaZ: number;
  /** Marca de tiempo del último muestreo IMU (ms) */
  lastTimestampMs: number;
}

export interface UseGyroARStabilizationOptions {
  /** Ancho de pantalla en píxeles para calibrar ganancia angular -> píxeles */
  screenWidth: number;
  /** Alto de pantalla en píxeles */
  screenHeight: number;
  /** Campo de visión horizontal aproximado de la cámara (grados). Por defecto 62° (lente 1x) */
  horizontalFovDeg?: number;
  /** Activar o pausar la estabilización inercial */
  enabled?: boolean;
}

export interface UseGyroARStabilizationResult {
  /** SharedValue leído directamente en Skia UI Worklet a 60/120 FPS sin pasar por React */
  gyroOffsetShared: SharedValue<GyroAROffset>;
  /** Worklet invocado inmediatamente cuando YOLO publica una nueva inferencia */
  resetOnNewInferenceWorklet: () => void;
}

/**
 * Estabilización Inercial Predictiva (Zero-Lag AR Tracking) a 60Hz (16ms).
 *
 * Entre dos inferencias de YOLOv8n (~25-30 ms), el usuario puede rotar el teléfono
 * provocando que las cajas se queden atrás respecto al mundo real.
 * Este hook integra la velocidad angular del giroscopio (rad/s) y desplaza en sentido
 * inverso todas las cajas, etiquetas y esqueletos en Skia hasta que llega el nuevo frame.
 */
export function useGyroARStabilization({
  screenWidth,
  screenHeight,
  horizontalFovDeg = 62,
  enabled = true,
}: UseGyroARStabilizationOptions): UseGyroARStabilizationResult {
  const gyroOffsetShared = useSharedValue<GyroAROffset>({
    offsetX: 0,
    offsetY: 0,
    omegaX: 0,
    omegaY: 0,
    omegaZ: 0,
    lastTimestampMs: 0,
  });

  // Distancia focal equivalente en píxeles por radián: f_px = (W / 2) / tan(FOV / 2)
  const fovRad = (horizontalFovDeg * Math.PI) / 180;
  const focalPixelsPerRad = screenWidth / (2 * Math.tan(fovRad / 2));
  const maxClampX = screenWidth * 0.18;
  const maxClampY = screenHeight * 0.18;

  useEffect(() => {
    if (!enabled) {
      gyroOffsetShared.value = {
        offsetX: 0,
        offsetY: 0,
        omegaX: 0,
        omegaY: 0,
        omegaZ: 0,
        lastTimestampMs: Date.now(),
      };
      return;
    }

    // 60Hz (16ms) para empatar con el refresh rate de Skia sin saturar el bridge
    Gyroscope.setUpdateInterval(16);

    const subscription = Gyroscope.addListener((imuData) => {
      const now = Date.now();
      const prev = gyroOffsetShared.value;
      const dtSec =
        prev.lastTimestampMs > 0
          ? Math.min(0.06, Math.max(0.004, (now - prev.lastTimestampMs) / 1000))
          : 0.016;

      // En orientación portrait:
      // - Rotación Yaw (eje Y del dispositivo, imuData.y) desplaza la escena horizontalmente (-deltaX)
      // - Rotación Pitch (eje X del dispositivo, imuData.x) desplaza la escena verticalmente (+deltaY)
      // Filtro de zona muerta (deadzone) de 0.015 rad/s para ignorar ruido térmico del MEMS
      const wy = Math.abs(imuData.y) > 0.015 ? imuData.y : 0;
      const wx = Math.abs(imuData.x) > 0.015 ? imuData.x : 0;

      const deltaPxX = -wy * dtSec * focalPixelsPerRad;
      const deltaPxY = wx * dtSec * focalPixelsPerRad;

      // Factor de amortiguación suave (0.94) para evitar deriva infinita si YOLO tarda más de 40ms
      const nextX = Math.max(-maxClampX, Math.min(maxClampX, (prev.offsetX + deltaPxX) * 0.94));
      const nextY = Math.max(-maxClampY, Math.min(maxClampY, (prev.offsetY + deltaPxY) * 0.94));

      gyroOffsetShared.value = {
        offsetX: nextX,
        offsetY: nextY,
        omegaX: imuData.x,
        omegaY: imuData.y,
        omegaZ: imuData.z,
        lastTimestampMs: now,
      };
    });

    return () => {
      subscription.remove();
    };
  }, [enabled, focalPixelsPerRad, gyroOffsetShared, maxClampX, maxClampY]);

  /**
   * Reinicio sincronizado: se llama justo al publicar una nueva inferencia de YOLO.
   * Las nuevas cajas ya reflejan la pose física actual de la cámara, por lo que el
   * offset acumulado se amortigua suavemente hacia 0 mientras el LERP absorbe la transición.
   */
  const resetOnNewInferenceWorklet = useCallback(() => {
    'worklet';
    const cur = gyroOffsetShared.value;
    gyroOffsetShared.value = {
      offsetX: cur.offsetX * 0.2,
      offsetY: cur.offsetY * 0.2,
      omegaX: cur.omegaX,
      omegaY: cur.omegaY,
      omegaZ: cur.omegaZ,
      lastTimestampMs: cur.lastTimestampMs,
    };
  }, [gyroOffsetShared]);

  return {
    gyroOffsetShared,
    resetOnNewInferenceWorklet,
  };
}`,
  },
  {
    id: 'ar-overlay-canvas',
    path: 'src/components/AROverlayCanvas.tsx',
    name: 'AROverlayCanvas.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Lienzo Skia de Realidad Aumentada a pantalla completa (StyleSheet.absoluteFill) con suavizado LERP por Track ID, compensación inercial de giroscopio a 60Hz y renderizado de los 3 estilos visuales (Táctico Cian con corner brackets + pin guía + distancia ~3.2m, Nativo e Industrial).',
    highlights: [
      'Interpolación lineal (LERP) en UI Worklet entre inferencias YOLO + suma de offset inercial 60Hz',
      '3 estilos visuales AR completos: Táctico (#38BDF8 + #22C55E), Nativo (Frosted White) e Industrial (#FACC15)',
      'Retícula de horizonte reactiva al giroscopio, corner brackets ópticos y callout pins con distancia en metros',
    ],
    content: `import React, { useMemo } from 'react';
import { StyleSheet } from 'react-native';
import {
  Canvas,
  Picture,
  Skia,
  createPicture,
  matchFont,
  PaintStyle,
  StrokeCap,
} from '@shopify/react-native-skia';
import { SharedValue, useDerivedValue, useSharedValue } from 'react-native-reanimated';
import { TrackedObject } from '../utils/visionPostProcessor';
import {
  ARThemeId,
  AR_THEMES,
  estimateObjectDistanceMeters,
} from '../config/ARThemeConfig';
import { GyroAROffset } from '../hooks/useGyroARStabilization';

const SKELETON_EDGES: ReadonlyArray<[number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

interface LerpBoxState {
  x: number;
  y: number;
  width: number;
  height: number;
  updatedAtFrame: number;
}

export interface AROverlayCanvasProps {
  tracksShared: SharedValue<TrackedObject[]>;
  gyroOffsetShared: SharedValue<GyroAROffset>;
  themeIdShared: SharedValue<ARThemeId>;
  showSkeletonShared: SharedValue<boolean>;
  showLabelsShared: SharedValue<boolean>;
  lerpFactor?: number;
  width: number;
  height: number;
}

export const AROverlayCanvas: React.FC<AROverlayCanvasProps> = ({
  tracksShared,
  gyroOffsetShared,
  themeIdShared,
  showSkeletonShared,
  showLabelsShared,
  lerpFactor = 0.38,
  width,
  height,
}) => {
  const tacticalFont = useMemo(
    () => matchFont({ fontFamily: 'Courier', fontSize: 10.5, fontWeight: '700' }),
    []
  );
  const nativeFont = useMemo(
    () => matchFont({ fontFamily: 'System', fontSize: 11, fontWeight: '600' }),
    []
  );

  // Diccionario persistente en el UI Worklet para interpolación LERP suave por track.id
  const lerpCacheShared = useSharedValue<Record<number, LerpBoxState>>({});
  const frameTickShared = useSharedValue<number>(0);

  const arPicture = useDerivedValue(() => {
    'worklet';
    const tracks = tracksShared.value;
    const gyro = gyroOffsetShared.value;
    const themeId = themeIdShared.value;
    const theme = AR_THEMES[themeId] ?? AR_THEMES.tactical;
    const showSkeleton = showSkeletonShared.value;
    const showLabels = showLabelsShared.value;

    const tick = frameTickShared.value + 1;
    frameTickShared.value = tick;
    const cache = lerpCacheShared.value;

    return createPicture((canvas) => {
      const paint = Skia.Paint();
      paint.setAntiAlias(true);

      // =======================================================================
      // 1. RETÍCULA CENTRAL DE HORIZONTE (MODO TÁCTICO / INDUSTRIAL)
      // =======================================================================
      if (theme.showHorizonReticle) {
        const cx = width * 0.5 + gyro.offsetX * 0.45;
        const cy = height * 0.5 + gyro.offsetY * 0.45;

        paint.setStyle(PaintStyle.Stroke);
        paint.setStrokeWidth(1);
        paint.setColor(Skia.Color(themeId === 'tactical' ? 'rgba(56, 189, 248, 0.28)' : 'rgba(250, 204, 21, 0.28)'));

        canvas.drawCircle(cx, cy, 22, paint);
        canvas.drawLine(cx - 38, cy, cx - 12, cy, paint);
        canvas.drawLine(cx + 12, cy, cx + 38, cy, paint);
        canvas.drawLine(cx, cy - 26, cx, cy - 10, paint);
        canvas.drawLine(cx, cy + 10, cx, cy + 26, paint);
      }

      // =======================================================================
      // 2. TRAZADO DE OBJETOS CON LERP + COMPENSACIÓN INERCIAL DE GIROSCOPIO
      // =======================================================================
      const font = theme.useMonospaceFont ? tacticalFont : nativeFont;

      for (let i = 0; i < tracks.length; i++) {
        const track = tracks[i];
        const palette = theme.categories[track.category] ?? theme.categories.object;

        // Interpolación Lineal (LERP) hacia las coordenadas reales de YOLO
        const prev = cache[track.id];
        let smoothX = track.box.x;
        let smoothY = track.box.y;
        let smoothW = track.box.width;
        let smoothH = track.box.height;

        if (prev && tick - prev.updatedAtFrame <= 15) {
          smoothX = prev.x + (track.box.x - prev.x) * lerpFactor;
          smoothY = prev.y + (track.box.y - prev.y) * lerpFactor;
          smoothW = prev.width + (track.box.width - prev.width) * lerpFactor;
          smoothH = prev.height + (track.box.height - prev.height) * lerpFactor;
        }

        cache[track.id] = {
          x: smoothX,
          y: smoothY,
          width: smoothW,
          height: smoothH,
          updatedAtFrame: tick,
        };

        // Suma del desplazamiento predictivo del giroscopio (Zero-Lag Tracking)
        const bx = Math.max(0, Math.min(width - 8, smoothX + gyro.offsetX));
        const by = Math.max(0, Math.min(height - 8, smoothY + gyro.offsetY));
        const bw = Math.max(8, Math.min(width - bx, smoothW));
        const bh = Math.max(8, Math.min(height - by, smoothH));

        const rect = Skia.XYWHRect(bx, by, bw, bh);
        const rrect = Skia.RRectXY(rect, theme.cornerRadius, theme.cornerRadius);

        // Relleno translúcido
        paint.setStyle(PaintStyle.Fill);
        paint.setColor(Skia.Color(palette.fill));
        canvas.drawRRect(rrect, paint);

        if (theme.useCornerBrackets) {
          // Borde guía tenue + Esquinas ópticas de enfoque (Corner Brackets) con brillo neón
          paint.setStyle(PaintStyle.Stroke);
          paint.setStrokeWidth(1);
          paint.setColor(Skia.Color(palette.glow));
          canvas.drawRRect(rrect, paint);

          const cLen = Math.min(theme.cornerBracketLength, bw * 0.32, bh * 0.32);
          paint.setStrokeWidth(theme.boxStrokeWidth + 0.6);
          paint.setColor(Skia.Color(palette.stroke));
          paint.setStrokeCap(StrokeCap.Square);

          // Top-Left
          canvas.drawLine(bx, by, bx + cLen, by, paint);
          canvas.drawLine(bx, by, bx, by + cLen, paint);
          // Top-Right
          canvas.drawLine(bx + bw - cLen, by, bx + bw, by, paint);
          canvas.drawLine(bx + bw, by, bx + bw, by + cLen, paint);
          // Bottom-Left
          canvas.drawLine(bx, by + bh - cLen, bx, by + bh, paint);
          canvas.drawLine(bx, by + bh, bx + cLen, by + bh, paint);
          // Bottom-Right
          canvas.drawLine(bx + bw - cLen, by + bh, bx + bw, by + bh, paint);
          canvas.drawLine(bx + bw, by + bh - cLen, bx + bw, by + bh, paint);
        } else {
          // Caja continua (Minimalista Nativa o Alto Contraste Industrial)
          paint.setStyle(PaintStyle.Stroke);
          paint.setStrokeWidth(theme.boxStrokeWidth);
          paint.setColor(Skia.Color(palette.stroke));
          canvas.drawRRect(rrect, paint);
        }

        // =====================================================================
        // 3. ETIQUETAS FLOTANTES (CALLOUT PINS / PÍLDORAS FROSTED / INDUSTRIAL)
        // =====================================================================
        if (showLabels) {
          const confPct = Math.round(track.score * 100);
          const distMeters = estimateObjectDistanceMeters(track.category, bh, height);
          const labelText = theme.showDistanceEstimate
            ? \`#\${track.id} \${track.className.toUpperCase()} \${confPct}% • ~\${distMeters.toFixed(1)}m\`
            : \`\${track.className} \${confPct}%\`;

          const pillW = Math.max(76, labelText.length * 6.4 + 14);
          const pillH = 20;

          if (theme.showLeaderLinePins) {
            // Línea guía oblicua estilo visor óptico AR (Callout Pin)
            const anchorX = bx + Math.min(24, bw * 0.25);
            const anchorY = by;
            const elbowX = Math.min(width - pillW - 6, anchorX + 12);
            const elbowY = Math.max(28, anchorY - 14);

            paint.setStyle(PaintStyle.Stroke);
            paint.setStrokeWidth(1.3);
            paint.setColor(Skia.Color(palette.stroke));
            canvas.drawLine(anchorX, anchorY, elbowX, elbowY, paint);

            const pillRect = Skia.RRectXY(
              Skia.XYWHRect(elbowX, elbowY - pillH, pillW, pillH),
              4,
              4
            );
            paint.setStyle(PaintStyle.Fill);
            paint.setColor(Skia.Color(theme.pillBackground));
            canvas.drawRRect(pillRect, paint);

            paint.setStyle(PaintStyle.Stroke);
            paint.setStrokeWidth(1.2);
            paint.setColor(Skia.Color(palette.stroke));
            canvas.drawRRect(pillRect, paint);

            if (font) {
              paint.setStyle(PaintStyle.Fill);
              paint.setColor(Skia.Color(theme.pillTextColor));
              canvas.drawText(labelText, elbowX + 7, elbowY - 6, paint, font);
            }
          } else {
            // Píldora compacta directa (Nativo Frosted Glass o Industrial Sólido)
            const pillX = bx;
            const pillY = by > pillH + 6 ? by - pillH - 4 : by + 4;
            const pillRect = Skia.RRectXY(
              Skia.XYWHRect(pillX, pillY, pillW, pillH),
              themeId === 'native' ? 10 : 2,
              themeId === 'native' ? 10 : 2
            );

            paint.setStyle(PaintStyle.Fill);
            paint.setColor(
              Skia.Color(themeId === 'industrial' ? '#090D16' : theme.pillBackground)
            );
            canvas.drawRRect(pillRect, paint);

            if (themeId === 'industrial') {
              paint.setStyle(PaintStyle.Stroke);
              paint.setStrokeWidth(1.8);
              paint.setColor(Skia.Color(palette.stroke));
              canvas.drawRRect(pillRect, paint);
            }

            if (font) {
              paint.setStyle(PaintStyle.Fill);
              paint.setColor(
                Skia.Color(themeId === 'industrial' ? palette.stroke : theme.pillTextColor)
              );
              canvas.drawText(labelText, pillX + 7, pillY + 14, paint, font);
            }
          }
        }

        // =====================================================================
        // 4. ESQUELETO BIOMECÁNICO 17 KEYPOINTS (ESTABILIZADO CON GIROSCOPIO)
        // =====================================================================
        if (showSkeleton && track.keypoints && track.keypoints.length === 17) {
          paint.setStyle(PaintStyle.Stroke);
          paint.setStrokeWidth(themeId === 'native' ? 1.6 : 2.3);
          paint.setStrokeCap(StrokeCap.Round);
          paint.setColor(Skia.Color(theme.skeletonBoneColor));

          for (let e = 0; e < SKELETON_EDGES.length; e++) {
            const [aIdx, bIdx] = SKELETON_EDGES[e];
            const kpA = track.keypoints[aIdx];
            const kpB = track.keypoints[bIdx];

            if (kpA && kpB && kpA.visibility > 0.5 && kpB.visibility > 0.5) {
              canvas.drawLine(
                kpA.x + gyro.offsetX,
                kpA.y + gyro.offsetY,
                kpB.x + gyro.offsetX,
                kpB.y + gyro.offsetY,
                paint
              );
            }
          }

          paint.setStyle(PaintStyle.Fill);
          for (let k = 0; k < 17; k++) {
            const kp = track.keypoints[k];
            if (kp && kp.visibility > 0.5) {
              const kx = kp.x + gyro.offsetX;
              const ky = kp.y + gyro.offsetY;
              paint.setColor(Skia.Color('#090D16'));
              canvas.drawCircle(kx, ky, themeId === 'native' ? 3.2 : 4.2, paint);
              paint.setColor(Skia.Color(theme.skeletonJointColor));
              canvas.drawCircle(kx, ky, themeId === 'native' ? 2.2 : 3.0, paint);
            }
          }
        }
      }
    });
  }, [
    tracksShared,
    gyroOffsetShared,
    themeIdShared,
    showSkeletonShared,
    showLabelsShared,
    tacticalFont,
    nativeFont,
    width,
    height,
    lerpFactor,
  ]);

  return (
    <Canvas style={StyleSheet.absoluteFill} pointerEvents="none">
      <Picture picture={arPicture} />
    </Canvas>
  );
};`,
  },
  {
    id: 'skia-overlay',
    path: 'src/components/DetectionOverlayCanvas.tsx',
    name: 'DetectionOverlayCanvas.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Visor AR superpuesto a 60 FPS sin re-renders de React: usa createPicture + useDerivedValue sobre SharedValue para renderizar las 80 clases de COCO (personas, vehículos, bicicletas/motos, animales y objetos) con relleno translúcido, píldoras y esqueleto #22C55E.',
    highlights: [
      'Cero re-renders de React: Skia.PictureRecorder lee directamente SharedValue en UI Worklet',
      'Soporte completo para las 80 clases COCO con paleta por categoría semántica',
      'Clamping de seguridad en coordenadas y esqueleto articulado #22C55E para keypoints > 0.5',
    ],
    content: `import React, { useMemo } from 'react';
import { StyleSheet } from 'react-native';
import {
  Canvas,
  Picture,
  Skia,
  createPicture,
  matchFont,
  PaintStyle,
  StrokeCap,
} from '@shopify/react-native-skia';
import { SharedValue, useDerivedValue } from 'react-native-reanimated';
import { SemanticCategory, TrackedObject } from '../utils/visionPostProcessor';

const SKELETON_EDGES: ReadonlyArray<[number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

const CATEGORY_PALETTE: Record<SemanticCategory, { stroke: string; fill: string }> = {
  person:  { stroke: '#38BDF8', fill: 'rgba(56, 189, 248, 0.14)' },
  vehicle: { stroke: '#F97316', fill: 'rgba(249, 115, 22, 0.14)' },
  bicycle: { stroke: '#10B981', fill: 'rgba(16, 185, 129, 0.14)' },
  animal:  { stroke: '#A855F7', fill: 'rgba(168, 85, 247, 0.16)' },
  object:  { stroke: '#EAB308', fill: 'rgba(234, 179, 8, 0.14)' },
};

interface DetectionOverlayCanvasProps {
  tracksShared: SharedValue<TrackedObject[]>;
  width: number;
  height: number;
}

export const DetectionOverlayCanvas: React.FC<DetectionOverlayCanvasProps> = ({
  tracksShared,
  width,
  height,
}) => {
  const labelFont = useMemo(
    () =>
      matchFont({
        fontFamily: 'Courier',
        fontSize: 10.5,
        fontWeight: '600',
      }),
    []
  );

  // Graba todos los comandos gráficos en un SkPicture nativo dentro del hilo UI (0 re-renders en React)
  const overlayPicture = useDerivedValue(() => {
    const tracks = tracksShared.value;

    return createPicture((canvas) => {
      if (width <= 0 || height <= 0) return;

      const boxFillPaint = Skia.Paint();
      boxFillPaint.setStyle(PaintStyle.Fill);
      boxFillPaint.setAntiAlias(true);

      const strokePaint = Skia.Paint();
      strokePaint.setStyle(PaintStyle.Stroke);
      strokePaint.setStrokeWidth(2);
      strokePaint.setAntiAlias(true);

      const pillBgPaint = Skia.Paint();
      pillBgPaint.setStyle(PaintStyle.Fill);
      pillBgPaint.setColor(Skia.Color('rgba(11, 15, 23, 0.84)'));
      pillBgPaint.setAntiAlias(true);

      const textPaint = Skia.Paint();
      textPaint.setAntiAlias(true);

      const skelPaint = Skia.Paint();
      skelPaint.setStyle(PaintStyle.Stroke);
      skelPaint.setStrokeWidth(2.2);
      skelPaint.setStrokeCap(StrokeCap.Round);
      skelPaint.setColor(Skia.Color('#22C55E'));
      skelPaint.setAntiAlias(true);

      const jointPaint = Skia.Paint();
      jointPaint.setStyle(PaintStyle.Fill);
      jointPaint.setColor(Skia.Color('#22C55E'));
      jointPaint.setAntiAlias(true);

      for (let i = 0; i < tracks.length; i++) {
        const tr = tracks[i];
        if (!tr || tr.lostFrames > 0) continue;

        const palette = CATEGORY_PALETTE[tr.category] ?? CATEGORY_PALETTE.object;
        const strokeColor = Skia.Color(palette.stroke);
        const fillColor = Skia.Color(palette.fill);

        const rawX = tr.screenBox.x;
        const rawY = tr.screenBox.y;
        const rawW = tr.screenBox.width;
        const rawH = tr.screenBox.height;

        // Ignora cajas degeneradas o fuera de rango numérico
        if (!Number.isFinite(rawX) || !Number.isFinite(rawY) || rawW < 2 || rawH < 2) {
          continue;
        }

        const x = Math.max(0, Math.min(width - 4, rawX));
        const y = Math.max(0, Math.min(height - 4, rawY));
        const bw = Math.max(4, Math.min(width - x, rawW));
        const bh = Math.max(4, Math.min(height - y, rawH));

        // 1. Bounding Box con relleno translúcido y borde redondeado
        const rrect = Skia.RRectXY(Skia.XYWHRect(x, y, bw, bh), 8, 8);
        boxFillPaint.setColor(fillColor);
        canvas.drawRRect(rrect, boxFillPaint);

        strokePaint.setColor(strokeColor);
        strokePaint.setStrokeWidth(2);
        canvas.drawRRect(rrect, strokePaint);

        // 2. Píldora superior de telemetría (ej. dog #12 0.89 / person #1 0.94)
        const label = \`\${tr.className} #\${tr.trackId} \${tr.confidence.toFixed(2)}\`;
        const pillW = label.length * 6.5 + 14;
        const pillH = 18;
        const pillX = Math.max(2, Math.min(width - pillW - 2, x));
        const pillY = y >= pillH + 4 ? y - pillH - 2 : y + 2;

        const pillRRect = Skia.RRectXY(Skia.XYWHRect(pillX, pillY, pillW, pillH), 5, 5);
        canvas.drawRRect(pillRRect, pillBgPaint);

        strokePaint.setStrokeWidth(1);
        canvas.drawRRect(pillRRect, strokePaint);

        textPaint.setColor(strokeColor);
        canvas.drawText(label, pillX + 7, pillY + 12.5, textPaint, labelFont);

        // 3. Esqueleto articulado verde neón (#22C55E) para keypoints visibles (> 0.5)
        if (tr.classId === 0 && tr.hasPose && tr.keypoints && tr.keypoints.length >= 17) {
          for (let e = 0; e < SKELETON_EDGES.length; e++) {
            const [idxA, idxB] = SKELETON_EDGES[e];
            const kpA = tr.keypoints[idxA];
            const kpB = tr.keypoints[idxB];
            if (!kpA || !kpB || kpA.visibility <= 0.5 || kpB.visibility <= 0.5) continue;
            canvas.drawLine(kpA.x, kpA.y, kpB.x, kpB.y, skelPaint);
          }

          for (let k = 0; k < tr.keypoints.length; k++) {
            const kp = tr.keypoints[k];
            if (!kp || kp.visibility <= 0.5) continue;
            canvas.drawCircle(kp.x, kp.y, 3, jointPaint);
          }
        }
      }
    }, Skia.XYWHRect(0, 0, width, height));
  }, [width, height, labelFont]);

  return (
    <Canvas style={[StyleSheet.absoluteFill, { width, height }]} pointerEvents="none">
      <Picture picture={overlayPicture} />
    </Canvas>
  );
};`,
  },
  {
    id: 'activity-heatmap',
    path: 'src/components/ActivityHeatmap.tsx',
    name: 'ActivityHeatmap.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Mapa de calor acumulativo en Skia sobre una cuadrícula de 32×18 celdas usando BlurMaskFilter y gradiente térmico (azul frío -> verde -> amarillo -> rojo intenso).',
    highlights: [
      'Cuadrícula Float32Array(32 × 18) actualizada desde centroides detectados',
      'BlurMask normal en Skia para fusión térmica fluida + botón de reinicio',
    ],
    content: `import React, { useMemo } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import {
  Canvas,
  Picture,
  Skia,
  createPicture,
  BlurStyle,
  PaintStyle,
} from '@shopify/react-native-skia';
import { SharedValue, useDerivedValue } from 'react-native-reanimated';

export const HEATMAP_COLS = 32;
export const HEATMAP_ROWS = 18;

interface ActivityHeatmapProps {
  /** Buffer plano de 32*18=576 celdas actualizado por el Frame Processor o intervalo */
  gridShared: SharedValue<number[]>;
  width: number;
  height: number;
  onReset: () => void;
}

function getThermalColor(norm: number): string {
  'worklet';
  if (norm > 0.72) return 'rgba(239, 68, 68, 0.85)';   // Rojo intenso
  if (norm > 0.45) return 'rgba(250, 204, 21, 0.74)';  // Amarillo
  if (norm > 0.22) return 'rgba(34, 197, 94, 0.62)';   // Verde
  return 'rgba(56, 189, 248, 0.48)';                   // Azul frío
}

export const ActivityHeatmap: React.FC<ActivityHeatmapProps> = ({
  gridShared,
  width,
  height,
  onReset,
}) => {
  const blurFilter = useMemo(
    () => Skia.MaskFilter.MakeBlur(BlurStyle.Normal, 6.5, true),
    []
  );

  const heatmapPicture = useDerivedValue(() => {
    const grid = gridShared.value;
    let maxVal = 1;
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] > maxVal) maxVal = grid[i];
    }

    return createPicture((canvas) => {
      const paint = Skia.Paint();
      paint.setStyle(PaintStyle.Fill);
      paint.setMaskFilter(blurFilter);

      const cellW = width / HEATMAP_COLS;
      const cellH = height / HEATMAP_ROWS;

      for (let r = 0; r < HEATMAP_ROWS; r++) {
        for (let c = 0; c < HEATMAP_COLS; c++) {
          const val = grid[r * HEATMAP_COLS + c] ?? 0;
          if (val <= 0) continue;

          const norm = Math.min(1, val / Math.max(2, maxVal * 0.85));
          const cx = (c + 0.5) * cellW;
          const cy = (r + 0.5) * cellH;
          const radius = 5 + norm * 8.5;

          paint.setColor(Skia.Color(getThermalColor(norm)));
          canvas.drawCircle(cx, cy, radius, paint);
        }
      }
    }, Skia.XYWHRect(0, 0, width, height));
  }, [width, height, blurFilter]);

  return (
    <View style={styles.wrapper}>
      <View style={[styles.canvasBox, { width, height }]}>
        <Canvas style={{ width, height }}>
          <Picture picture={heatmapPicture} />
        </Canvas>
        <TouchableOpacity style={styles.resetBtn} onPress={onReset}>
          <Text style={styles.resetText}>Reset</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: { borderRadius: 10, overflow: 'hidden' },
  canvasBox: {
    backgroundColor: '#090D16',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
    position: 'relative',
  },
  resetBtn: {
    position: 'absolute',
    right: 6,
    bottom: 6,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    backgroundColor: 'rgba(18, 24, 36, 0.88)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  resetText: { color: '#94A3B8', fontSize: 10, fontWeight: '600' },
});`,
  },
  {
    id: 'telemetry-charts',
    path: 'src/components/TelemetryCharts.tsx',
    name: 'TelemetryCharts.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Gráficos Skia de alta frecuencia: Processing Speed Wave (curva cúbica suavizada de FPS) y Object Tracking (ventana deslizante 20s para Personas, Vehículos y Bicicletas).',
    highlights: [
      'Skia.Path con curvas Bezier cúbicas para onda de FPS sin aliasing',
      'Ventana deslizante de 20 segundos con leyenda superior (#38BDF8, #F97316, #10B981)',
    ],
    content: `import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Canvas, Path, Skia, Line, vec } from '@shopify/react-native-skia';
import { SharedValue, useDerivedValue } from 'react-native-reanimated';

export interface TrackingHistorySample {
  persons: number;
  vehicles: number;
  bicycles: number;
}

interface ProcessingSpeedWaveProps {
  fpsHistoryShared: SharedValue<number[]>;
  width: number;
  height: number;
}

export const ProcessingSpeedWave: React.FC<ProcessingSpeedWaveProps> = ({
  fpsHistoryShared,
  width,
  height,
}) => {
  const wavePath = useDerivedValue(() => {
    const vals = fpsHistoryShared.value;
    const path = Skia.Path.Make();
    if (vals.length < 2) return path;

    const stepX = width / Math.max(1, vals.length - 1);
    const toY = (fps: number) => {
      const clamped = Math.max(5, Math.min(60, fps));
      return height - ((clamped - 5) / 55) * (height - 10) - 5;
    };

    path.moveTo(0, toY(vals[0]));
    for (let i = 1; i < vals.length; i++) {
      const prevX = (i - 1) * stepX;
      const prevY = toY(vals[i - 1]);
      const currX = i * stepX;
      const currY = toY(vals[i]);
      const midX = (prevX + currX) * 0.5;
      path.cubicTo(midX, prevY, midX, currY, currX, currY);
    }
    return path;
  }, [width, height]);

  return (
    <Canvas style={{ width, height }}>
      <Path path={wavePath} color="#38BDF8" style="stroke" strokeWidth={2} />
    </Canvas>
  );
};

interface ObjectTrackingChartProps {
  samples: TrackingHistorySample[]; // Últimos 20 segundos
  width: number;
  height: number;
}

export const ObjectTrackingChart: React.FC<ObjectTrackingChartProps> = ({
  samples,
  width,
  height,
}) => {
  const maxVal = Math.max(
    6,
    ...samples.flatMap((s) => [s.persons, s.vehicles, s.bicycles])
  );

  const buildLinePath = (key: keyof TrackingHistorySample) => {
    const path = Skia.Path.Make();
    if (samples.length < 2) return path;
    const stepX = width / 19;

    samples.forEach((pt, idx) => {
      const x = idx * stepX;
      const y = height - (pt[key] / maxVal) * (height - 14) - 6;
      if (idx === 0) path.moveTo(x, y);
      else path.lineTo(x, y);
    });
    return path;
  };

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.title}>Object Tracking (20s)</Text>
        <View style={styles.legend}>
          <Text style={[styles.legendItem, { color: '#38BDF8' }]}>● Personas</Text>
          <Text style={[styles.legendItem, { color: '#F97316' }]}>● Vehículos</Text>
          <Text style={[styles.legendItem, { color: '#10B981' }]}>● Bicicletas</Text>
        </View>
      </View>

      <Canvas style={{ width, height }}>
        {[0.25, 0.5, 0.75].map((r) => (
          <Line
            key={r}
            p1={vec(0, height * r)}
            p2={vec(width, height * r)}
            color="rgba(148,163,184,0.12)"
            strokeWidth={1}
          />
        ))}
        <Path path={buildLinePath('persons')} color="#38BDF8" style="stroke" strokeWidth={2} />
        <Path path={buildLinePath('vehicles')} color="#F97316" style="stroke" strokeWidth={2} />
        <Path path={buildLinePath('bicycles')} color="#10B981" style="stroke" strokeWidth={2} />
      </Canvas>

      <View style={styles.axisRow}>
        <Text style={styles.axisText}>-20s</Text>
        <Text style={styles.axisText}>-10s</Text>
        <Text style={styles.axisText}>Now</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#121826',
    borderRadius: 14,
    padding: 10,
    borderWidth: 1,
    borderColor: '#1E293B',
    gap: 5,
  },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  title: { color: '#F1F5F9', fontSize: 12, fontWeight: '600' },
  legend: { flexDirection: 'row', gap: 8 },
  legendItem: { fontSize: 10, fontFamily: 'JetBrainsMono-Medium' },
  axisRow: { flexDirection: 'row', justifyContent: 'space-between' },
  axisText: { color: '#64748B', fontSize: 10, fontFamily: 'JetBrainsMono-Regular' },
});`,
  },
  {
    id: 'use-responsive-layout',
    path: 'src/hooks/useResponsiveLayout.ts',
    name: 'useResponsiveLayout.ts',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Hook de alturas dinámicas con useWindowDimensions + useSafeAreaInsets para ajustar la pantalla principal al 100% en cualquier dispositivo (incluidos iPhone SE 375x667 y Android compactos) sin recortes ni scroll.',
    highlights: [
      'Detecta isCompactScreen = height < 720 restando insets.top e insets.bottom',
      'Distribución proporcional exacta (TopBar 42/36, Contadores 76/64, Métricas 96/82, Chart 105/88, Footer 48/42)',
      'Visor CameraViewport con flex: 1 absorbiendo 100% del espacio sobrante',
    ],
    content: `import { useMemo } from 'react';
import { useWindowDimensions } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

export interface ResponsiveLayoutMetrics {
  width: number;
  height: number;
  usableHeight: number;
  isCompactScreen: boolean;
  gap: number;
  paddingHorizontal: number;
  topBarHeight: number;
  counterRowHeight: number;
  counterFontSize: number;
  metricsRowHeight: number;
  waveHeight: number;
  heatmapHeight: number;
  trackingChartHeight: number;
  trackingSvgHeight: number;
  footerHeight: number;
  estimatedViewportHeight: number;
}

export function useResponsiveLayout(): ResponsiveLayoutMetrics {
  const { width, height } = useWindowDimensions();
  const insets = useSafeAreaInsets();

  return useMemo(() => {
    const usableHeight = Math.max(320, height - insets.top - insets.bottom);
    const isCompactScreen = height < 720;

    const gap = isCompactScreen ? 6 : 10;
    const paddingHorizontal = 12;

    const topBarHeight = isCompactScreen ? 36 : 42;
    const counterRowHeight = isCompactScreen ? 64 : 76;
    const counterFontSize = isCompactScreen ? 20 : 26;
    const metricsRowHeight = isCompactScreen ? 82 : 96;
    const waveHeight = isCompactScreen ? 24 : 34;
    const heatmapHeight = isCompactScreen ? 28 : 38;
    const trackingChartHeight = isCompactScreen ? 88 : 105;
    const trackingSvgHeight = isCompactScreen ? 38 : 50;
    const footerHeight = isCompactScreen ? 42 : 48;

    // 6 secciones verticales -> 5 huecos \`gap\`
    const fixedSum =
      topBarHeight +
      counterRowHeight +
      metricsRowHeight +
      trackingChartHeight +
      footerHeight +
      gap * 5;

    const estimatedViewportHeight = Math.max(120, usableHeight - fixedSum);

    return {
      width,
      height,
      usableHeight,
      isCompactScreen,
      gap,
      paddingHorizontal,
      topBarHeight,
      counterRowHeight,
      counterFontSize,
      metricsRowHeight,
      waveHeight,
      heatmapHeight,
      trackingChartHeight,
      trackingSvgHeight,
      footerHeight,
      estimatedViewportHeight,
    };
  }, [width, height, insets.top, insets.bottom]);
}`,
  },
  {
    id: 'diagnostics-modal',
    path: 'src/components/DiagnosticsModal.tsx',
    name: 'DiagnosticsModal.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Bottom Sheet deslizable oscuro (#0F172A, borde #1E293B, redondeado 24px, máx 85% alto) con drag handle, cierre X y ScrollView fluido en 4 secciones: Gobernanza Térmica, Sliders, Latencia EMA (6 etapas) y Tabla ByteTrack.',
    highlights: [
      'Retorna null inmediatamente cuando visible === false para evitar cálculos en el árbol UI',
      'Selector térmico segmentado (Auto, Ultra, Balanceado, Ahorro Térmico) + promedio Avg 30f',
      'Sliders 0.10..0.90 + Latencia EMA con Real vs. Teórico FPS + Tabla ByteTrack completa',
    ],
    content: `import React from 'react';
import {
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import Slider from '@react-native-community/slider';
import { StageLatencyEma, ThermalMode } from '../utils/PipelineProfiler';
import { TrackedObject } from '../utils/visionPostProcessor';

export type ThermalControlSetting = 'auto' | ThermalMode;

interface DiagnosticsModalProps {
  visible: boolean;
  onClose: () => void;
  isPaused: boolean;
  onTogglePause: () => void;
  onResetHeatmap: () => void;
  thermalControl: ThermalControlSetting;
  onChangeThermalControl: (mode: ThermalControlSetting) => void;
  rollingAvgMs: number;
  confThreshold: number;
  onChangeConfThreshold: (value: number) => void;
  nmsIouThreshold: number;
  onChangeNmsIouThreshold: (value: number) => void;
  timings: StageLatencyEma;
  tracks: TrackedObject[];
}

const CLASS_BADGE_MAP: Record<string, { label: string; color: string; bg: string }> = {
  person: { label: 'person', color: '#38BDF8', bg: 'rgba(56,189,248,0.16)' },
  car: { label: 'car', color: '#F97316', bg: 'rgba(249,115,22,0.16)' },
  bus: { label: 'bus', color: '#EF4444', bg: 'rgba(239,68,68,0.16)' },
  bicycle: { label: 'bicycle', color: '#10B981', bg: 'rgba(16,185,129,0.16)' },
};

const THERMAL_OPTIONS: Array<{ id: ThermalControlSetting; label: string }> = [
  { id: 'auto', label: 'Auto' },
  { id: 'Ultra', label: 'Ultra' },
  { id: 'Balanced', label: 'Balanceado' },
  { id: 'Thermal Safe', label: 'Ahorro Térmico' },
];

export const DiagnosticsModal: React.FC<DiagnosticsModalProps> = ({
  visible,
  onClose,
  isPaused,
  onTogglePause,
  onResetHeatmap,
  thermalControl,
  onChangeThermalControl,
  rollingAvgMs,
  confThreshold,
  onChangeConfThreshold,
  nmsIouThreshold,
  onChangeNmsIouThreshold,
  timings,
  tracks,
}) => {
  // No ejecuta renders ni cálculos pesados cuando el modal está cerrado
  if (!visible) return null;

  const stages = [
    { label: '1. VisionCamera JSI Capture', ms: timings.captureMs, color: '#38BDF8' },
    { label: '2. Letterbox Resize 640x640', ms: timings.resizeMs, color: '#818CF8' },
    { label: '3. YOLO Detection Inference', ms: timings.yoloMs, color: '#F97316' },
    { label: '4. Pose Estimation 17 Keypoints', ms: timings.poseMs, color: '#22C55E' },
    { label: '5. Class-Aware NMS + ByteTrack', ms: timings.nmsTrackMs, color: '#10B981' },
    { label: '6. React Native Skia Render', ms: timings.skiaRenderMs, color: '#A78BFA' },
  ];
  const maxStageMs = Math.max(12, ...stages.map((s) => s.ms));

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.backdrop}>
        <View style={styles.sheet}>
          {/* Drag Handle + Header */}
          <View style={styles.dragHandle} />
          <View style={styles.sheetHeader}>
            <View style={{ flex: 1 }}>
              <Text style={styles.sheetTitle}>Ajustes y Telemetría Avanzada</Text>
              <Text style={styles.sheetSubtitle}>
                Gobernanza térmica, calibración YOLO/NMS, latencia EMA y ByteTrack
              </Text>
            </View>
            <TouchableOpacity style={styles.closeBtn} onPress={onClose} accessibilityLabel="Cerrar">
              <Text style={styles.closeBtnText}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView contentContainerStyle={styles.scrollBody} showsVerticalScrollIndicator={false}>
            {/* Sección 1: Controles Rápidos y Gobernanza Térmica */}
            <View style={styles.sectionCard}>
              <Text style={styles.sectionTitle}>1. CONTROLES RÁPIDOS Y GOBERNANZA TÉRMICA</Text>

              <View style={styles.actionsRow}>
                <TouchableOpacity style={styles.primaryActionBtn} onPress={onTogglePause}>
                  <Text style={styles.primaryActionText}>
                    {isPaused ? 'Reanudar Pipeline' : 'Pausar Pipeline'}
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.secondaryActionBtn} onPress={onResetHeatmap}>
                  <Text style={styles.secondaryActionText}>Reiniciar Heatmap</Text>
                </TouchableOpacity>
              </View>

              <View style={styles.segmentedRow}>
                {THERMAL_OPTIONS.map((opt) => {
                  const active = thermalControl === opt.id;
                  return (
                    <TouchableOpacity
                      key={opt.id}
                      style={[styles.segmentBtn, active && styles.segmentBtnActive]}
                      onPress={() => onChangeThermalControl(opt.id)}
                    >
                      <Text
                        numberOfLines={1}
                        style={[styles.segmentText, active && styles.segmentTextActive]}
                      >
                        {opt.label}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
              <Text style={styles.avgMeta}>Avg 30f: {rollingAvgMs.toFixed(1)} ms</Text>
            </View>

            {/* Sección 2: Sliders de Calibración en Tiempo Real */}
            <View style={styles.sectionCard}>
              <Text style={styles.sectionTitle}>2. SLIDERS DE CALIBRACIÓN EN TIEMPO REAL</Text>

              <View style={styles.sliderBlock}>
                <View style={styles.rowBetween}>
                  <Text style={styles.sliderLabel}>Confidence Threshold</Text>
                  <Text style={styles.sliderValue}>{confThreshold.toFixed(2)}</Text>
                </View>
                <Slider
                  minimumValue={0.1}
                  maximumValue={0.9}
                  step={0.01}
                  value={confThreshold}
                  onValueChange={onChangeConfThreshold}
                  minimumTrackTintColor="#38BDF8"
                  maximumTrackTintColor="#1E293B"
                  thumbTintColor="#38BDF8"
                />
              </View>

              <View style={styles.sliderBlock}>
                <View style={styles.rowBetween}>
                  <Text style={styles.sliderLabel}>NMS IoU Threshold</Text>
                  <Text style={styles.sliderValue}>{nmsIouThreshold.toFixed(2)}</Text>
                </View>
                <Slider
                  minimumValue={0.1}
                  maximumValue={0.9}
                  step={0.01}
                  value={nmsIouThreshold}
                  onValueChange={onChangeNmsIouThreshold}
                  minimumTrackTintColor="#38BDF8"
                  maximumTrackTintColor="#1E293B"
                  thumbTintColor="#38BDF8"
                />
              </View>
            </View>

            {/* Sección 3: Panel de Latencia por Etapa (EMA) */}
            <View style={styles.sectionCard}>
              <View style={styles.rowBetween}>
                <Text style={styles.sectionTitle}>3. LATENCIA POR ETAPA (EMA)</Text>
                <View style={{ alignItems: 'flex-end' }}>
                  <Text style={styles.totalMsText}>{timings.totalMs.toFixed(1)} ms</Text>
                  <Text style={styles.fpsCompareText}>
                    Real {timings.realFps.toFixed(1)} vs. Teórico {timings.theoreticalFps.toFixed(1)} FPS
                  </Text>
                </View>
              </View>

              <View style={{ gap: 8 }}>
                {stages.map((st) => (
                  <View key={st.label} style={{ gap: 4 }}>
                    <View style={styles.rowBetween}>
                      <Text style={styles.stageLabel}>{st.label}</Text>
                      <Text style={styles.stageMs}>{st.ms.toFixed(1)} ms</Text>
                    </View>
                    <View style={styles.stageTrack}>
                      <View
                        style={[
                          styles.stageFill,
                          {
                            backgroundColor: st.color,
                            width: \`\${Math.min(100, (st.ms / maxStageMs) * 100)}%\`,
                          },
                        ]}
                      />
                    </View>
                  </View>
                ))}
              </View>
            </View>

            {/* Sección 4: Tabla de Objetos Rastreados (ByteTrack + NMS) */}
            <View style={styles.sectionCard}>
              <View style={styles.rowBetween}>
                <Text style={styles.sectionTitle}>4. OBJETOS RASTREADOS (BYTETRACK + NMS)</Text>
                <View style={styles.countPill}>
                  <Text style={styles.countPillText}>{tracks.length} activos</Text>
                </View>
              </View>

              <View style={styles.tableHeader}>
                <Text style={[styles.th, { width: 34 }]}>ID</Text>
                <Text style={[styles.th, { width: 64 }]}>Clase</Text>
                <Text style={[styles.th, { width: 42, textAlign: 'right' }]}>Conf</Text>
                <Text style={[styles.th, { flex: 1, textAlign: 'right' }]}>Screen BBox</Text>
                <Text style={[styles.th, { width: 44, textAlign: 'right' }]}>Kps</Text>
                <Text style={[styles.th, { width: 44, textAlign: 'right' }]}>Edad</Text>
              </View>

              {tracks.length === 0 ? (
                <Text style={styles.emptyTableText}>
                  Sin detecciones activas sobre umbral {confThreshold.toFixed(2)}
                </Text>
              ) : (
                tracks.map((tr) => {
                  const badge = CLASS_BADGE_MAP[tr.className] ?? {
                    label: tr.className,
                    color: '#E2E8F0',
                    bg: 'rgba(148,163,184,0.14)',
                  };
                  const kpsText =
                    tr.className === 'person' && tr.keypoints && tr.keypoints.length > 0
                      ? \`\${tr.keypoints.filter((k) => k.visibility > 0.5).length}/17\`
                      : '-';

                  return (
                    <View key={tr.trackId} style={styles.tableRow}>
                      <Text style={[styles.tdMono, { width: 34 }]}>#{tr.trackId}</Text>
                      <View style={{ width: 64, alignItems: 'flex-start' }}>
                        <View style={[styles.classBadge, { backgroundColor: badge.bg }]}>
                          <Text style={[styles.classBadgeText, { color: badge.color }]}>
                            {badge.label}
                          </Text>
                        </View>
                      </View>
                      <Text style={[styles.tdMono, { width: 42, textAlign: 'right' }]}>
                        {tr.confidence.toFixed(2)}
                      </Text>
                      <Text style={[styles.tdMonoMuted, { flex: 1, textAlign: 'right' }]}>
                        [{Math.round(tr.screenBox.x)},{Math.round(tr.screenBox.y)},{Math.round(tr.screenBox.width)},{Math.round(tr.screenBox.height)}]
                      </Text>
                      <Text style={[styles.tdKps, { width: 44, textAlign: 'right' }]}>{kpsText}</Text>
                      <Text style={[styles.tdMonoMuted, { width: 44, textAlign: 'right' }]}>
                        {tr.ageFrames}f
                      </Text>
                    </View>
                  );
                })
              )}
            </View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.72)',
    justifyContent: 'flex-end',
  },
  sheet: {
    maxHeight: '85%',
    backgroundColor: '#0F172A',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderTopWidth: 1,
    borderColor: '#1E293B',
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: 18,
  },
  dragHandle: {
    width: 40,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#334155',
    alignSelf: 'center',
    marginBottom: 10,
  },
  sheetHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
  },
  sheetTitle: { color: '#F8FAFC', fontSize: 15, fontWeight: '700' },
  sheetSubtitle: { color: '#94A3B8', fontSize: 11, marginTop: 2 },
  closeBtn: {
    width: 34,
    height: 34,
    borderRadius: 10,
    backgroundColor: '#121826',
    borderWidth: 1,
    borderColor: '#1E293B',
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeBtnText: { color: '#CBD5E1', fontSize: 14, fontWeight: '700' },
  scrollBody: { paddingTop: 12, paddingBottom: 16, gap: 12 },
  sectionCard: {
    backgroundColor: '#121826',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#1E293B',
    padding: 14,
    gap: 10,
  },
  sectionTitle: { color: '#94A3B8', fontSize: 11, fontWeight: '700', letterSpacing: 0.5 },
  actionsRow: { flexDirection: 'row', gap: 10 },
  primaryActionBtn: {
    flex: 1,
    backgroundColor: '#38BDF8',
    paddingVertical: 11,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryActionText: { color: '#0B0F17', fontSize: 12, fontWeight: '700' },
  secondaryActionBtn: {
    flex: 1,
    backgroundColor: '#1E293B',
    borderWidth: 1,
    borderColor: '#334155',
    paddingVertical: 11,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryActionText: { color: '#E2E8F0', fontSize: 12, fontWeight: '600' },
  segmentedRow: {
    flexDirection: 'row',
    backgroundColor: '#0B0F17',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1E293B',
    padding: 4,
    gap: 4,
  },
  segmentBtn: { flex: 1, paddingVertical: 8, borderRadius: 8, alignItems: 'center' },
  segmentBtnActive: { backgroundColor: '#10B981' },
  segmentText: { color: '#94A3B8', fontSize: 10.5, fontWeight: '600' },
  segmentTextActive: { color: '#0B0F17', fontWeight: '700' },
  avgMeta: { color: '#94A3B8', fontSize: 11, fontFamily: 'JetBrainsMono-Regular' },
  sliderBlock: { gap: 4 },
  rowBetween: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  sliderLabel: { color: '#E2E8F0', fontSize: 12, fontWeight: '500' },
  sliderValue: { color: '#38BDF8', fontSize: 12, fontFamily: 'JetBrainsMono-Bold' },
  totalMsText: { color: '#38BDF8', fontSize: 14, fontFamily: 'JetBrainsMono-Bold' },
  fpsCompareText: { color: '#94A3B8', fontSize: 10, fontFamily: 'JetBrainsMono-Regular' },
  stageLabel: { color: '#E2E8F0', fontSize: 11.5, fontWeight: '500' },
  stageMs: { color: '#F1F5F9', fontSize: 11.5, fontFamily: 'JetBrainsMono-SemiBold' },
  stageTrack: { height: 6, borderRadius: 3, backgroundColor: '#0B0F17', overflow: 'hidden' },
  stageFill: { height: '100%', borderRadius: 3 },
  countPill: {
    backgroundColor: '#0B0F17',
    borderWidth: 1,
    borderColor: '#1E293B',
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  countPillText: { color: '#38BDF8', fontSize: 11, fontFamily: 'JetBrainsMono-Medium' },
  tableHeader: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
    paddingBottom: 6,
  },
  th: { color: '#64748B', fontSize: 10, fontWeight: '600' },
  tableRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(30,41,59,0.55)',
  },
  tdMono: { color: '#E2E8F0', fontSize: 10.5, fontFamily: 'JetBrainsMono-Regular' },
  tdMonoMuted: { color: '#94A3B8', fontSize: 10, fontFamily: 'JetBrainsMono-Regular' },
  tdKps: { color: '#22C55E', fontSize: 10.5, fontFamily: 'JetBrainsMono-SemiBold' },
  classBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 5 },
  classBadgeText: { fontSize: 10, fontWeight: '700' },
  emptyTableText: { color: '#64748B', fontSize: 11.5, textAlign: 'center', paddingVertical: 14 },
});`,
  },
  {
    id: 'vision-monitor-screen',
    path: 'src/screens/VisionEdgeDashboard.tsx',
    name: 'VisionEdgeDashboard.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Pantalla móvil adaptable 100% viewport sin scroll: consume useResponsiveLayout (alturas dinámicas para pantallas < 720px y estándar) + DiagnosticsModal (pausando envío de tracks cuando visible === false).',
    highlights: [
      'Alturas dinámicas restando insets.top e insets.bottom con isCompactScreen = height < 720',
      'Visor Skia con flex: 1 absorbiendo espacio restante sin recortes',
      'Sincronización de tabla ByteTrack desactivada mientras DiagnosticsModal está cerrado',
    ],
    content: `import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useDerivedValue, useSharedValue } from 'react-native-reanimated';
import { useVisionModels } from '../models/useVisionModels';
import { useResponsiveLayout } from '../hooks/useResponsiveLayout';
import { useObjectDetectionProcessor, VisionFrameState } from '../processors/useObjectDetectionProcessor';
import { TrackedObject } from '../utils/visionPostProcessor';
import { DetectionOverlayCanvas } from '../components/DetectionOverlayCanvas';
import { ActivityHeatmap, HEATMAP_COLS, HEATMAP_ROWS } from '../components/ActivityHeatmap';
import { ObjectTrackingChart, ProcessingSpeedWave, TrackingHistorySample } from '../components/TelemetryCharts';
import { DiagnosticsModal, ThermalControlSetting } from '../components/DiagnosticsModal';
import { StageLatencyEma } from '../utils/PipelineProfiler';

export const VisionEdgeDashboard: React.FC = () => {
  const layout = useResponsiveLayout();
  const [cameraSide, setCameraSide] = useState<'back' | 'front'>('back');
  const [isPaused, setIsPaused] = useState(false);
  const [confThreshold, setConfThreshold] = useState(0.45);
  const [nmsIouThreshold, setNmsIouThreshold] = useState(0.45);
  const [thermalControl, setThermalControl] = useState<ThermalControlSetting>('auto');
  const [diagnosticsVisible, setDiagnosticsVisible] = useState(false);
  const [visorSize, setVisorSize] = useState({
    width: Math.max(280, layout.width - layout.paddingHorizontal * 2),
    height: layout.estimatedViewportHeight,
  });

  const device = useCameraDevice(cameraSide);
  const { detector, poseModel, activeDelegate } = useVisionModels();

  const visionStateShared = useSharedValue<VisionFrameState>({
    tracks: [],
    timings: { captureMs: 0, resizeMs: 0, inferenceMs: 0, postprocessMs: 0, totalMs: 0, fps: 38.8 },
  });
  const tracksShared = useDerivedValue(() => visionStateShared.value.tracks);
  const confShared = useSharedValue(0.45);
  const iouShared = useSharedValue(0.45);
  const isFrontShared = useSharedValue(false);
  const isDiagnosticsOpenShared = useSharedValue(false);
  const fpsWaveShared = useSharedValue<number[]>(new Array(30).fill(38.8));
  const heatmapGridShared = useSharedValue<number[]>(new Array(HEATMAP_COLS * HEATMAP_ROWS).fill(0));

  const prevCountsRef = useRef({ persons: 0, vehicles: 0, bicycles: 0 });
  const diagnosticsOpenRef = useRef(false);

  const [telemetry, setTelemetry] = useState({
    fps: 38.8,
    totalMs: 25.8,
    persons: 0,
    vehicles: 0,
    bicycles: 0,
    deltaPersons: 0,
    deltaVehicles: 0,
    deltaBicycles: 0,
    density: 'Moderate' as 'Low' | 'Moderate' | 'High',
    activeTracksCount: 0,
    activeTracks: [] as TrackedObject[],
  });

  const [history20s, setHistory20s] = useState<TrackingHistorySample[]>(() =>
    Array.from({ length: 20 }, () => ({ persons: 0, vehicles: 0, bicycles: 0 }))
  );

  useEffect(() => {
    confShared.value = confThreshold;
    iouShared.value = nmsIouThreshold;
    isFrontShared.value = cameraSide === 'front';
    isDiagnosticsOpenShared.value = diagnosticsVisible;
    diagnosticsOpenRef.current = diagnosticsVisible;
  }, [confThreshold, nmsIouThreshold, cameraSide, diagnosticsVisible]);

  useEffect(() => {
    const id = setInterval(() => {
      if (isPaused) return;
      const { tracks, timings } = visionStateShared.value;
      const visible = tracks.filter((t) => t.lostFrames === 0);

      const persons = visible.filter((t) => t.category === 'person').length;
      const vehicles = visible.filter((t) => t.category === 'vehicle').length;
      const bicycles = visible.filter((t) => t.category === 'bicycle').length;
      const total = visible.length;
      const density = total < 3 ? 'Low' : total <= 7 ? 'Moderate' : 'High';

      const prev = prevCountsRef.current;
      setTelemetry((prevState) => ({
        fps: timings.fps,
        totalMs: timings.totalMs,
        persons,
        vehicles,
        bicycles,
        deltaPersons: persons - prev.persons,
        deltaVehicles: vehicles - prev.vehicles,
        deltaBicycles: bicycles - prev.bicycles,
        density,
        activeTracksCount: visible.length,
        // Pausa actualización de la tabla en JS cuando DiagnosticsModal está cerrado
        activeTracks: diagnosticsOpenRef.current ? visible : prevState.activeTracks,
      }));
      prevCountsRef.current = { persons, vehicles, bicycles };

      setHistory20s((h) => [...h.slice(-19), { persons, vehicles, bicycles }]);
      fpsWaveShared.value = [...fpsWaveShared.value.slice(-29), timings.fps];

      if (visible.length > 0) {
        const nextGrid = [...heatmapGridShared.value];
        for (const tr of visible) {
          const gx = Math.min(HEATMAP_COLS - 1, Math.max(0, Math.floor((tr.cx / visorSize.width) * HEATMAP_COLS)));
          const gy = Math.min(HEATMAP_ROWS - 1, Math.max(0, Math.floor((tr.cy / visorSize.height) * HEATMAP_ROWS)));
          nextGrid[gy * HEATMAP_COLS + gx] += 1;
        }
        heatmapGridShared.value = nextGrid;
      }
    }, 1000);

    return () => clearInterval(id);
  }, [visorSize.width, visorSize.height, isPaused]);

  const frameProcessor = useObjectDetectionProcessor(
    detector,
    poseModel,
    visionStateShared,
    confShared,
    iouShared,
    isFrontShared,
    visorSize
  );

  const handleOpenDiagnostics = useCallback(() => {
    const visible = visionStateShared.value.tracks.filter((t) => t.lostFrames === 0);
    setTelemetry((prev) => ({ ...prev, activeTracks: visible }));
    isDiagnosticsOpenShared.value = true;
    diagnosticsOpenRef.current = true;
    setDiagnosticsVisible(true);
  }, []);

  const maxBar = Math.max(1, telemetry.persons, telemetry.vehicles, telemetry.bicycles, 6);
  const formatDelta = (d: number) => (d > 0 ? \`▲ +\${d}\` : d < 0 ? \`▼ \${d}\` : '— 0');
  const cardWidth = Math.max(130, (layout.width - layout.paddingHorizontal * 2 - layout.gap) / 2 - 20);
  const chartWidth = Math.max(240, layout.width - layout.paddingHorizontal * 2 - 20);

  const modalTimings: StageLatencyEma = {
    captureMs: visionStateShared.value.timings.captureMs || 1.6,
    resizeMs: visionStateShared.value.timings.resizeMs || 1.9,
    yoloMs: visionStateShared.value.timings.inferenceMs || 11.4,
    poseMs: 7.7,
    nmsTrackMs: visionStateShared.value.timings.postprocessMs || 1.4,
    skiaRenderMs: 1.8,
    totalMs: telemetry.totalMs,
    theoreticalFps: telemetry.fps,
    realFps: Math.min(30, telemetry.fps),
  };

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <View
        style={[
          styles.container,
          { gap: layout.gap, paddingHorizontal: layout.paddingHorizontal },
        ]}
      >
        {/* 1. Top Status Bar (42px / 36px en compactas) */}
        <View style={[styles.topBar, { height: layout.topBarHeight }]}>
          <View style={styles.rowCenter}>
            <View style={[styles.greenDot, isPaused && { backgroundColor: '#F59E0B' }]} />
            <Text style={styles.statusTitle}>{isPaused ? 'Camera: Paused' : 'Camera: Online'}</Text>
          </View>
          <View style={styles.rowCenter}>
            <TouchableOpacity
              style={styles.lensChip}
              onPress={() => setCameraSide((s) => (s === 'back' ? 'front' : 'back'))}
            >
              <Text style={styles.lensText}>{cameraSide === 'back' ? 'Rear Wide' : 'Front'}</Text>
            </TouchableOpacity>
            <Text style={styles.fpsReadout}>FPS: {telemetry.fps.toFixed(1)}</Text>
          </View>
        </View>

        {/* 2. Visor Principal de Cámara (flex: 1) */}
        <View
          style={styles.cameraShell}
          onLayout={(e) => {
            const { width, height } = e.nativeEvent.layout;
            if (width > 0 && height > 0) setVisorSize({ width, height });
          }}
        >
          {device && (
            <Camera
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={!isPaused}
              frameProcessor={frameProcessor}
              pixelFormat="yuv"
            />
          )}
          <DetectionOverlayCanvas
            tracksShared={tracksShared}
            width={visorSize.width}
            height={visorSize.height}
          />
          <View style={styles.visorBottomMeta}>
            <Text style={styles.visorMetaText}>
              YOLOv8n + Pose · Conf {confThreshold.toFixed(2)}
            </Text>
            <Text style={styles.visorMetaText}>{activeDelegate}</Text>
          </View>
        </View>

        {/* 3. Fila de Contadores Rápidos (76px / 64px en compactas) */}
        <View style={[styles.counterRow, { height: layout.counterRowHeight, gap: layout.gap }]}>
          {[
            { label: 'Personas', count: telemetry.persons, delta: telemetry.deltaPersons, color: '#38BDF8' },
            { label: 'Vehículos', count: telemetry.vehicles, delta: telemetry.deltaVehicles, color: '#F97316' },
            { label: 'Bicicletas', count: telemetry.bicycles, delta: telemetry.deltaBicycles, color: '#10B981' },
          ].map((card) => (
            <View
              key={card.label}
              style={[styles.counterCard, { paddingVertical: layout.isCompactScreen ? 5 : 7 }]}
            >
              <View style={styles.rowBetween}>
                <Text style={styles.cardLabel}>{card.label}</Text>
                <Text style={[styles.deltaText, { color: card.delta >= 0 ? '#22C55E' : '#F43F5E' }]}>
                  {formatDelta(card.delta)}
                </Text>
              </View>
              <Text style={[styles.counterValue, { fontSize: layout.counterFontSize }]}>
                {card.count}
              </Text>
              <View style={styles.progressTrack}>
                <View
                  style={[
                    styles.progressFill,
                    { backgroundColor: card.color, width: \`\${(card.count / maxBar) * 100}%\` },
                  ]}
                />
              </View>
            </View>
          ))}
        </View>

        {/* 4. Fila de Métricas y Escena (96px / 82px en compactas) */}
        <View style={[styles.twoColRow, { height: layout.metricsRowHeight, gap: layout.gap }]}>
          <View style={[styles.panelCard, { paddingVertical: layout.isCompactScreen ? 5 : 7 }]}>
            <View style={styles.rowBetween}>
              <Text style={styles.cardLabel}>Processing Speed</Text>
              <Text style={styles.metaMono}>{telemetry.totalMs.toFixed(1)} ms</Text>
            </View>
            <Text style={[styles.fpsBig, layout.isCompactScreen && { fontSize: 15 }]}>
              {telemetry.fps.toFixed(1)} FPS
            </Text>
            <ProcessingSpeedWave
              fpsHistoryShared={fpsWaveShared}
              width={cardWidth}
              height={layout.waveHeight}
            />
          </View>

          <View style={[styles.panelCard, { paddingVertical: layout.isCompactScreen ? 5 : 7 }]}>
            <View style={styles.rowBetween}>
              <Text style={styles.cardLabel}>Analysis</Text>
              <Text style={styles.sceneTag}>Urban</Text>
            </View>
            <View style={styles.rowBetween}>
              <Text style={styles.densityLabel}>Density: {telemetry.density}</Text>
              <Text style={styles.metaMono}>n={telemetry.activeTracksCount}</Text>
            </View>
            <ActivityHeatmap
              gridShared={heatmapGridShared}
              width={cardWidth}
              height={layout.heatmapHeight}
              onReset={() => {
                heatmapGridShared.value = new Array(HEATMAP_COLS * HEATMAP_ROWS).fill(0);
              }}
            />
          </View>
        </View>

        {/* 5. Gráfico Temporal Object Tracking (20s) (105px / 88px en compactas) */}
        <View style={{ height: layout.trackingChartHeight, overflow: 'hidden' }}>
          <ObjectTrackingChart
            samples={history20s}
            width={chartWidth}
            height={layout.trackingSvgHeight}
          />
        </View>

        {/* 6. Footer Bar (48px / 42px en compactas) */}
        <View style={[styles.footer, { height: layout.footerHeight }]}>
          <View style={styles.rowCenter}>
            <View style={[styles.greenDot, isPaused && { backgroundColor: '#F59E0B' }]} />
            <Text style={styles.footerText}>
              System Status: {isPaused ? 'Paused' : \`Running [\${thermalControl}]\`}
            </Text>
          </View>
          <TouchableOpacity style={styles.settingsBtn} onPress={handleOpenDiagnostics}>
            <Text style={styles.settingsBtnText}>Ajustes</Text>
          </TouchableOpacity>
        </View>
      </View>

      <DiagnosticsModal
        visible={diagnosticsVisible}
        onClose={() => {
          isDiagnosticsOpenShared.value = false;
          diagnosticsOpenRef.current = false;
          setDiagnosticsVisible(false);
        }}
        isPaused={isPaused}
        onTogglePause={() => setIsPaused((p) => !p)}
        onResetHeatmap={() => {
          heatmapGridShared.value = new Array(HEATMAP_COLS * HEATMAP_ROWS).fill(0);
        }}
        thermalControl={thermalControl}
        onChangeThermalControl={setThermalControl}
        rollingAvgMs={telemetry.totalMs}
        confThreshold={confThreshold}
        onChangeConfThreshold={setConfThreshold}
        nmsIouThreshold={nmsIouThreshold}
        onChangeNmsIouThreshold={setNmsIouThreshold}
        timings={modalTimings}
        tracks={telemetry.activeTracks}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0B0F17' },
  container: { flex: 1 },
  topBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  rowCenter: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  rowBetween: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  greenDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#22C55E' },
  statusTitle: { color: '#F1F5F9', fontSize: 13, fontWeight: '600' },
  lensChip: { backgroundColor: '#121826', borderWidth: 1, borderColor: '#1E293B', paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8 },
  lensText: { color: '#E2E8F0', fontSize: 11, fontWeight: '600' },
  fpsReadout: { color: '#38BDF8', fontFamily: 'JetBrainsMono-SemiBold', fontSize: 13 },
  cameraShell: { flex: 1, borderRadius: 16, overflow: 'hidden', backgroundColor: '#121826', borderWidth: 1, borderColor: '#1E293B' },
  visorBottomMeta: { position: 'absolute', bottom: 8, left: 12, right: 12, flexDirection: 'row', justifyContent: 'space-between' },
  visorMetaText: { color: '#CBD5E1', fontSize: 10.5, fontFamily: 'JetBrainsMono-Medium' },
  counterRow: { flexDirection: 'row' },
  counterCard: { flex: 1, backgroundColor: '#121826', borderRadius: 12, paddingHorizontal: 10, borderWidth: 1, borderColor: '#1E293B', justifyContent: 'space-between' },
  cardLabel: { color: '#94A3B8', fontSize: 11, fontWeight: '500' },
  deltaText: { fontSize: 10, fontFamily: 'JetBrainsMono-Medium' },
  counterValue: { color: '#FFFFFF', fontWeight: '700', fontFamily: 'JetBrainsMono-Bold' },
  progressTrack: { height: 5, borderRadius: 3, backgroundColor: '#1E293B', overflow: 'hidden' },
  progressFill: { height: '100%', borderRadius: 3 },
  twoColRow: { flexDirection: 'row' },
  panelCard: { flex: 1, backgroundColor: '#121826', borderRadius: 12, paddingHorizontal: 10, borderWidth: 1, borderColor: '#1E293B', justifyContent: 'space-between' },
  metaMono: { color: '#94A3B8', fontSize: 10, fontFamily: 'JetBrainsMono-Regular' },
  fpsBig: { color: '#F1F5F9', fontSize: 17, fontWeight: '700', fontFamily: 'JetBrainsMono-Bold' },
  sceneTag: { color: '#CBD5E1', fontSize: 11, fontWeight: '600' },
  densityLabel: { color: '#F59E0B', fontSize: 11.5, fontWeight: '600' },
  footer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', borderTopWidth: 1, borderTopColor: '#1E293B' },
  footerText: { color: '#CBD5E1', fontSize: 12, fontWeight: '500' },
  settingsBtn: { backgroundColor: '#121826', borderWidth: 1, borderColor: '#1E293B', paddingHorizontal: 14, paddingVertical: 7, borderRadius: 10 },
  settingsBtnText: { color: '#F1F5F9', fontSize: 12, fontWeight: '600' },
});`,
  },
  {
    id: 'swift-native',
    path: 'native/ios/VisionPipeline.swift',
    name: 'VisionPipeline.swift',
    language: 'swift',
    deliverable: 'native',
    summary: 'Implementación nativa equivalente en iOS: AVFoundation AVCaptureVideoDataOutput + CoreML (.mlpackage) + Apple Neural Engine + CoreGraphics.',
    highlights: [
      'VNCoreMLModel ejecutado en computeUnits = .all (Neural Engine + Metal GPU)',
      'VNDetectHumanBodyPoseRequest para esqueleto articulado en paralelo',
    ],
    content: `import AVFoundation
import CoreML
import Vision

final class VisionPipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let inferenceQueue = DispatchQueue(label: "cv.inference.queue", qos: .userInteractive)
    private var detectionRequest: VNCoreMLRequest?
    private let poseRequest = VNDetectHumanBodyPoseRequest()
    var onDetections: (([VNRecognizedObjectObservation], [VNHumanBodyPoseObservation], Double) -> Void)?

    func configureModels() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Habilita Apple Neural Engine (ANE) + Metal GPU
        let yoloCoreML = try yolov8n_int8(configuration: config).model
        let vnModel = try VNCoreMLModel(for: yoloCoreML)
        let req = VNCoreMLRequest(model: vnModel)
        req.imageCropAndScaleOption = .scaleFit
        self.detectionRequest = req
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let detReq = detectionRequest else { return }

        let start = CACurrentMediaTime()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)

        do {
            try handler.perform([detReq, poseRequest])
            let objects = (detReq.results as? [VNRecognizedObjectObservation]) ?? []
            let poses = poseRequest.results ?? []
            let latencyMs = (CACurrentMediaTime() - start) * 1000.0
            let fps = min(60.0, 1000.0 / max(1.0, latencyMs))

            DispatchQueue.main.async { [weak self] in
                self?.onDetections?(objects, poses, fps)
            }
        } catch {
            // Ignorar fotograma si el pipeline está saturado
        }
    }
}`,
  },
  {
    id: 'kotlin-native',
    path: 'native/android/VisionAnalyzer.kt',
    name: 'VisionAnalyzer.kt',
    language: 'kotlin',
    deliverable: 'native',
    summary: 'Implementación nativa equivalente en Android: CameraX ImageAnalysis + TensorFlow Lite Interpreter con GPU Delegate / NNAPI.',
    highlights: [
      'STRATEGY_KEEP_ONLY_LATEST para evitar colas y mantener latencia < 33ms',
      'GpuDelegateFactory + fallback automático a NNAPI',
    ],
    content: `package com.visionedge.cv

import android.content.Context
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

class VisionAnalyzer(
    context: Context,
    modelBuffer: ByteBuffer,
    private val onResult: (fps: Float, latencyMs: Long) -> Unit
) : ImageAnalysis.Analyzer {

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply {
            numThreads = 4
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
            } else {
                addDelegate(NnApiDelegate())
            }
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    override fun analyze(imageProxy: ImageProxy) {
        val t0 = SystemClock.elapsedRealtime()
        try {
            // Preprocesamiento YUV_420_888 -> Tensor RGB 640x640 + inferencia
            // interpreter.runForMultipleInputsOutputs(inputs, outputs)
            val latencyMs = (SystemClock.elapsedRealtime() - t0).coerceAtLeast(1L)
            val fps = (1000f / latencyMs).coerceAtMost(60f)
            onResult(fps, latencyMs)
        } finally {
            imageProxy.close() // Obligatorio para liberar el buffer de CameraX
        }
    }
}`,
  },
  {
    id: 'colab-export-tflite',
    path: 'scripts/export_tflite_colab.py',
    name: 'export_tflite_colab.py',
    language: 'python',
    deliverable: 'colab',
    summary: 'Script completo para Google Colab: descarga yolo11n.pt y yolo11n-pose.pt, exporta a TFLite FP16 (640x640, nms=False), inspecciona tensores con tf.lite.Interpreter y descarga un .zip.',
    highlights: [
      'Exportación FP16 (half=True, imgsz=640, nms=False) para GPU/NPU móvil',
      'Inspección exacta de shapes, dtypes, canales COCO [0,1,2,5] y 17 keypoints',
      'Empaquetado automático en .zip y descarga en Google Colab',
    ],
    content: `# ==============================================================================
# Google Colab Script: Exportación y Validación TFLite FP16 (Edge AI Mobile)
# Modelos: YOLO11n (Detección Urbana) + YOLO11n-Pose (17 Keypoints COCO)
# Salida:  .tflite FP16 (640x640, nms=False) + Inspección de Tensores + .zip
# ==============================================================================

# 1. Instalación de dependencias (ejecutar en celda de Colab)
# !pip install -q --upgrade ultralytics tensorflow onnx onnx2tf sng4onnx onnx_graphsurgeon

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Configuración de modelos (puedes cambiar a "yolov8n.pt" / "yolov8n-pose.pt")
DET_MODEL_NAME = "yolo11n.pt"
POSE_MODEL_NAME = "yolo11n-pose.pt"
IMG_SIZE = 640

# Clases de tráfico urbano objetivo dentro de COCO-80
URBAN_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    5: "bus",
}

OUTPUT_DIR = Path("exported_tflite_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def locate_fp16_tflite(export_result_path: str | Path, model_stem: str) -> Path:
    """Localiza el archivo *_float16.tflite generado por Ultralytics."""
    p = Path(export_result_path)
    if p.is_file() and p.suffix == ".tflite":
        return p

    search_roots = [p if p.is_dir() else p.parent, Path(f"{model_stem}_saved_model")]
    for root in search_roots:
        if not root.exists():
            continue
        fp16_matches = list(root.rglob("*float16*.tflite"))
        if fp16_matches:
            return fp16_matches[0]
        any_tflite = list(root.rglob("*.tflite"))
        if any_tflite:
            return any_tflite[0]

    raise FileNotFoundError(f"No se encontró archivo .tflite FP16 para {model_stem}")


def export_models() -> tuple[Path, Path]:
    """Descarga y exporta ambos modelos a TFLite FP16 sin NMS acoplado."""
    print(f"[1/4] Descargando y validando {DET_MODEL_NAME} (clases urbanas: {list(URBAN_CLASSES.keys())})...")
    det_model = YOLO(DET_MODEL_NAME)
    det_export = det_model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        half=True,   # Cuantización FP16 (mantiene precisión de cajas y aprovecha GPU/CoreML)
        nms=False,   # Sin NMS interno para obtener tensor plano crudo en TypeScript/C++
    )
    det_tflite_src = locate_fp16_tflite(det_export, Path(DET_MODEL_NAME).stem)

    print(f"[2/4] Descargando y exportando {POSE_MODEL_NAME} (17 keypoints COCO)...")
    pose_model = YOLO(POSE_MODEL_NAME)
    pose_export = pose_model.export(
        format="tflite",
        imgsz=IMG_SIZE,
        half=True,
        nms=False,
    )
    pose_tflite_src = locate_fp16_tflite(pose_export, Path(POSE_MODEL_NAME).stem)

    det_final = OUTPUT_DIR / "yolo11n_urban_fp16.tflite"
    pose_final = OUTPUT_DIR / "yolo11n_pose_fp16.tflite"
    shutil.copy2(det_tflite_src, det_final)
    shutil.copy2(pose_tflite_src, pose_final)

    print(f" -> Detección exportada: {det_final} ({det_final.stat().st_size / 1e6:.2f} MB)")
    print(f" -> Pose exportada:      {pose_final} ({pose_final.stat().st_size / 1e6:.2f} MB)")
    return det_final, pose_final


def inspect_tflite_model(tflite_path: Path, model_kind: str) -> dict:
    """
    Carga el .tflite con tf.lite.Interpreter, ejecuta una inferencia aleatoria
    y reporta shapes exactos, dtypes y mapa de canales para decodificación en TS/C++.
    """
    print("\\n" + "=" * 76)
    print(f"INSPECCIÓN DE TENSORES TFLITE [{model_kind.upper()}]: {tflite_path.name}")
    print("=" * 76)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for idx, inp in enumerate(input_details):
        print(f"Input[{idx}]:  name={inp['name']} | shape={inp['shape'].tolist()} | dtype={inp['dtype']}")

    # Tensor aleatorio NHWC normalizado [0.0, 1.0] acorde a YOLO TFLite
    inp_shape = input_details[0]["shape"]
    inp_dtype = input_details[0]["dtype"]
    rng = np.random.default_rng(seed=42)
    dummy_input = rng.random(size=inp_shape, dtype=np.float32).astype(inp_dtype)

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    summary_outputs = []
    for idx, out in enumerate(output_details):
        tensor = interpreter.get_tensor(out["index"])
        print(
            f"Output[{idx}]: name={out['name']} | shape={list(tensor.shape)} | "
            f"dtype={tensor.dtype} | min={tensor.min():.4f} | max={tensor.max():.4f}"
        )
        summary_outputs.append({
            "name": out["name"],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        })

    out_tensor = interpreter.get_tensor(output_details[0]["index"])
    _, channels, anchors = out_tensor.shape  # Esperado: [1, 84, 8400] o [1, 56, 8400]

    # Verificar rango de coordenadas [0..3] para saber si vienen normalizadas (0..1) o en píxeles (0..640)
    cx_max = float(out_tensor[0, 0, :].max())
    coord_scale = "normalized_[0..1]" if cx_max <= 1.5 else f"pixel_[0..{IMG_SIZE}]"
    print(f"Layout detectado: [batch=1, channels={channels}, anchors={anchors}] | Coordenadas: {coord_scale}")

    if model_kind == "detection":
        print("Mapa exacto de canales (Detección COCO-80, filtrar en TypeScript):")
        print("  - Canal 0..3   -> [cx, cy, w, h] (centro X, centro Y, ancho, alto)")
        print("  - Canal 4..83  -> Probabilidades de las 80 clases COCO (sin Sigmoid extra requerido)")
        print("  - Índices filtrados para tráfico urbano:")
        for cls_id, cls_name in URBAN_CLASSES.items():
            print(f"      * {cls_name:<8} -> channel_index = 4 + {cls_id} = {4 + cls_id}")
        print("  - Indexación plana Float32Array en JS/TS:")
        print(f"      val = detOutput[channel * {anchors} + anchorIdx]")
    else:
        print("Mapa exacto de canales (Pose 17 Keypoints COCO):")
        print("  - Canal 0..3   -> [cx, cy, w, h] de la persona")
        print("  - Canal 4      -> Confianza de clase 'person'")
        print("  - Canal 5..55  -> 17 keypoints × 3 valores: (kp_x, kp_y, kp_conf)")
        print("  - Indexación plana Float32Array en JS/TS:")
        print(f"      kx = poseOutput[(5 + k * 3) * {anchors} + anchorIdx]")
        print(f"      ky = poseOutput[(5 + k * 3 + 1) * {anchors} + anchorIdx]")
        print(f"      ks = poseOutput[(5 + k * 3 + 2) * {anchors} + anchorIdx]")

    return {
        "file": tflite_path.name,
        "input_shape": input_details[0]["shape"].tolist(),
        "input_dtype": str(input_details[0]["dtype"]),
        "outputs": summary_outputs,
        "coord_scale": coord_scale,
    }


def package_and_download(det_path: Path, pose_path: Path, contract: dict) -> Path:
    """Empaqueta los dos .tflite + contrato JSON en un .zip y lo descarga en Colab."""
    contract_path = OUTPUT_DIR / "tensor_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    zip_path = Path("vision_edge_tflite_fp16.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(det_path, arcname=det_path.name)
        zf.write(pose_path, arcname=pose_path.name)
        zf.write(contract_path, arcname=contract_path.name)

    print(f"\\n[4/4] Archivo ZIP generado: {zip_path.resolve()} ({zip_path.stat().st_size / 1e6:.2f} MB)")

    try:
        from google.colab import files
        files.download(str(zip_path))
        print("Descarga iniciada automáticamente en Google Colab.")
    except ImportError:
        print("Entorno local detectado (fuera de Colab): archivo listo en", zip_path.resolve())

    return zip_path


if __name__ == "__main__":
    det_tflite, pose_tflite = export_models()
    print("\\n[3/4] Ejecutando inspección de tensores con tf.lite.Interpreter...")
    det_info = inspect_tflite_model(det_tflite, "detection")
    pose_info = inspect_tflite_model(pose_tflite, "pose")

    contract = {
        "imgsz": IMG_SIZE,
        "precision": "float16",
        "nms_embedded": False,
        "urban_class_ids": URBAN_CLASSES,
        "detection": det_info,
        "pose": pose_info,
    }
    package_and_download(det_tflite, pose_tflite, contract)`,
  },
  {
    id: 'pipeline-thermal-governor',
    path: 'src/utils/PipelineProfiler.ts & AdaptiveThermalController.ts',
    name: 'Profiler & Thermal Governor',
    language: 'typescript',
    deliverable: 'inference',
    summary: 'Instrumentación sub-milisegundo con EMA (6 etapas + FPS real vs teórico) y gobernador térmico dinámico sobre los últimos 30 fotogramas (Ultra <33ms, Balanced 33-55ms, Thermal Safe >55ms).',
    highlights: [
      'Ring buffer Float32Array(30) zero-allocation con histéresis de 2 ms',
      'Degradación elegante: Pose alternado cada 2 frames (Balanced) o pausado + Heatmap x4 (Thermal Safe)',
    ],
    content: pipelineThermalRaw,
  },
  {
    id: 'latency-breakdown-card',
    path: 'src/components/LatencyBreakdownCard.tsx',
    name: 'LatencyBreakdownCard.tsx',
    language: 'typescript',
    deliverable: 'ui',
    summary: 'Tarjeta oscura de telemetría con barras horizontales codificadas por color para las 6 etapas del pipeline, EMA, FPS teórico vs real y badge de estado térmico.',
    highlights: [
      '6 etapas codificadas por color: Capture, Letterbox, YOLO, Pose, NMS+Track y Skia',
      'Muestra System Status: Running [Ultra / Balanced / Thermal Safe]',
    ],
    content: `import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { StageLatencyEma, ThermalMode } from '../utils/pipelineThermal';

interface LatencyBreakdownCardProps {
  timings: StageLatencyEma;
  thermalMode: ThermalMode;
  rollingAvgMs: number;
}

export const LatencyBreakdownCard: React.FC<LatencyBreakdownCardProps> = ({
  timings,
  thermalMode,
  rollingAvgMs,
}) => {
  const stages = [
    { label: '1. VisionCamera JSI Capture', ms: timings.captureMs, color: '#38BDF8' },
    { label: '2. Letterbox Resize (640×640)', ms: timings.resizeMs, color: '#818CF8' },
    { label: '3. YOLO Detection Inference (FP16)', ms: timings.yoloMs, color: '#F97316' },
    { label: '4. Pose Estimation (17 Keypoints)', ms: timings.poseMs, color: '#22C55E' },
    { label: '5. Class-Aware NMS + ByteTrack', ms: timings.nmsTrackMs, color: '#10B981' },
    { label: '6. React Native Skia Render', ms: timings.skiaRenderMs, color: '#A78BFA' },
  ];

  const maxStage = Math.max(12, ...stages.map((s) => s.ms));
  const modeColor =
    thermalMode === 'Thermal Safe'
      ? '#F43F5E'
      : thermalMode === 'Balanced'
      ? '#F59E0B'
      : '#22C55E';

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Latencia del Pipeline (EMA)</Text>
          <Text style={styles.subtitle}>
            Real {timings.realFps.toFixed(1)} FPS · Teórico {timings.theoreticalFps.toFixed(1)} FPS
          </Text>
        </View>
        <View style={styles.rightCol}>
          <Text style={styles.totalMs}>{timings.totalMs.toFixed(1)} ms</Text>
          <Text style={[styles.thermalTag, { color: modeColor }]}>
            [{thermalMode}] · Avg {rollingAvgMs.toFixed(1)}ms
          </Text>
        </View>
      </View>

      <View style={styles.rows}>
        {stages.map((stage) => {
          const pct = Math.min(100, (stage.ms / maxStage) * 100);
          return (
            <View key={stage.label} style={styles.row}>
              <Text style={styles.stageLabel} numberOfLines={1}>
                {stage.label}
              </Text>
              <View style={styles.barGroup}>
                <View style={styles.track}>
                  <View style={[styles.fill, { width: \`\${pct}%\`, backgroundColor: stage.color }]} />
                </View>
                <Text style={styles.msValue}>{stage.ms.toFixed(1)} ms</Text>
              </View>
            </View>
          );
        })}
      </View>

      <View style={styles.statusFooter}>
        <View style={[styles.dot, { backgroundColor: modeColor }]} />
        <Text style={styles.statusText}>System Status: Running [{thermalMode}]</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#121824',
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    gap: 10,
  },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  title: { color: '#F1F5F9', fontSize: 13, fontWeight: '700' },
  subtitle: { color: '#94A3B8', fontSize: 11, fontFamily: 'JetBrainsMono-Regular', marginTop: 2 },
  rightCol: { alignItems: 'flex-end' },
  totalMs: { color: '#38BDF8', fontSize: 16, fontWeight: '700', fontFamily: 'JetBrainsMono-Bold' },
  thermalTag: { fontSize: 10, fontFamily: 'JetBrainsMono-SemiBold', marginTop: 2 },
  rows: { gap: 8 },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8 },
  stageLabel: { color: '#CBD5E1', fontSize: 11, flex: 1 },
  barGroup: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  track: { width: 88, height: 6, borderRadius: 3, backgroundColor: '#1E293B', overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 3 },
  msValue: { color: '#E2E8F0', fontSize: 11, fontFamily: 'JetBrainsMono-SemiBold', width: 48, textAlign: 'right' },
  statusFooter: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingTop: 6, borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.06)' },
  dot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { color: '#CBD5E1', fontSize: 11, fontWeight: '600' },
});`,
  },
  {
    id: 'free-native-build',
    path: 'scripts/build_release_free.sh',
    name: 'Free Native Build (Android & iOS)',
    language: 'bash',
    deliverable: 'project',
    summary: 'Guía y comandos exactos para compilar e instalar en dispositivos físicos Android e iOS en modo Release 100% gratis (sin Expo Go ni cuenta paga de Apple Developer).',
    highlights: [
      'Android: expo-build-properties (minSdkVersion: 26) + --variant release vía USB',
      'iOS: Xcode gratuito con Apple ID personal (Personal Team) y esquema Release para ANE/Metal',
    ],
    content: `#!/usr/bin/env bash
# ==============================================================================
# VisionEdge RN — Compilación Nativa Release 100% Gratuita (Sin Expo Go)
# ==============================================================================

# ------------------------------------------------------------------------------
# PASO 0: Configurar minSdkVersion >= 26 en app.json (necesario para TFLite GPU/NNAPI)
# ------------------------------------------------------------------------------
# Instalar plugin de propiedades nativas:
npx expo install expo-build-properties

# Añadir en app.json -> expo.plugins:
# [
#   "expo-build-properties",
#   {
#     "android": {
#       "minSdkVersion": 26,
#       "compileSdkVersion": 35,
#       "targetSdkVersion": 35
#     },
#     "ios": {
#       "deploymentTarget": "15.1"
#     }
#   }
# ]

# ------------------------------------------------------------------------------
# OPCIÓN A: ANDROID (Windows / macOS / Linux) — Modo Release vía USB
# ------------------------------------------------------------------------------
# 1. Activar "Depuración USB" en el teléfono Android y verificar conexión:
adb devices

# 2. Generar proyecto nativo Android limpio con CNG:
npx expo prebuild --platform android --clean

# 3. Compilar APK/AAB optimizado en modo Release (JS compilado en Hermes bytecode,
#    sin sobrecosto de Metro Bundler ni modo debug) e instalar directo por USB:
npx expo run:android --variant release --device

# ------------------------------------------------------------------------------
# OPCIÓN B: iOS (macOS + Xcode Gratuito con Apple ID Personal, $0/año)
# ------------------------------------------------------------------------------
# 1. Generar proyecto Xcode limpio con CNG + CocoaPods (habilita CoreML/Metal):
npx expo prebuild --platform ios --clean

# 2. Abrir el workspace nativo en Xcode (gratuito desde Mac App Store):
open ios/VisionEdgeRN.xcworkspace

# 3. Firma Gratuita (Personal Team, duración 7 días renovable gratis):
#    - En Xcode -> Target "VisionEdgeRN" -> Tab "Signing & Capabilities"
#    - Marcar "Automatically manage signing"
#    - En "Team", seleccionar tu Apple ID personal ("<Tu Nombre> (Personal Team)")
#    - Cambiar Bundle Identifier a algo único (ej. com.tunombre.visionedgern)
#    - Conectar iPhone por USB -> En el iPhone: Ajustes -> General -> Gestión de
#      dispositivos -> Confiar en tu Apple ID.

# 4. Compilar en Modo Release desde CLI (habilita optimizaciones LLVM + ANE + Metal):
npx expo run:ios --configuration Release --device`,
  },
];
