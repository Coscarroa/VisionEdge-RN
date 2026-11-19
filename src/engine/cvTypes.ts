export type TargetClass =
  | 'person'
  | 'car'
  | 'bus'
  | 'truck'
  | 'bicycle'
  | 'motorcycle'
  | 'dog'
  | 'backpack';

export type CategoryBucket = 'persons' | 'vehicles' | 'bicycles' | 'animalsOther';

export type ARVisualTheme = 'tactical' | 'native' | 'industrial';

export interface GyroStabilizationTelemetry {
  omegaX: number; // rad/s Pitch
  omegaY: number; // rad/s Yaw
  omegaZ: number; // rad/s Roll
  offsetX: number; // px inter-frame compensation X
  offsetY: number; // px inter-frame compensation Y
  activeHz: number;
  lastResetAgoMs: number;
}

export type CameraSourceMode = 'rear-crosswalk' | 'bike-boulevard' | 'webcam';

export type HardwareDelegate = 'coreml-metal' | 'nnapi-gpu' | 'cpu-xnnpack';

export type DensityLevel = 'Low' | 'Moderate' | 'High';

export interface Keypoint {
  index: number;
  name: string;
  x: number; // Screen px
  y: number; // Screen px
  score: number;
}

export interface NormalizedBox {
  cx: number; // 0..640 tensor space
  cy: number; // 0..640 tensor space
  w: number;  // 0..640 tensor space
  h: number;  // 0..640 tensor space
}

export interface RawProposal {
  id: string;
  classId: number;
  className: TargetClass;
  confidence: number;
  tensorBox: NormalizedBox;
  tensorKeypoints?: Array<{ x: number; y: number; score: number }>;
}

export interface ScreenBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionResult {
  classId: number;
  className: TargetClass;
  confidence: number;
  screenBox: ScreenBox;
  keypoints: Keypoint[];
}

export type VehicleMotionStatus = 'EN MOVIMIENTO' | 'DETENIDO';

export interface TrackedObject extends DetectionResult {
  trackId: number;
  age: number;
  lostFrames: number;
  vx: number;
  vy: number;
  centerX: number;
  centerY: number;
  rawClassName?: string;
  speedKmh?: number;
  motionStatus?: VehicleMotionStatus;
  isOverspeed?: boolean;
  hasCrossedLine?: boolean;
  crossingDirection?: 'ENTRADA' | 'SALIDA' | null;
  /** Indica si el centroide del vehículo se encuentra dentro del Polígono ROI */
  insideRoi?: boolean;
  /** Segundos acumulados detenido dentro del Polígono ROI */
  roiDwellSeconds?: number;
  /** Alerta activa de invasión / detención prohibida en Polígono ROI */
  isRoiViolation?: boolean;
  /** Matrícula leída por ALPR (tesseract.js Web Worker) */
  licensePlate?: string;
  /** Indica si se detectó posible caída humana por análisis biomecánico de 17 keypoints */
  isFallDetected?: boolean;
  /** Segundos en postura horizontal inmóvil */
  fallDurationSeconds?: number;
  /** Indica si el peatón o vehículo está en riesgo crítico de colisión por proximidad */
  hasProximityRisk?: boolean;
}

export interface CategoryStats {
  persons: number;
  vehicles: number;
  bicycles: number;
  animalsOther: number;
  deltaPersons: number;
  deltaVehicles: number;
  deltaBicycles: number;
  deltaAnimalsOther: number;
}

export interface PipelineStageTimings {
  captureMs: number;
  preprocessMs: number;
  yoloMs: number;
  poseMs: number;
  nmsTrackMs: number;
  renderMs: number;
  totalMs: number;
}

export interface PipelineMetrics {
  fps: number;
  theoreticalFps: number;
  realFps: number;
  thermalMode: 'Ultra' | 'Balanced' | 'Thermal Safe';
  rollingAvgMs: number;
  rawProposalsCount: number;
  postNmsCount: number;
  suppressedCount: number;
  density: DensityLevel;
  sceneLabel: string;
  timings: PipelineStageTimings;
  letterbox: {
    scale: number;
    padX: number;
    padY: number;
  };
}

export interface TimeSeriesPoint {
  timestampLabel: string;
  secondIndex: number;
  persons: number;
  vehicles: number;
  bicycles: number;
}

export type ThermalControlMode = 'auto' | 'Ultra' | 'Balanced' | 'Thermal Safe';

export interface PipelineSettings {
  confidenceThreshold: number;
  nmsIouThreshold: number;
  cameraMode: CameraSourceMode;
  delegate: HardwareDelegate;
  thermalControlMode: ThermalControlMode;
  arTheme: ARVisualTheme;
  enablePose: boolean;
  enableGyro: boolean;
  crowdBoost: boolean;
  isRunning: boolean;
  /** Modo Exclusivo de Vehículos (filtra 8 clases COCO vehiculares y pausa Pose) */
  onlyVehiclesMode?: boolean;
  /** Límite de velocidad en km/h (20..140, por defecto 60 km/h) */
  speedLimitKmh?: number;
  /** Activa el pitido táctico doble Web Audio API al detectar exceso de velocidad */
  enableSpeedSoundAlert?: boolean;
  /** Factor de calibración de escala monocular (0.5x .. 2.0x, por defecto 1.0x) */
  distanceCalibrationFactor?: number;
  /** Muestra u oculta la Línea Virtual de Conteo y Aforo (Tripwire AR) */
  showCountingLine?: boolean;
  /** Captura automática de foto-multa AR en memoria + IndexedDB al detectar infracción */
  enableAutoEvidence?: boolean;
  /** Límite circular de capturas almacenadas en IndexedDB (10..100, por defecto 50) */
  maxEvidenceRecords?: number;
  /** Muestra y activa el Polígono ROI de 4 vértices (Zona de Exclusión / Estacionamiento Prohibido) */
  enableRoiPolygon?: boolean;
  /** Activa el ajuste magnético (Snapping) de los vértices ROI a carriles en perspectiva, ejes ortogonales y línea de aforo */
  enableRoiSnapping?: boolean;
  /** Segundos detenido dentro del Polígono ROI para disparar alerta de invasión (3..30s, por defecto 10s) */
  roiDwellThresholdSeconds?: number;
  /** Lectura opcional de matrículas ALPR con tesseract.js en Web Worker (apagado por defecto) */
  enableAlpr?: boolean;
  /** Análisis biomecánico de caídas por keypoints y riesgo de colisión peatón-vehículo */
  enableHumanSafety?: boolean;
  /** Asistente de voz sintetizada en tiempo real (Web Speech API es-ES / es-MX) */
  enableVoiceAssistant?: boolean;
  /** Modo Dashcam con compensación de velocidad propia por GPS (navigator.geolocation.watchPosition) */
  enableDashcamGps?: boolean;
}

export const COCO_KEYPOINT_NAMES = [
  'nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
] as const;

export const SKELETON_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], [0, 2], [1, 3], [2, 4],       // Head
  [5, 6],                               // Shoulders
  [5, 7], [7, 9],                       // Left Arm
  [6, 8], [8, 10],                      // Right Arm
  [5, 11], [6, 12], [11, 12],           // Torso
  [11, 13], [13, 15],                   // Left Leg
  [12, 14], [14, 16],                   // Right Leg
];

export const CLASS_COLOR_MAP: Record<TargetClass, { stroke: string; fill: string; label: string }> = {
  person: { stroke: '#38BDF8', fill: 'rgba(56, 189, 248, 0.14)', label: 'person' },
  car: { stroke: '#F97316', fill: 'rgba(249, 115, 22, 0.14)', label: 'car' },
  bus: { stroke: '#F97316', fill: 'rgba(249, 115, 22, 0.14)', label: 'bus' },
  truck: { stroke: '#F97316', fill: 'rgba(249, 115, 22, 0.14)', label: 'truck' },
  bicycle: { stroke: '#10B981', fill: 'rgba(16, 185, 129, 0.14)', label: 'bicycle' },
  motorcycle: { stroke: '#10B981', fill: 'rgba(16, 185, 129, 0.14)', label: 'motorcycle' },
  dog: { stroke: '#A855F7', fill: 'rgba(168, 85, 247, 0.16)', label: 'dog' },
  backpack: { stroke: '#EAB308', fill: 'rgba(234, 179, 8, 0.14)', label: 'backpack' },
};
