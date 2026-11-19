import { VehicleMotionStatus } from '../engine/cvTypes';

/**
 * Las 8 clases oficiales de vehículos del dataset COCO (80 clases):
 * 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck, 8: boat
 */
export const COCO_VEHICLE_CLASS_IDS = new Set<number>([1, 2, 3, 4, 5, 6, 7, 8]);

export const COCO_VEHICLE_CLASS_NAMES = new Set<string>([
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
]);

/**
 * Dimensiones físicas estándar (anchura/escala característica en metros) por clase
 * para proyección monocular inversa (relación dinámica píxeles-por-metro en pantalla).
 */
export const VEHICLE_REFERENCE_SPAN_METERS: Record<string, number> = {
  car: 1.85,
  bus: 2.55,
  truck: 2.50,
  train: 2.85,
  airplane: 3.60,
  boat: 2.20,
  motorcycle: 0.80,
  bicycle: 0.80,
  person: 0.65,
};

export interface KinematicPositionSample {
  centerX: number;
  centerY: number;
  /** Coordenada X compensada restando el desplazamiento angular de la mano (useInertialARTracker) */
  compensatedX: number;
  /** Coordenada Y compensada restando el desplazamiento angular de la mano (useInertialARTracker) */
  compensatedY: number;
  width: number;
  height: number;
  timestampMs: number;
}

export interface VehicleKinematicReading {
  trackId: number;
  speedKmh: number;
  speedMetersPerSec: number;
  speedPxPerSec: number;
  status: VehicleMotionStatus;
  pixelsPerMeter: number;
  stationaryDurationMs: number;
  isVehicle: boolean;
}

interface InternalTrackKinematics {
  samples: KinematicPositionSample[];
  emaSpeedKmh: number;
  emaPixelsPerMeter: number;
  status: VehicleMotionStatus;
  belowStopThresholdSinceMs: number | null;
  lastUpdateMs: number;
}

const MAX_HISTORY_SAMPLES = 15;
const EMA_WINDOW_MS = 500;
const STOP_HYSTERESIS_KMH = 3.5;
const MOVE_HYSTERESIS_KMH = 4.5;
const STOP_HOLD_TIME_MS = 600;

export function isCocoVehicleClass(classId: number, className: string): boolean {
  return (
    COCO_VEHICLE_CLASS_IDS.has(classId) ||
    COCO_VEHICLE_CLASS_NAMES.has(className.toLowerCase())
  );
}

/**
 * Motor Cinemático de Velocidad y Estado (VehicleSpeedEstimator.ts):
 * - Almacena las últimas 15 posiciones por trackId de ByteTrack.
 * - Resta el movimiento inercial de la cámara (deltaGyroX, deltaGyroY) de useInertialARTracker.
 * - Calcula la escala dinámica píxeles-por-metro por perspectiva monocular según clase y tamaño de caja.
 * - Suaviza la lectura con un Filtro EMA sobre una ventana temporal de 500 ms.
 * - Clasifica el estado en DETENIDO (< 3.5 km/h durante > 600 ms) vs. EN MOVIMIENTO (> 4.5 km/h) con histéresis.
 */
export class VehicleSpeedEstimator {
  private readonly tracks = new Map<number, InternalTrackKinematics>();

  /**
   * Actualiza la cinemática de un objeto rastreado por ByteTrack y devuelve su velocidad en km/h y estado.
   */
  public updateTrack(params: {
    trackId: number;
    classId: number;
    className: string;
    centerX: number;
    centerY: number;
    width: number;
    height: number;
    timestampMs: number;
    gyroOffsetX: number;
    gyroOffsetY: number;
    distanceCalibrationFactor?: number;
    /** Velocidad propia del teléfono leída por GPS en Modo Dashcam (km/h) */
    egoGpsSpeedKmh?: number;
  }): VehicleKinematicReading {
    const {
      trackId,
      classId,
      className,
      centerX,
      centerY,
      width,
      height,
      timestampMs,
      gyroOffsetX,
      gyroOffsetY,
      distanceCalibrationFactor = 1.0,
      egoGpsSpeedKmh = 0,
    } = params;

    const normalizedName = className.toLowerCase();
    const isVehicle = isCocoVehicleClass(classId, normalizedName);

    // Coordenadas compensadas inercialmente (restando el desplazamiento angular del giroscopio)
    const compensatedX = centerX - gyroOffsetX;
    const compensatedY = centerY - gyroOffsetY;

    // 1. Escala por Perspectiva Monocular (píxeles por metro según dimensión física estándar de la clase)
    const refMeters = VEHICLE_REFERENCE_SPAN_METERS[normalizedName] ?? 1.5;
    // Usamos la dimensión característica aparente combinada (ancho y alto con corrección de escorzo)
    const effectiveSpanPx = Math.max(18, width * 0.68 + height * 0.32);
    const calibration = Math.max(0.5, Math.min(2.0, distanceCalibrationFactor));
    const instantPixelsPerMeter = Math.max(
      6.0,
      effectiveSpanPx / Math.max(0.35, refMeters * calibration)
    );

    let record = this.tracks.get(trackId);
    if (!record) {
      record = {
        samples: [],
        emaSpeedKmh: 0,
        emaPixelsPerMeter: instantPixelsPerMeter,
        status: 'DETENIDO',
        belowStopThresholdSinceMs: timestampMs,
        lastUpdateMs: timestampMs,
      };
      this.tracks.set(trackId, record);
    }

    // Agregar muestra actual al buffer circular de 15 posiciones
    record.samples.push({
      centerX,
      centerY,
      compensatedX,
      compensatedY,
      width,
      height,
      timestampMs,
    });
    if (record.samples.length > MAX_HISTORY_SAMPLES) {
      record.samples.shift();
    }

    // Suavizar escala píxeles-por-metro para evitar saltos por fluctuación del bounding box
    record.emaPixelsPerMeter =
      record.emaPixelsPerMeter * 0.78 + instantPixelsPerMeter * 0.22;

    // 2. Buscar muestra de referencia dentro de la ventana temporal (~250ms - 500ms) para derivada robusta
    let anchorSample = record.samples[0];
    for (let i = record.samples.length - 2; i >= 0; i--) {
      const candidate = record.samples[i];
      const ageMs = timestampMs - candidate.timestampMs;
      anchorSample = candidate;
      if (ageMs >= 260) {
        break;
      }
    }

    const dtWindowSec = Math.max(0.016, (timestampMs - anchorSample.timestampMs) * 0.001);
    const frameDtMs = Math.max(8, timestampMs - record.lastUpdateMs);
    record.lastUpdateMs = timestampMs;

    let speedPxPerSec = 0;
    let instantKmh = 0;

    if (record.samples.length >= 2 && timestampMs > anchorSample.timestampMs) {
      const dxPx = compensatedX - anchorSample.compensatedX;
      const dyPx = compensatedY - anchorSample.compensatedY;
      // El desplazamiento en perspectiva vertical (profundidad/carril) tiene factor geométrico 1.35x
      const displacementPx = Math.hypot(dxPx, dyPx * 1.35);

      // Filtrar micro-vibraciones menores a 1.6 píxeles del detector neuronal
      const cleanDispPx = theDisplacementFilter(displacementPx, width);
      speedPxPerSec = cleanDispPx / dtWindowSec;

      const speedMetersPerSec = speedPxPerSec / Math.max(4.0, record.emaPixelsPerMeter);
      const rawRelativeKmh = Math.min(220, Math.max(0, speedMetersPerSec * 3.6));

      // Compensación Modo Dashcam GPS: restar la velocidad propia del vehículo medida por GPS
      const safeEgoKmh = Math.max(0, egoGpsSpeedKmh);
      instantKmh =
        safeEgoKmh > 0.5
          ? Math.min(220, Math.max(0, Math.abs(rawRelativeKmh - safeEgoKmh)))
          : rawRelativeKmh;
    }

    // 3. Filtro EMA sobre ventana de 500 ms: alpha = 1 - exp(-dt / 500ms)
    const alphaEma = Math.min(0.45, Math.max(0.1, 1 - Math.exp(-frameDtMs / EMA_WINDOW_MS)));
    record.emaSpeedKmh = record.emaSpeedKmh * (1 - alphaEma) + instantKmh * alphaEma;

    if (record.emaSpeedKmh < 0.8) {
      record.emaSpeedKmh = 0;
    }

    // 4. Clasificación de Estado con Histéresis (DETENIDO < 3.5 km/h por > 600 ms vs. EN MOVIMIENTO > 4.5 km/h)
    let stationaryDurationMs = 0;
    if (record.emaSpeedKmh < STOP_HYSTERESIS_KMH) {
      if (record.belowStopThresholdSinceMs === null) {
        record.belowStopThresholdSinceMs = timestampMs;
      }
      stationaryDurationMs = timestampMs - record.belowStopThresholdSinceMs;
      if (stationaryDurationMs >= STOP_HOLD_TIME_MS) {
        record.status = 'DETENIDO';
      }
    } else if (record.emaSpeedKmh > MOVE_HYSTERESIS_KMH) {
      record.belowStopThresholdSinceMs = null;
      stationaryDurationMs = 0;
      record.status = 'EN MOVIMIENTO';
    } else if (record.belowStopThresholdSinceMs !== null) {
      stationaryDurationMs = timestampMs - record.belowStopThresholdSinceMs;
    }

    const displayKmh =
      record.status === 'DETENIDO' && record.emaSpeedKmh < STOP_HYSTERESIS_KMH
        ? Number(Math.min(record.emaSpeedKmh, 2.4).toFixed(1))
        : Number(record.emaSpeedKmh.toFixed(1));

    return {
      trackId,
      speedKmh: displayKmh,
      speedMetersPerSec: Number((displayKmh / 3.6).toFixed(2)),
      speedPxPerSec: Number(speedPxPerSec.toFixed(1)),
      status: record.status,
      pixelsPerMeter: Number(record.emaPixelsPerMeter.toFixed(1)),
      stationaryDurationMs,
      isVehicle,
    };
  }

  /**
   * Elimina tracks inactivos de la memoria para evitar fugas tras sesiones largas.
   */
  public pruneStaleTracks(nowMs: number, maxIdleMs = 2500): void {
    if (this.tracks.size <= 32) return;
    for (const [trackId, rec] of this.tracks.entries()) {
      if (nowMs - rec.lastUpdateMs > maxIdleMs) {
        this.tracks.delete(trackId);
      }
    }
  }

  public reset(): void {
    this.tracks.clear();
  }
}

function theDisplacementFilter(rawDisplacementPx: number, boxWidthPx: number): number {
  const deadzonePx = Math.max(1.5, boxWidthPx * 0.015);
  return rawDisplacementPx <= deadzonePx ? 0 : rawDisplacementPx - deadzonePx * 0.5;
}
