import { useCallback, useEffect, useRef } from 'react';
import { OnnxKeypoint2D, OnnxTrackedDetection } from './useOnnxVisionEngine';

/**
 * Filtro de Kalman 1D con velocidad para suprimir el ruido blanco (jitter)
 * de los sensores MEMS del navegador (devicemotion / deviceorientation) y de las coordenadas de caja.
 */
export class ScalarKalmanFilter {
  private x: number;
  private v: number;
  private p00: number;
  private p01: number;
  private p10: number;
  private p11: number;
  private readonly qPos: number;
  private readonly qVel: number;
  private readonly rMeas: number;

  constructor(initialValue = 0, qPos = 0.018, qVel = 0.08, rMeas = 0.14) {
    this.x = initialValue;
    this.v = 0;
    this.p00 = 1.0;
    this.p01 = 0.0;
    this.p10 = 0.0;
    this.p11 = 1.0;
    this.qPos = qPos;
    this.qVel = qVel;
    this.rMeas = rMeas;
  }

  public predictAndUpdate(measurement: number, dtSec: number): number {
    const dt = Math.max(0.004, Math.min(0.08, dtSec));

    // 1. Predicción: x_k|k-1 = x_k-1 + v_k-1 * dt
    const predX = this.x + this.v * dt;
    const predV = this.v;

    const p00 = this.p00 + dt * (this.p10 + this.p01) + dt * dt * this.p11 + this.qPos;
    const p01 = this.p01 + dt * this.p11;
    const p10 = this.p10 + dt * this.p11;
    const p11 = this.p11 + this.qVel;

    // 2. Innovación y Ganancia de Kalman K
    const y = measurement - predX;
    const s = p00 + this.rMeas;
    const invS = 1.0 / Math.max(1e-6, s);
    const k0 = p00 * invS;
    const k1 = p10 * invS;

    // 3. Actualización del estado y matriz de covarianza P
    this.x = predX + k0 * y;
    this.v = predV + k1 * y;

    this.p00 = (1 - k0) * p00;
    this.p01 = (1 - k0) * p01;
    this.p10 = p10 - k1 * p00;
    this.p11 = p11 - k1 * p01;

    return this.x;
  }

  public reset(value = 0): void {
    this.x = value;
    this.v = 0;
    this.p00 = 1.0;
    this.p01 = 0;
    this.p10 = 0;
    this.p11 = 1.0;
  }

  public getValue(): number {
    return this.x;
  }

  public getVelocity(): number {
    return this.v;
  }
}

export interface InertialARState {
  omegaX: number; // rad/s Pitch filtrado por Kalman
  omegaY: number; // rad/s Yaw filtrado por Kalman
  omegaZ: number; // rad/s Roll
  offsetX: number; // Compensación espacial inter-frame X (px)
  offsetY: number; // Compensación espacial inter-frame Y (px)
  kalmanGain: number;
  hasHardwareSensor: boolean;
  lastResetTimestampMs: number;
}

export interface SmoothedSpatialBox {
  x: number;
  y: number;
  width: number;
  height: number;
  keypoints: OnnxKeypoint2D[];
  lastSeenMs: number;
  kx: ScalarKalmanFilter;
  ky: ScalarKalmanFilter;
  kw: ScalarKalmanFilter;
  kh: ScalarKalmanFilter;
}

export interface UseInertialARTrackerOptions {
  enabled: boolean;
  screenWidth: number;
  screenHeight: number;
  horizontalFovDeg?: number;
}

export interface UseInertialARTrackerResult {
  inertialStateRef: React.MutableRefObject<InertialARState>;
  onNewOnnxInferenceBatch: () => void;
  injectPointerDelta: (dxPx: number, dyPx: number) => void;
  computeAnchoredDetectionState: (
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
}

/**
 * Hook de Anclaje Espacial Inercial (useInertialARTracker.ts):
 * Combina devicemotion (velocidad angular MEMS en deg/s -> rad/s) y deviceorientation (ángulos Euler)
 * con un Filtro de Kalman + LERP para compensar el movimiento de la mano en tiempo real entre cada
 * inferencia de ONNX, logrando que las cajas y esqueletos floten anclados al mundo físico a 60/120 FPS sin jitter.
 */
export function useInertialARTracker({
  enabled,
  screenWidth,
  screenHeight,
  horizontalFovDeg = 64,
}: UseInertialARTrackerOptions): UseInertialARTrackerResult {
  const inertialStateRef = useRef<InertialARState>({
    omegaX: 0,
    omegaY: 0,
    omegaZ: 0,
    offsetX: 0,
    offsetY: 0,
    kalmanGain: 0.34,
    hasHardwareSensor: false,
    lastResetTimestampMs: performance.now(),
  });

  // Filtros de Kalman 1D para los ejes angulares del giroscopio (Pitch y Yaw)
  const gyroKalmanXRef = useRef(new ScalarKalmanFilter(0, 0.012, 0.05, 0.16));
  const gyroKalmanYRef = useRef(new ScalarKalmanFilter(0, 0.012, 0.05, 0.16));

  // Mapa persistente de Filtros de Kalman + LERP por trackId de ByteTrack
  const spatialTrackMapRef = useRef<Map<number, SmoothedSpatialBox>>(new Map());

  const prevEulerRef = useRef<{ alpha: number | null; beta: number | null; gamma: number | null }>({
    alpha: null,
    beta: null,
    gamma: null,
  });

  const configRef = useRef({
    enabled,
    screenWidth,
    screenHeight,
    horizontalFovDeg,
  });
  configRef.current = {
    enabled,
    screenWidth,
    screenHeight,
    horizontalFovDeg,
  };

  /**
   * Reinicio sincronizado cuando llega una nueva inferencia de ONNX:
   * Amortigua suavemente el desplazamiento acumulado hacia 0 sin saltos bruscos.
   */
  const onNewOnnxInferenceBatch = useCallback(() => {
    const state = inertialStateRef.current;
    state.offsetX *= 0.2;
    state.offsetY *= 0.2;
    state.lastResetTimestampMs = performance.now();
  }, []);

  /**
   * Permite inyectar desplazamiento táctil o de puntero para probar la estabilización inercial
   * en navegadores de escritorio que no poseen giroscopio físico.
   */
  const injectPointerDelta = useCallback((dxPx: number, dyPx: number) => {
    if (!configRef.current.enabled) return;
    const state = inertialStateRef.current;
    state.omegaY = -dxPx * 0.04;
    state.omegaX = dyPx * 0.04;
    state.offsetX = Math.max(-72, Math.min(72, state.offsetX + dxPx * 0.68));
    state.offsetY = Math.max(-72, Math.min(72, state.offsetY + dyPx * 0.68));
  }, []);

  // Suscripción a devicemotion y deviceorientation a 60/120Hz con filtrado de Kalman
  useEffect(() => {
    let lastSensorTs = performance.now();

    const handleDeviceMotion = (ev: DeviceMotionEvent) => {
      if (!configRef.current.enabled) return;
      const rot = ev.rotationRate;
      if (!rot) return;

      const now = performance.now();
      const dtSec = Math.min(0.05, Math.max(0.004, (now - lastSensorTs) * 0.001));
      lastSensorTs = now;

      const degToRad = Math.PI / 180;
      const rawPitchRadS = (rot.alpha ?? 0) * degToRad;
      const rawYawRadS = (rot.beta ?? 0) * degToRad;
      const rawRollRadS = (rot.gamma ?? 0) * degToRad;

      if (Math.abs(rawPitchRadS) > 0.001 || Math.abs(rawYawRadS) > 0.001) {
        inertialStateRef.current.hasHardwareSensor = true;
      }

      // Filtrado de Kalman para eliminar ruido de alta frecuencia (jitter de la mano)
      const filteredPitch = gyroKalmanXRef.current.predictAndUpdate(rawPitchRadS, dtSec);
      const filteredYaw = gyroKalmanYRef.current.predictAndUpdate(rawYawRadS, dtSec);

      // Zona muerta de 0.012 rad/s para evitar deriva estática cuando el teléfono está quieto
      const wx = Math.abs(filteredPitch) > 0.012 ? filteredPitch : 0;
      const wy = Math.abs(filteredYaw) > 0.012 ? filteredYaw : 0;

      const fovRad = (configRef.current.horizontalFovDeg * Math.PI) / 180;
      const focalPx =
        Math.max(320, configRef.current.screenWidth) / (2 * Math.tan(fovRad * 0.5));

      const state = inertialStateRef.current;
      state.omegaX = wx;
      state.omegaY = wy;
      state.omegaZ = rawRollRadS;

      const maxClamp = 74;
      state.offsetX = Math.max(
        -maxClamp,
        Math.min(maxClamp, (state.offsetX - wy * dtSec * focalPx) * 0.94)
      );
      state.offsetY = Math.max(
        -maxClamp,
        Math.min(maxClamp, (state.offsetY + wx * dtSec * focalPx) * 0.94)
      );
    };

    const handleDeviceOrientation = (ev: DeviceOrientationEvent) => {
      if (!configRef.current.enabled || inertialStateRef.current.hasHardwareSensor) return;
      if (ev.gamma === null || ev.beta === null) return;

      const prev = prevEulerRef.current;
      if (prev.gamma !== null && prev.beta !== null) {
        const dGamma = ev.gamma - prev.gamma;
        const dBeta = ev.beta - prev.beta;

        if (Math.abs(dGamma) < 25 && Math.abs(dBeta) < 25) {
          const state = inertialStateRef.current;
          const smoothDGamma = gyroKalmanYRef.current.predictAndUpdate(dGamma, 0.016);
          const smoothDBeta = gyroKalmanXRef.current.predictAndUpdate(dBeta, 0.016);
          state.offsetX = Math.max(-68, Math.min(68, (state.offsetX - smoothDGamma * 2.4) * 0.92));
          state.offsetY = Math.max(-68, Math.min(68, (state.offsetY + smoothDBeta * 2.4) * 0.92));
        }
      }
      prevEulerRef.current = { alpha: ev.alpha, beta: ev.beta, gamma: ev.gamma };
    };

    window.addEventListener('devicemotion', handleDeviceMotion, { passive: true });
    window.addEventListener('deviceorientation', handleDeviceOrientation, { passive: true });

    return () => {
      window.removeEventListener('devicemotion', handleDeviceMotion);
      window.removeEventListener('deviceorientation', handleDeviceOrientation);
    };
  }, []);

  /**
   * Decae suavemente el offset inercial en cada tick de renderizado del Canvas (60/120 FPS).
   */
  const decayInertialOffsetOnRenderFrame = useCallback(() => {
    const state = inertialStateRef.current;
    state.offsetX *= 0.95;
    state.offsetY *= 0.95;
    if (!configRef.current.enabled) {
      return { gx: 0, gy: 0 };
    }
    return { gx: state.offsetX, gy: state.offsetY };
  }, []);

  /**
   * Combina Filtro de Kalman por coordenada + Interpolación Lineal (LERP) + Offset Inercial
   * para cada objeto rastreado por ByteTrack.
   */
  const computeAnchoredDetectionState = useCallback(
    (det: OnnxTrackedDetection, nowMs: number, dtSec: number) => {
      const cache = spatialTrackMapRef.current;
      const existing = cache.get(det.trackId);
      const lerpAlpha = 0.36;

      let sx = det.screenBox.x;
      let sy = det.screenBox.y;
      let sW = det.screenBox.width;
      let sH = det.screenBox.height;
      let smoothKps = det.keypoints;

      if (existing && nowMs - existing.lastSeenMs < 550) {
        // 1. Predicción y actualización con Filtro de Kalman 1D por eje
        const kalmanX = existing.kx.predictAndUpdate(det.screenBox.x, dtSec);
        const kalmanY = existing.ky.predictAndUpdate(det.screenBox.y, dtSec);
        const kalmanW = existing.kw.predictAndUpdate(det.screenBox.width, dtSec);
        const kalmanH = existing.kh.predictAndUpdate(det.screenBox.height, dtSec);

        // 2. Fusión Kalman + LERP para transición visual a 60/120 FPS libre de jitter
        sx = existing.x + (kalmanX - existing.x) * lerpAlpha;
        sy = existing.y + (kalmanY - existing.y) * lerpAlpha;
        sW = existing.width + (kalmanW - existing.width) * lerpAlpha;
        sH = existing.height + (kalmanH - existing.height) * lerpAlpha;

        if (det.keypoints.length === 17 && existing.keypoints.length === 17) {
          smoothKps = det.keypoints.map((kp, idx) => ({
            ...kp,
            x: existing.keypoints[idx].x + (kp.x - existing.keypoints[idx].x) * lerpAlpha,
            y: existing.keypoints[idx].y + (kp.y - existing.keypoints[idx].y) * lerpAlpha,
          }));
        }

        existing.x = sx;
        existing.y = sy;
        existing.width = sW;
        existing.height = sH;
        existing.keypoints = smoothKps;
        existing.lastSeenMs = nowMs;
      } else {
        cache.set(det.trackId, {
          x: sx,
          y: sy,
          width: sW,
          height: sH,
          keypoints: smoothKps,
          lastSeenMs: nowMs,
          kx: new ScalarKalmanFilter(sx),
          ky: new ScalarKalmanFilter(sy),
          kw: new ScalarKalmanFilter(sW),
          kh: new ScalarKalmanFilter(sH),
        });
      }

      // Limpieza periódica de tracks antiguos
      if (cache.size > 48) {
        for (const [tid, rec] of cache.entries()) {
          if (nowMs - rec.lastSeenMs > 1600) {
            cache.delete(tid);
          }
        }
      }

      const gx = configRef.current.enabled ? inertialStateRef.current.offsetX : 0;
      const gy = configRef.current.enabled ? inertialStateRef.current.offsetY : 0;

      return {
        x: sx + gx,
        y: sy + gy,
        width: sW,
        height: sH,
        keypoints: smoothKps,
        gyroOffsetX: gx,
        gyroOffsetY: gy,
      };
    },
    []
  );

  return {
    inertialStateRef,
    onNewOnnxInferenceBatch,
    injectPointerDelta,
    computeAnchoredDetectionState,
    decayInertialOffsetOnRenderFrame,
  };
}
