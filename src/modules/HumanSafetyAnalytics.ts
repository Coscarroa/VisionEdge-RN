/**
 * Módulo 3: Análisis Avanzado de Personas y Seguridad Laboral (HumanSafetyAnalytics.ts)
 * (src/modules/HumanSafetyAnalytics.ts)
 *
 * Activo cuando el Modo Universal (con yolo11n-pose.onnx de 17 keypoints) está encendido:
 * 1. Detección de Caídas por Keypoints:
 *    - Analiza los 17 puntos clave de cada persona (hombros 5/6, caderas 11/12, tobillos 15/16).
 *    - Calcula la relación de aspecto del esqueleto (ancho / alto) y la altura relativa de los
 *      hombros respecto a las caderas/tobillos.
 *    - Si una persona pasa a posición horizontal cerca del suelo y permanece inmóvil por más
 *      de 2.5 segundos, activa marco rojo pulsante con alerta "⚠️ POSIBLE CAÍDA DETECTADA".
 * 2. Proximidad Peatón-Vehículo / Aglomeración:
 *    - Calcula la distancia euclidiana normalizada por perspectiva entre centroides.
 *    - Si una persona y un vehículo en movimiento entran en zona de proximidad crítica,
 *      genera una línea de advertencia ámbar/roja conectando ambos objetos ("RIESGO DE COLISIÓN").
 */

import { OnnxKeypoint2D } from '../hooks/useOnnxVisionEngine';
import { VehicleMotionStatus } from '../engine/cvTypes';

export interface PersonFallEvaluation {
  isHorizontalPosture: boolean;
  isFallDetected: boolean;
  fallDurationSeconds: number;
  justTriggeredFall: boolean;
}

export interface ProximityHazardLink {
  id: string;
  personTrackId: number;
  vehicleTrackId: number;
  personX: number;
  personY: number;
  vehicleX: number;
  vehicleY: number;
  estimatedDistanceMeters: number;
  severity: 'CRITICAL' | 'WARNING';
  label: string;
}

const FALL_HOLD_THRESHOLD_MS = 2500; // 2.5 segundos inmóvil en postura horizontal

export class HumanSafetyAnalyticsManager {
  private readonly horizontalSinceByTrack = new Map<number, number>();
  private readonly lastCentroidByTrack = new Map<number, { x: number; y: number; ts: number }>();
  private readonly fallCreditedTracks = new Set<number>();
  private totalFallAlerts = 0;
  private totalProximityAlerts = 0;
  private readonly proximityCreditedPairs = new Set<string>();

  public getTotalFallAlerts(): number {
    return this.totalFallAlerts;
  }

  public getTotalProximityAlerts(): number {
    return this.totalProximityAlerts;
  }

  /**
   * Analiza la biomecánica de los 17 keypoints de una persona y la relación de aspecto de su caja
   * para detectar caídas sostenidas (> 2.5 segundos en postura horizontal e inmóvil).
   */
  public evaluatePersonFall(params: {
    enabled: boolean;
    trackId: number;
    screenBox: { x: number; y: number; width: number; height: number };
    keypoints: OnnxKeypoint2D[];
    speedPxPerSec: number;
    nowMs: number;
  }): PersonFallEvaluation {
    const { enabled, trackId, screenBox, keypoints, speedPxPerSec, nowMs } = params;

    if (!enabled) {
      return {
        isHorizontalPosture: false,
        isFallDetected: false,
        fallDurationSeconds: 0,
        justTriggeredFall: false,
      };
    }

    const boxAspect = screenBox.width / Math.max(1, screenBox.height);

    // Evaluar geometría de los 17 keypoints COCO:
    // 5: left_shoulder, 6: right_shoulder, 11: left_hip, 12: right_hip, 15: left_ankle, 16: right_ankle
    let kpHorizontalScore = 0;
    let validKpCount = 0;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < keypoints.length; i++) {
      const kp = keypoints[i];
      if (kp && kp.score > 0.42) {
        validKpCount++;
        if (kp.x < minX) minX = kp.x;
        if (kp.x > maxX) maxX = kp.x;
        if (kp.y < minY) minY = kp.y;
        if (kp.y > maxY) maxY = kp.y;
      }
    }

    if (validKpCount >= 6) {
      const skelW = Math.max(1, maxX - minX);
      const skelH = Math.max(1, maxY - minY);
      const skelAspect = skelW / skelH;
      if (skelAspect >= 1.15) {
        kpHorizontalScore += 1;
      }

      const lShoulder = keypoints[5];
      const rShoulder = keypoints[6];
      const lHip = keypoints[11];
      const rHip = keypoints[12];
      const lAnkle = keypoints[15];
      const rAnkle = keypoints[16];

      const shoulderY =
        lShoulder?.score > 0.4 && rShoulder?.score > 0.4
          ? (lShoulder.y + rShoulder.y) * 0.5
          : lShoulder?.score > 0.4
          ? lShoulder.y
          : rShoulder?.score > 0.4
          ? rShoulder.y
          : null;

      const hipY =
        lHip?.score > 0.4 && rHip?.score > 0.4
          ? (lHip.y + rHip.y) * 0.5
          : lHip?.score > 0.4
          ? lHip.y
          : rHip?.score > 0.4
          ? rHip.y
          : null;

      const ankleY =
        lAnkle?.score > 0.35 && rAnkle?.score > 0.35
          ? (lAnkle.y + rAnkle.y) * 0.5
          : lAnkle?.score > 0.35
          ? lAnkle.y
          : rAnkle?.score > 0.35
          ? rAnkle.y
          : null;

      // Si la diferencia vertical entre hombros y caderas/tobillos colapsa respecto al ancho del torso
      if (shoulderY !== null && hipY !== null) {
        const torsoVerticalSpan = Math.abs(hipY - shoulderY);
        if (torsoVerticalSpan < skelW * 0.34) {
          kpHorizontalScore += 1;
        }
      }
      if (shoulderY !== null && ankleY !== null) {
        const fullBodyVerticalSpan = Math.abs(ankleY - shoulderY);
        if (fullBodyVerticalSpan < skelW * 0.55) {
          kpHorizontalScore += 1;
        }
      }
    }

    const isHorizontalPosture = boxAspect >= 1.18 || kpHorizontalScore >= 2;
    const isImmobile = speedPxPerSec < 48;

    if (isHorizontalPosture && isImmobile) {
      let sinceMs = this.horizontalSinceByTrack.get(trackId);
      if (sinceMs === undefined) {
        sinceMs = nowMs;
        this.horizontalSinceByTrack.set(trackId, sinceMs);
      }

      const elapsedMs = nowMs - sinceMs;
      const fallDurationSeconds = Number((elapsedMs / 1000).toFixed(1));
      const isFallDetected = elapsedMs >= FALL_HOLD_THRESHOLD_MS;
      let justTriggeredFall = false;

      if (isFallDetected && !this.fallCreditedTracks.has(trackId)) {
        this.fallCreditedTracks.add(trackId);
        this.totalFallAlerts += 1;
        justTriggeredFall = true;
      }

      return {
        isHorizontalPosture: true,
        isFallDetected,
        fallDurationSeconds,
        justTriggeredFall,
      };
    } else {
      this.horizontalSinceByTrack.delete(trackId);
      return {
        isHorizontalPosture: false,
        isFallDetected: false,
        fallDurationSeconds: 0,
        justTriggeredFall: false,
      };
    }
  }

  /**
   * Evalúa la proximidad espacial normalizada entre todos los peatones activos y los vehículos
   * en movimiento para detectar riesgo de colisión (RIESGO DE COLISIÓN).
   */
  public computePedestrianVehicleProximityLinks(params: {
    enabled: boolean;
    persons: Array<{
      trackId: number;
      centerX: number;
      centerY: number;
      width: number;
      height: number;
    }>;
    vehicles: Array<{
      trackId: number;
      className: string;
      centerX: number;
      centerY: number;
      width: number;
      height: number;
      speedKmh: number;
      motionStatus: VehicleMotionStatus;
    }>;
    screenWidth: number;
    screenHeight: number;
  }): {
    links: ProximityHazardLink[];
    riskTrackIds: Set<number>;
    newlyTriggeredLink: ProximityHazardLink | null;
  } {
    const { enabled, persons, vehicles, screenWidth, screenHeight } = params;
    const links: ProximityHazardLink[] = [];
    const riskTrackIds = new Set<number>();
    let newlyTriggeredLink: ProximityHazardLink | null = null;

    if (!enabled || persons.length === 0 || vehicles.length === 0) {
      return { links, riskTrackIds, newlyTriggeredLink };
    }

    const diagScreen = Math.max(400, Math.hypot(screenWidth, screenHeight));

    for (const person of persons) {
      // Escala de referencia en metros usando la estatura aparente del peatón (~1.70m)
      const personPixelsPerMeter = Math.max(18, person.height / 1.7);

      for (const veh of vehicles) {
        // Priorizar vehículos en movimiento o muy cercanos al peatón
        if (veh.motionStatus !== 'EN MOVIMIENTO' && veh.speedKmh < 3.0) {
          continue;
        }

        const distPx = Math.hypot(person.centerX - veh.centerX, person.centerY - veh.centerY);
        const avgScalePxPerM = Math.max(
          16,
          (personPixelsPerMeter + Math.max(16, veh.width / 1.9)) * 0.5
        );
        const estimatedDistanceMeters = Number((distPx / avgScalePxPerM).toFixed(1));
        const normDist = distPx / diagScreen;

        // Umbral crítico dinámico: mayor radio de seguridad cuando el vehículo va más rápido
        const dynamicSafeMeters = 2.8 + Math.min(3.2, veh.speedKmh * 0.045);

        if (estimatedDistanceMeters <= dynamicSafeMeters && normDist <= 0.34) {
          const severity: 'CRITICAL' | 'WARNING' =
            estimatedDistanceMeters <= 2.1 || veh.speedKmh >= 35 ? 'CRITICAL' : 'WARNING';

          const pairId = `p${person.trackId}_v${veh.trackId}`;
          const link: ProximityHazardLink = {
            id: pairId,
            personTrackId: person.trackId,
            vehicleTrackId: veh.trackId,
            personX: person.centerX,
            personY: person.centerY,
            vehicleX: veh.centerX,
            vehicleY: veh.centerY,
            estimatedDistanceMeters,
            severity,
            label: `⚠️ RIESGO DE COLISIÓN • ${estimatedDistanceMeters}m`,
          };

          links.push(link);
          riskTrackIds.add(person.trackId);
          riskTrackIds.add(veh.trackId);

          if (!this.proximityCreditedPairs.has(pairId)) {
            this.proximityCreditedPairs.add(pairId);
            this.totalProximityAlerts += 1;
            if (!newlyTriggeredLink) {
              newlyTriggeredLink = link;
            }
          }
        }
      }
    }

    return { links, riskTrackIds, newlyTriggeredLink };
  }

  public resetStats(): void {
    this.horizontalSinceByTrack.clear();
    this.lastCentroidByTrack.clear();
    this.fallCreditedTracks.clear();
    this.proximityCreditedPairs.clear();
    this.totalFallAlerts = 0;
    this.totalProximityAlerts = 0;
  }
}

export const globalHumanSafetyAnalytics = new HumanSafetyAnalyticsManager();
