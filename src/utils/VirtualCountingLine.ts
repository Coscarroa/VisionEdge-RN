/**
 * Línea Virtual de Conteo y Aforo Vehicular Bidireccional (VirtualCountingLine.ts)
 * - Geometría Computacional 2D: Intersección exacta de segmentos mediante producto vectorial.
 * - Distingue dirección del cruce (ENTRADA / Sentido A vs. SALIDA / Sentido B).
 * - Garantiza conteo único por vehículo mediante Set<number> (countedTrackIds).
 * - Desglosa el aforo acumulado en tiempo real por tipo: Autos, Motos/Bicis y Buses/Camiones.
 * - Soporta arrastre táctil de los nodos extremos (A y B) y alternancia Horizontal / Vertical.
 */

export type CrossingDirection = 'ENTRADA' | 'SALIDA';

export interface Point2D {
  x: number;
  y: number;
}

export interface VirtualLineEndpointsNorm {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrafficCountBreakdown {
  total: number;
  inbound: number;  // ↑ Entrada (Sentido A)
  outbound: number; // ↓ Salida (Sentido B)
  cars: number;     // car
  twoWheelers: number; // motorcycle, bicycle
  heavyVehicles: number; // bus, truck, train, boat, airplane
  lastCrossingTrackId: number | null;
  lastCrossingClass: string | null;
  lastCrossingDirection: CrossingDirection | null;
  lastFlashTimestampMs: number;
  lastIntersectionPoint: Point2D | null;
}

/**
 * Producto vectorial 2D: (B - A) x (C - A)
 */
export function crossProduct2D(a: Point2D, b: Point2D, c: Point2D): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

/**
 * Verifica si un punto C cae dentro de la caja delimitadora del segmento AB (para colinealidad).
 */
function isPointOnSegmentBoundingBox(a: Point2D, b: Point2D, c: Point2D): boolean {
  return (
    c.x <= Math.max(a.x, b.x) + 1e-4 &&
    c.x >= Math.min(a.x, b.x) - 1e-4 &&
    c.y <= Math.max(a.y, b.y) + 1e-4 &&
    c.y >= Math.min(a.y, b.y) - 1e-4
  );
}

/**
 * Comprueba si el segmento de trayectoria del vehículo (p1 -> p2) intersecta
 * el segmento de la Línea Virtual de Aforo (q1 -> q2) y devuelve el punto de cruce y sentido.
 */
export function computeSegmentIntersection(
  p1: Point2D,
  p2: Point2D,
  q1: Point2D,
  q2: Point2D
): { intersects: boolean; direction: CrossingDirection; point: Point2D } | null {
  const d1 = crossProduct2D(q1, q2, p1);
  const d2 = crossProduct2D(q1, q2, p2);
  const d3 = crossProduct2D(p1, p2, q1);
  const d4 = crossProduct2D(p1, p2, q2);

  const hasProperIntersection =
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));

  const hasCollinearIntersection =
    (Math.abs(d1) < 1e-5 && isPointOnSegmentBoundingBox(q1, q2, p1)) ||
    (Math.abs(d2) < 1e-5 && isPointOnSegmentBoundingBox(q1, q2, p2)) ||
    (Math.abs(d3) < 1e-5 && isPointOnSegmentBoundingBox(p1, p2, q1)) ||
    (Math.abs(d4) < 1e-5 && isPointOnSegmentBoundingBox(p1, p2, q2));

  if (!hasProperIntersection && !hasCollinearIntersection) {
    return null;
  }

  // Calcular punto exacto de intersección paramétrica
  const denom = (p1.x - p2.x) * (q1.y - q2.y) - (p1.y - p2.y) * (q1.x - q2.x);
  let ix = (p1.x + p2.x) * 0.5;
  let iy = (p1.y + p2.y) * 0.5;

  if (Math.abs(denom) > 1e-5) {
    const t = ((p1.x - q1.x) * (q1.y - q2.y) - (p1.y - q1.y) * (q1.x - q2.x)) / denom;
    ix = p1.x + t * (p2.x - p1.x);
    iy = p1.y + t * (p2.y - p1.y);
  }

  // El signo del producto vectorial entre el vector de la línea (q2 - q1) y el desplazamiento (p2 - p1)
  // determina de forma invariante si el vehículo cruzó en Sentido A (ENTRADA) o Sentido B (SALIDA)
  const lineVecX = q2.x - q1.x;
  const lineVecY = q2.y - q1.y;
  const moveVecX = p2.x - p1.x;
  const moveVecY = p2.y - p1.y;
  const crossDir = lineVecX * moveVecY - lineVecY * moveVecX;

  const direction: CrossingDirection = crossDir < 0 ? 'ENTRADA' : 'SALIDA';

  return {
    intersects: true,
    direction,
    point: { x: ix, y: iy },
  };
}

export class VirtualCountingLine {
  private endpointsNorm: VirtualLineEndpointsNorm = {
    x1: 0.14,
    y1: 0.62,
    x2: 0.86,
    y2: 0.62,
  };

  private isHorizontalPreset = true;
  private readonly countedTrackIds = new Set<number>();
  private readonly trackDirectionMap = new Map<number, CrossingDirection>();
  private readonly lastCentroidByTrack = new Map<number, Point2D>();

  private stats: TrafficCountBreakdown = {
    total: 0,
    inbound: 0,
    outbound: 0,
    cars: 0,
    twoWheelers: 0,
    heavyVehicles: 0,
    lastCrossingTrackId: null,
    lastCrossingClass: null,
    lastCrossingDirection: null,
    lastFlashTimestampMs: 0,
    lastIntersectionPoint: null,
  };

  public getEndpointsNorm(): VirtualLineEndpointsNorm {
    return this.endpointsNorm;
  }

  public getEndpointsPixels(screenWidth: number, screenHeight: number): {
    a: Point2D;
    b: Point2D;
  } {
    return {
      a: {
        x: this.endpointsNorm.x1 * screenWidth,
        y: this.endpointsNorm.y1 * screenHeight,
      },
      b: {
        x: this.endpointsNorm.x2 * screenWidth,
        y: this.endpointsNorm.y2 * screenHeight,
      },
    };
  }

  public setEndpointNorm(which: 'a' | 'b', xNorm: number, yNorm: number): void {
    const cx = Math.max(0.04, Math.min(0.96, xNorm));
    const cy = Math.max(0.10, Math.min(0.90, yNorm));
    if (which === 'a') {
      this.endpointsNorm = { ...this.endpointsNorm, x1: cx, y1: cy };
    } else {
      this.endpointsNorm = { ...this.endpointsNorm, x2: cx, y2: cy };
    }
  }

  /**
   * Alterna rápidamente la orientación de la línea virtual entre Horizontal (cruce de carril)
   * y Vertical (flujo lateral de avenida).
   */
  public toggleOrientation(): VirtualLineEndpointsNorm {
    this.isHorizontalPreset = !this.isHorizontalPreset;
    if (this.isHorizontalPreset) {
      this.endpointsNorm = { x1: 0.14, y1: 0.62, x2: 0.86, y2: 0.62 };
    } else {
      this.endpointsNorm = { x1: 0.50, y1: 0.22, x2: 0.50, y2: 0.82 };
    }
    return this.endpointsNorm;
  }

  /**
   * Evalúa si el vector de desplazamiento del centroide del vehículo (prevCenter -> currentCenter)
   * cruza el segmento de la línea virtual (lineA -> lineB).
   */
  public checkVehicleCrossing(params: {
    trackId: number;
    className: string;
    currentCenter: Point2D;
    trail?: Array<{ x: number; y: number }>;
    screenWidth: number;
    screenHeight: number;
    nowMs: number;
    enabled: boolean;
  }): { hasCrossed: boolean; direction: CrossingDirection | null } {
    const {
      trackId,
      className,
      currentCenter,
      trail,
      screenWidth,
      screenHeight,
      nowMs,
      enabled,
    } = params;

    // Si este trackId ya fue contabilizado, devolvemos su estado persistente sin duplicar conteo
    if (this.countedTrackIds.has(trackId)) {
      this.lastCentroidByTrack.set(trackId, currentCenter);
      return {
        hasCrossed: true,
        direction: this.trackDirectionMap.get(trackId) ?? 'ENTRADA',
      };
    }

    const prevCenter =
      this.lastCentroidByTrack.get(trackId) ??
      (trail && trail.length >= 2 ? trail[0] : null);

    this.lastCentroidByTrack.set(trackId, currentCenter);

    if (!enabled || !prevCenter) {
      return { hasCrossed: false, direction: null };
    }

    const { a: lineA, b: lineB } = this.getEndpointsPixels(screenWidth, screenHeight);

    // Comprobar intersección entre el vector (prevCenter -> currentCenter) y (lineA -> lineB)
    let hit = computeSegmentIntersection(prevCenter, currentCenter, lineA, lineB);

    // Si el vehículo se movió rápido entre inferencias, comprobar también con el inicio de su estela reciente
    if (!hit && trail && trail.length >= 3) {
      const olderPoint = trail[Math.max(0, trail.length - 4)];
      hit = computeSegmentIntersection(olderPoint, currentCenter, lineA, lineB);
    }

    if (!hit) {
      return { hasCrossed: false, direction: null };
    }

    // Registrar cruce único
    this.countedTrackIds.add(trackId);
    this.trackDirectionMap.set(trackId, hit.direction);

    const lower = className.toLowerCase();
    let cars = this.stats.cars;
    let twoWheelers = this.stats.twoWheelers;
    let heavyVehicles = this.stats.heavyVehicles;

    if (lower === 'car') {
      cars += 1;
    } else if (lower === 'motorcycle' || lower === 'bicycle') {
      twoWheelers += 1;
    } else {
      heavyVehicles += 1;
    }

    this.stats = {
      total: this.stats.total + 1,
      inbound: hit.direction === 'ENTRADA' ? this.stats.inbound + 1 : this.stats.inbound,
      outbound: hit.direction === 'SALIDA' ? this.stats.outbound + 1 : this.stats.outbound,
      cars,
      twoWheelers,
      heavyVehicles,
      lastCrossingTrackId: trackId,
      lastCrossingClass: className.toUpperCase(),
      lastCrossingDirection: hit.direction,
      lastFlashTimestampMs: nowMs,
      lastIntersectionPoint: hit.point,
    };

    return {
      hasCrossed: true,
      direction: hit.direction,
    };
  }

  public hasTrackCrossed(trackId: number): boolean {
    return this.countedTrackIds.has(trackId);
  }

  public getTrackDirection(trackId: number): CrossingDirection | null {
    return this.trackDirectionMap.get(trackId) ?? null;
  }

  public getStats(): TrafficCountBreakdown {
    return this.stats;
  }

  public resetCounts(): TrafficCountBreakdown {
    this.countedTrackIds.clear();
    this.trackDirectionMap.clear();
    this.lastCentroidByTrack.clear();
    this.stats = {
      total: 0,
      inbound: 0,
      outbound: 0,
      cars: 0,
      twoWheelers: 0,
      heavyVehicles: 0,
      lastCrossingTrackId: null,
      lastCrossingClass: null,
      lastCrossingDirection: null,
      lastFlashTimestampMs: 0,
      lastIntersectionPoint: null,
    };
    return this.stats;
  }
}

export const globalVirtualCountingLine = new VirtualCountingLine();
