/**
 * Módulo 1: Evidencia Automática (Photo-Multa AR), Galería en IndexedDB y Exportación CSV
 * (src/modules/EvidenceAndReportsModule.ts)
 *
 * - Captura automática de infracciones (Exceso de velocidad, Invasión/Detención en Polígono ROI,
 *   Posible caída humana o Riesgo crítico de colisión) con cooldown de 5 segundos por trackId.
 * - Extrae un recorte (crop) de alta resolución desde <video> + overlay del objeto usando un <canvas> temporal.
 * - Estampa un cintillo telemétrico inferior con: Fecha/Hora ISO, Clase + Track ID, Velocidad (km/h),
 *   Matrícula ALPR (si disponible), Coordenadas GPS (si activas) y tipo de evento.
 * - Almacena en memoria y en IndexedDB con límite circular configurable (por defecto 50 capturas).
 * - Genera y descarga reportes CSV completos mediante Blob y URL.createObjectURL.
 */

import { CrossingDirection } from '../utils/VirtualCountingLine';
import { VehicleMotionStatus } from '../engine/cvTypes';

export type EvidenceEventType =
  | 'EXCESO_VELOCIDAD'
  | 'INVASION_ZONA_ROI'
  | 'CAIDA_DETECTADA'
  | 'RIESGO_COLISION';

export interface EvidenceCaptureRecord {
  id: string;
  timestampMs: number;
  isoTimestamp: string;
  trackId: number;
  className: string;
  speedKmh: number;
  speedLimitKmh: number;
  eventType: EvidenceEventType;
  eventLabel: string;
  licensePlate: string | null;
  gpsCoordsText: string;
  /** Imagen JPEG codificada en DataURL con recorte + cintillo telemétrico estampado */
  imageDataUrl: string;
}

export interface TelemetryCsvEventLog {
  timestampIso: string;
  trackId: number;
  className: string;
  speedKmh: number;
  egoGpsSpeedKmh: number;
  motionStatus: VehicleMotionStatus;
  crossingDirection: CrossingDirection | 'NINGUNO';
  insideRoi: boolean;
  licensePlate: string;
  gpsCoords: string;
  alertTriggered: string;
}

const DB_NAME = 'VisionEdgeEvidenceDB';
const DB_VERSION = 1;
const STORE_NAME = 'evidences';
const EVIDENCE_COOLDOWN_MS = 5000;
const MAX_CSV_LOG_ENTRIES = 1500;

export class EvidenceAndReportsManager {
  private evidences: EvidenceCaptureRecord[] = [];
  private csvLogs: TelemetryCsvEventLog[] = [];
  private readonly lastCaptureByTrackAndEvent = new Map<string, number>();
  private readonly lastCsvLogByTrack = new Map<number, number>();
  private maxRecords = 50;
  private cropCanvas: HTMLCanvasElement | null = null;
  private dbPromise: Promise<IDBDatabase | null> | null = null;
  private listeners = new Set<() => void>();

  constructor() {
    if (typeof window !== 'undefined') {
      this.initIndexedDB().then(() => this.loadFromIndexedDB());
    }
  }

  public subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notifyListeners(): void {
    this.listeners.forEach((fn) => fn());
  }

  public setMaxRecords(limit: number): void {
    this.maxRecords = Math.max(10, Math.min(100, Math.round(limit)));
    if (this.evidences.length > this.maxRecords) {
      const removed = this.evidences.splice(this.maxRecords);
      removed.forEach((r) => this.deleteFromIndexedDB(r.id));
      this.notifyListeners();
    }
  }

  public getEvidences(): EvidenceCaptureRecord[] {
    return this.evidences;
  }

  public getCsvLogsCount(): number {
    return this.csvLogs.length;
  }

  private initIndexedDB(): Promise<IDBDatabase | null> {
    if (this.dbPromise) return this.dbPromise;
    if (typeof window === 'undefined' || !('indexedDB' in window)) {
      this.dbPromise = Promise.resolve(null);
      return this.dbPromise;
    }

    this.dbPromise = new Promise((resolve) => {
      try {
        const req = window.indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });

    return this.dbPromise;
  }

  private async loadFromIndexedDB(): Promise<void> {
    const db = await this.initIndexedDB();
    if (!db) return;

    try {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => {
        const items = (req.result as EvidenceCaptureRecord[]) || [];
        items.sort((a, b) => b.timestampMs - a.timestampMs);
        this.evidences = items.slice(0, this.maxRecords);
        this.notifyListeners();
      };
    } catch {
      // Ignorar si IndexedDB está restringido en modo incógnito
    }
  }

  private async saveToIndexedDB(record: EvidenceCaptureRecord): Promise<void> {
    const db = await this.initIndexedDB();
    if (!db) return;

    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(record);
    } catch {
      // Continuar con almacenamiento en memoria si falla escritura IDB
    }
  }

  private async deleteFromIndexedDB(id: string): Promise<void> {
    const db = await this.initIndexedDB();
    if (!db) return;

    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(id);
    } catch {
      // Silencioso
    }
  }

  public async deleteEvidence(id: string): Promise<void> {
    this.evidences = this.evidences.filter((item) => item.id !== id);
    await this.deleteFromIndexedDB(id);
    this.notifyListeners();
  }

  public async clearAllEvidences(): Promise<void> {
    this.evidences = [];
    this.lastCaptureByTrackAndEvent.clear();
    const db = await this.initIndexedDB();
    if (db) {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).clear();
      } catch {
        // Silencioso
      }
    }
    this.notifyListeners();
  }

  /**
   * Registra un evento en la bitácora histórica CSV (con throttling de 1s por trackId para estados normales,
   * o registro inmediato cuando se dispara una alerta).
   */
  public recordTelemetryLog(params: {
    nowMs: number;
    trackId: number;
    className: string;
    speedKmh: number;
    egoGpsSpeedKmh: number;
    motionStatus: VehicleMotionStatus;
    crossingDirection: CrossingDirection | null;
    insideRoi: boolean;
    licensePlate: string | null;
    gpsCoordsText: string;
    alertTriggered: string;
    forceImmediate?: boolean;
  }): void {
    const {
      nowMs,
      trackId,
      className,
      speedKmh,
      egoGpsSpeedKmh,
      motionStatus,
      crossingDirection,
      insideRoi,
      licensePlate,
      gpsCoordsText,
      alertTriggered,
      forceImmediate = false,
    } = params;

    const lastLog = this.lastCsvLogByTrack.get(trackId) ?? 0;
    if (!forceImmediate && nowMs - lastLog < 1200) {
      return;
    }
    this.lastCsvLogByTrack.set(trackId, nowMs);

    this.csvLogs.push({
      timestampIso: new Date().toISOString(),
      trackId,
      className,
      speedKmh: Number(speedKmh.toFixed(1)),
      egoGpsSpeedKmh: Number(egoGpsSpeedKmh.toFixed(1)),
      motionStatus,
      crossingDirection: crossingDirection ?? 'NINGUNO',
      insideRoi,
      licensePlate: licensePlate ?? '-',
      gpsCoords: gpsCoordsText || 'LOCAL_EDGE',
      alertTriggered: alertTriggered || 'NORMAL',
    });

    if (this.csvLogs.length > MAX_CSV_LOG_ENTRIES) {
      this.csvLogs.shift();
    }
  }

  /**
   * Captura automática de Foto-Multa AR con cooldown de 5 segundos por (trackId + eventType).
   * Extrae el recorte del <video> con margen contextual, dibuja la retícula AR de infracción
   * y estampa el cintillo telemétrico inferior en alta definición.
   */
  public captureAutomaticEvidence(params: {
    videoElement: HTMLVideoElement | null;
    screenWidth: number;
    screenHeight: number;
    isFrontCamera: boolean;
    trackId: number;
    className: string;
    screenBox: { x: number; y: number; width: number; height: number };
    speedKmh: number;
    speedLimitKmh: number;
    eventType: EvidenceEventType;
    licensePlate: string | null;
    gpsCoordsText: string;
    nowMs: number;
  }): EvidenceCaptureRecord | null {
    const {
      videoElement,
      screenWidth,
      screenHeight,
      isFrontCamera,
      trackId,
      className,
      screenBox,
      speedKmh,
      speedLimitKmh,
      eventType,
      licensePlate,
      gpsCoordsText,
      nowMs,
    } = params;

    if (!videoElement || videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      return null;
    }

    const cooldownKey = `${trackId}_${eventType}`;
    const lastCaptureMs = this.lastCaptureByTrackAndEvent.get(cooldownKey) ?? 0;
    if (nowMs - lastCaptureMs < EVIDENCE_COOLDOWN_MS) {
      return null;
    }
    this.lastCaptureByTrackAndEvent.set(cooldownKey, nowMs);

    if (!this.cropCanvas && typeof document !== 'undefined') {
      this.cropCanvas = document.createElement('canvas');
    }
    const canvas = this.cropCanvas;
    if (!canvas) return null;

    const outW = 520;
    const outH = 380;
    const footerH = 78;
    const viewportH = outH - footerH;

    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // 1. Convertir coordenadas de pantalla (object-fit: cover) a píxeles nativos del <video>
    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;
    const coverScale = Math.max(
      screenWidth / Math.max(1, vw),
      screenHeight / Math.max(1, vh)
    );
    const renderedW = vw * coverScale;
    const renderedH = vh * coverScale;
    const cropOffsetX = (renderedW - screenWidth) * 0.5;
    const cropOffsetY = (renderedH - screenHeight) * 0.5;

    // Expandir el recorte un 55% alrededor del objeto para incluir contexto vial claro
    const padX = Math.max(44, screenBox.width * 0.55);
    const padY = Math.max(44, screenBox.height * 0.55);

    const screenCropX = Math.max(0, screenBox.x - padX);
    const screenCropY = Math.max(0, screenBox.y - padY);
    const screenCropW = Math.min(screenWidth - screenCropX, screenBox.width + padX * 2);
    const screenCropH = Math.min(screenHeight - screenCropY, screenBox.height + padY * 2);

    let srcX = (screenCropX + cropOffsetX) / coverScale;
    const srcY = (screenCropY + cropOffsetY) / coverScale;
    const srcW = Math.max(32, screenCropW / coverScale);
    const srcH = Math.max(32, screenCropH / coverScale);

    if (isFrontCamera) {
      srcX = Math.max(0, vw - srcX - srcW);
    }

    ctx.fillStyle = '#06090F';
    ctx.fillRect(0, 0, outW, outH);

    // 2. Dibujar recorte del video real
    ctx.save();
    if (isFrontCamera) {
      ctx.translate(outW, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(
      videoElement,
      Math.max(0, srcX),
      Math.max(0, srcY),
      Math.min(vw, srcW),
      Math.min(vh, srcH),
      0,
      0,
      outW,
      viewportH
    );
    ctx.restore();

    // 3. Dibujar retícula táctica de infracción sobre el vehículo/persona recortado
    const relX = ((screenBox.x - screenCropX) / Math.max(1, screenCropW)) * outW;
    const relY = ((screenBox.y - screenCropY) / Math.max(1, screenCropH)) * viewportH;
    const relW = (screenBox.width / Math.max(1, screenCropW)) * outW;
    const relH = (screenBox.height / Math.max(1, screenCropH)) * viewportH;

    const accentColor =
      eventType === 'EXCESO_VELOCIDAD'
        ? '#EF4444'
        : eventType === 'INVASION_ZONA_ROI'
        ? '#F97316'
        : '#E11D48';

    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(relX, relY, relW, relH);

    // Esquinas tácticas (Corner Brackets)
    const cLen = Math.min(22, relW * 0.28, relH * 0.28);
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(relX, relY + cLen);
    ctx.lineTo(relX, relY);
    ctx.lineTo(relX + cLen, relY);
    ctx.moveTo(relX + relW - cLen, relY);
    ctx.lineTo(relX + relW, relY);
    ctx.lineTo(relX + relW, relY + cLen);
    ctx.moveTo(relX, relY + relH - cLen);
    ctx.lineTo(relX, relY + relH);
    ctx.lineTo(relX + cLen, relY + relH);
    ctx.moveTo(relX + relW - cLen, relY + relH);
    ctx.lineTo(relX + relW, relY + relH);
    ctx.lineTo(relX + relW, relY + relH - cLen);
    ctx.stroke();

    // Sello superior de evidencia en el recorte
    const eventLabelMap: Record<EvidenceEventType, string> = {
      EXCESO_VELOCIDAD: `EXCESO DE VELOCIDAD (${Math.round(speedKmh)} km/h > Lím ${speedLimitKmh})`,
      INVASION_ZONA_ROI: 'DETENCIÓN PROHIBIDA EN ZONA ROI',
      CAIDA_DETECTADA: 'ALERTA BIOMECÁNICA: POSIBLE CAÍDA',
      RIESGO_COLISION: 'RIESGO CRÍTICO DE COLISIÓN PEATÓN-VEHÍCULO',
    };
    const eventLabel = eventLabelMap[eventType];

    ctx.fillStyle = 'rgba(127, 29, 29, 0.88)';
    ctx.fillRect(10, 10, outW - 20, 26);
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(10, 10, outW - 20, 26);

    ctx.font = '700 11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#FEE2E2';
    ctx.textBaseline = 'middle';
    ctx.fillText(`PHOTO-MULTA AR • ${eventLabel}`, 18, 23);

    // 4. Estampar Cintillo Telemétrico Inferior
    ctx.fillStyle = '#090D16';
    ctx.fillRect(0, viewportH, outW, footerH);
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, viewportH);
    ctx.lineTo(outW, viewportH);
    ctx.stroke();

    const nowDate = new Date();
    const isoTimestamp = nowDate.toISOString();
    const readableTime = isoTimestamp.replace('T', ' ').slice(0, 23) + ' UTC';
    const plateBadge = licensePlate ? ` • PLACA: ${licensePlate}` : '';

    ctx.font = '700 12px "JetBrains Mono", monospace';
    ctx.fillStyle = '#F8FAFC';
    ctx.fillText(
      `OBJ: ${className.toUpperCase()} #${trackId} • VEL: ${Math.round(speedKmh)} km/h${plateBadge}`,
      14,
      viewportH + 20
    );

    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#38BDF8';
    ctx.fillText(`TS: ${readableTime}`, 14, viewportH + 42);

    ctx.fillStyle = '#94A3B8';
    ctx.fillText(
      `GPS: ${gpsCoordsText || 'SENSOR LOCAL ON-DEVICE'} • EDGE AI VERIFIED`,
      14,
      viewportH + 62
    );

    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.85);

    const record: EvidenceCaptureRecord = {
      id: `ev_${nowDate.getTime()}_${trackId}_${Math.random().toString(36).slice(2, 6)}`,
      timestampMs: nowMs,
      isoTimestamp,
      trackId,
      className: className.toUpperCase(),
      speedKmh: Math.round(speedKmh),
      speedLimitKmh,
      eventType,
      eventLabel,
      licensePlate,
      gpsCoordsText: gpsCoordsText || 'LOCAL EDGE',
      imageDataUrl,
    };

    // Insertar al inicio y mantener buffer circular <= maxRecords
    this.evidences.unshift(record);
    this.saveToIndexedDB(record);

    if (this.evidences.length > this.maxRecords) {
      const removed = this.evidences.splice(this.maxRecords);
      removed.forEach((old) => this.deleteFromIndexedDB(old.id));
    }

    this.notifyListeners();
    return record;
  }

  /**
   * Actualiza la placa ALPR de una evidencia ya capturada cuando el Web Worker de Tesseract
   * termina de procesar el recorte segundos después.
   */
  public attachLicensePlateToTrackEvidences(trackId: number, plate: string): void {
    let changed = false;
    for (const ev of this.evidences) {
      if (ev.trackId === trackId && !ev.licensePlate) {
        ev.licensePlate = plate;
        this.saveToIndexedDB(ev);
        changed = true;
      }
    }
    for (const log of this.csvLogs) {
      if (log.trackId === trackId && log.licensePlate === '-') {
        log.licensePlate = plate;
      }
    }
    if (changed) {
      this.notifyListeners();
    }
  }

  /**
   * Descarga una foto-multa individual en formato .jpg
   */
  public downloadEvidenceJpg(record: EvidenceCaptureRecord): void {
    if (typeof document === 'undefined') return;
    const link = document.createElement('a');
    link.href = record.imageDataUrl;
    const safeTs = record.isoTimestamp.replace(/[:.]/g, '-');
    link.download = `VisionEdge_Evidencia_${record.className}_ID${record.trackId}_${safeTs}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  /**
   * Genera y descarga un archivo .csv con el historial completo de eventos y evidencias
   * usando Blob y URL.createObjectURL.
   */
  public exportTelemetryCSV(): void {
    if (typeof document === 'undefined') return;

    const headers = [
      'timestamp_iso',
      'track_id',
      'clase',
      'velocidad_kmh',
      'velocidad_gps_dashcam_kmh',
      'estado_movimiento',
      'direccion_cruce_aforo',
      'en_zona_roi',
      'placa_alpr',
      'coordenadas_gps',
      'alerta_disparada',
    ];

    const rows: string[][] = [];

    // Primero volcamos todas las evidencias capturadas y luego los logs telemétricos
    for (const ev of this.evidences) {
      rows.push([
        ev.isoTimestamp,
        String(ev.trackId),
        ev.className,
        String(ev.speedKmh),
        '0',
        ev.speedKmh > 4.0 ? 'EN MOVIMIENTO' : 'DETENIDO',
        'EVIDENCIA_FOTO',
        ev.eventType === 'INVASION_ZONA_ROI' ? 'SI' : 'NO',
        ev.licensePlate ?? '-',
        ev.gpsCoordsText,
        ev.eventLabel,
      ]);
    }

    for (const log of this.csvLogs) {
      rows.push([
        log.timestampIso,
        String(log.trackId),
        log.className,
        String(log.speedKmh),
        String(log.egoGpsSpeedKmh),
        log.motionStatus,
        log.crossingDirection,
        log.insideRoi ? 'SI' : 'NO',
        log.licensePlate,
        log.gpsCoords,
        log.alertTriggered,
      ]);
    }

    const escapeCsvCell = (val: string) => {
      const cleaned = String(val ?? '').replace(/"/g, '""');
      return `"${cleaned}"`;
    };

    const csvContent =
      '\uFEFF' +
      headers.join(',') +
      '\n' +
      rows.map((r) => r.map(escapeCsvCell).join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    link.href = url;
    link.download = `VisionEdge_Reporte_Trafico_${stamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
}

export const globalEvidenceManager = new EvidenceAndReportsManager();
