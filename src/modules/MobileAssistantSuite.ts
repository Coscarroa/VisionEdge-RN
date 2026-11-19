/**
 * Módulo 4: Mejoras de Experiencia Móvil, GPS Dashcam, Voz Sintetizada y Grabación AR
 * (src/modules/MobileAssistantSuite.ts)
 *
 * 1. Asistente de Voz en Tiempo Real (Web Speech API - window.speechSynthesis):
 *    - Voz sintetizada en español (es-ES / es-MX) desbloqueada en el gesto "Iniciar Cámara AR".
 *    - Cola de prioridad y cooldown por categoría para evitar saturación acústica.
 * 2. Modo Dashcam con Compensación GPS (navigator.geolocation.watchPosition):
 *    - Lee coords.speed (convertido de m/s a km/h) y coordenadas Lat/Lon en vivo.
 *    - Permite restar la velocidad propia del vehículo de la velocidad relativa medida en pantalla.
 * 3. Grabación de Video AR Compuesto (MediaRecorder + Canvas Compositor 30 FPS):
 *    - Combina el <video> real (con recorte object-fit: cover) y el <canvas> AR en un único
 *      stream (compositeCanvas.captureStream(30)) y lo descarga en .mp4 / .webm con todas las capas AR.
 */

export type VoiceAlertCategory =
  | 'OVERSPEED'
  | 'ROI_VIOLATION'
  | 'FALL_DETECTED'
  | 'COLLISION_RISK'
  | 'SYSTEM';

export interface DashcamGpsState {
  enabled: boolean;
  status: 'idle' | 'searching' | 'active' | 'error';
  egoSpeedKmh: number;
  latitude: number | null;
  longitude: number | null;
  accuracyMeters: number | null;
  formattedCoords: string;
  errorMessage: string | null;
}

export interface ArRecorderState {
  isRecording: boolean;
  durationSec: number;
  mimeType: string;
}

const VOICE_CATEGORY_COOLDOWN_MS: Record<VoiceAlertCategory, number> = {
  OVERSPEED: 5500,
  ROI_VIOLATION: 7000,
  FALL_DETECTED: 6000,
  COLLISION_RISK: 5500,
  SYSTEM: 2500,
};

export class MobileAssistantSuiteManager {
  // 1. Web Speech API
  private voiceUnlocked = false;
  private readonly lastSpokenByCategory = new Map<VoiceAlertCategory, number>();
  private preferredSpanishVoice: SpeechSynthesisVoice | null = null;

  // 2. Geolocation WatchPosition (Dashcam GPS)
  private gpsWatchId: number | null = null;
  private gpsState: DashcamGpsState = {
    enabled: false,
    status: 'idle',
    egoSpeedKmh: 0,
    latitude: null,
    longitude: null,
    accuracyMeters: null,
    formattedCoords: '',
    errorMessage: null,
  };
  private gpsListeners = new Set<(state: DashcamGpsState) => void>();

  // 3. MediaRecorder AR Video Compositor
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private compositorCanvas: HTMLCanvasElement | null = null;
  private compositorRafId: number | null = null;
  private recordTimerId: number | null = null;
  private recordStartMs = 0;
  private recorderState: ArRecorderState = {
    isRecording: false,
    durationSec: 0,
    mimeType: 'video/webm',
  };
  private recorderListeners = new Set<(state: ArRecorderState) => void>();

  /**
   * Desbloquea el motor window.speechSynthesis durante el gesto onClick de "Iniciar Cámara AR"
   * (obligatorio en iOS Safari y Chrome Android).
   */
  public unlockVoiceSynthesis(): void {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      const synth = window.speechSynthesis;
      const selectSpanishVoice = () => {
        const voices = synth.getVoices();
        this.preferredSpanishVoice =
          voices.find((v) => /^es[-_](MX|ES|US|419|AR|CO)/i.test(v.lang)) ||
          voices.find((v) => v.lang.toLowerCase().startsWith('es')) ||
          null;
      };
      selectSpanishVoice();
      synth.onvoiceschanged = selectSpanishVoice;

      if (!this.voiceUnlocked) {
        const silentUtterance = new SpeechSynthesisUtterance('');
        silentUtterance.volume = 0;
        silentUtterance.lang = 'es-ES';
        synth.speak(silentUtterance);
        this.voiceUnlocked = true;
      }
    } catch {
      // Silencioso si el navegador no soporta Web Speech API
    }
  }

  /**
   * Anuncia un evento crítico por voz en español con cola de prioridad y cooldown.
   */
  public speakAlert(params: {
    enabled: boolean;
    category: VoiceAlertCategory;
    message: string;
    priority?: 'HIGH' | 'NORMAL';
    nowMs?: number;
  }): void {
    const { enabled, category, message, priority = 'HIGH', nowMs = performance.now() } = params;
    if (!enabled || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }

    const cooldown = VOICE_CATEGORY_COOLDOWN_MS[category] ?? 5000;
    const lastSpoken = this.lastSpokenByCategory.get(category) ?? 0;
    if (nowMs - lastSpoken < cooldown) {
      return;
    }

    try {
      const synth = window.speechSynthesis;
      if (synth.speaking && priority === 'NORMAL') {
        return;
      }
      if (synth.speaking && priority === 'HIGH') {
        synth.cancel();
      }

      this.lastSpokenByCategory.set(category, nowMs);
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.lang = this.preferredSpanishVoice?.lang || 'es-ES';
      if (this.preferredSpanishVoice) {
        utterance.voice = this.preferredSpanishVoice;
      }
      utterance.rate = 1.06;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      synth.speak(utterance);
    } catch {
      // Silencioso
    }
  }

  // ============================================================================
  // MODO DASHCAM GPS (navigator.geolocation.watchPosition)
  // ============================================================================

  public subscribeGps(listener: (state: DashcamGpsState) => void): () => void {
    this.gpsListeners.add(listener);
    listener(this.gpsState);
    return () => {
      this.gpsListeners.delete(listener);
    };
  }

  private notifyGps(): void {
    this.gpsListeners.forEach((fn) => fn({ ...this.gpsState }));
  }

  public getGpsState(): DashcamGpsState {
    return this.gpsState;
  }

  public setDashcamGpsEnabled(enabled: boolean): void {
    if (enabled === this.gpsState.enabled) return;

    if (!enabled) {
      if (
        this.gpsWatchId !== null &&
        typeof navigator !== 'undefined' &&
        'geolocation' in navigator
      ) {
        navigator.geolocation.clearWatch(this.gpsWatchId);
      }
      this.gpsWatchId = null;
      this.gpsState = {
        enabled: false,
        status: 'idle',
        egoSpeedKmh: 0,
        latitude: null,
        longitude: null,
        accuracyMeters: null,
        formattedCoords: '',
        errorMessage: null,
      };
      this.notifyGps();
      return;
    }

    if (typeof navigator === 'undefined' || !('geolocation' in navigator)) {
      this.gpsState = {
        ...this.gpsState,
        enabled: true,
        status: 'error',
        errorMessage: 'Geolocation API no disponible en este navegador',
      };
      this.notifyGps();
      return;
    }

    this.gpsState = {
      ...this.gpsState,
      enabled: true,
      status: 'searching',
      errorMessage: null,
    };
    this.notifyGps();

    try {
      this.gpsWatchId = navigator.geolocation.watchPosition(
        (pos) => {
          const speedMs = pos.coords.speed;
          const egoSpeedKmh =
            speedMs !== null && !Number.isNaN(speedMs) && speedMs > 0.3
              ? Number((speedMs * 3.6).toFixed(1))
              : 0;
          const lat = Number(pos.coords.latitude.toFixed(5));
          const lon = Number(pos.coords.longitude.toFixed(5));
          const acc = Math.round(pos.coords.accuracy || 0);

          this.gpsState = {
            enabled: true,
            status: 'active',
            egoSpeedKmh,
            latitude: lat,
            longitude: lon,
            accuracyMeters: acc,
            formattedCoords: `${lat}, ${lon} (±${acc}m)`,
            errorMessage: null,
          };
          this.notifyGps();
        },
        (err) => {
          this.gpsState = {
            ...this.gpsState,
            enabled: true,
            status: 'error',
            errorMessage: err.message || 'Permiso de ubicación GPS denegado',
          };
          this.notifyGps();
        },
        {
          enableHighAccuracy: true,
          maximumAge: 1000,
          timeout: 10000,
        }
      );
    } catch {
      this.gpsState = {
        ...this.gpsState,
        enabled: true,
        status: 'error',
        errorMessage: 'No se pudo iniciar watchPosition GPS',
      };
      this.notifyGps();
    }
  }

  // ============================================================================
  // GRABACIÓN DE VIDEO AR COMPUESTO (MediaRecorder + Canvas 30 FPS)
  // ============================================================================

  public subscribeRecorder(listener: (state: ArRecorderState) => void): () => void {
    this.recorderListeners.add(listener);
    listener(this.recorderState);
    return () => {
      this.recorderListeners.delete(listener);
    };
  }

  private notifyRecorder(): void {
    this.recorderListeners.forEach((fn) => fn({ ...this.recorderState }));
  }

  public getRecorderState(): ArRecorderState {
    return this.recorderState;
  }

  public toggleArVideoRecording(params: {
    videoElement: HTMLVideoElement | null;
    arOverlayCanvas: HTMLCanvasElement | null;
    isFrontCamera: boolean;
  }): void {
    if (this.recorderState.isRecording) {
      this.stopArVideoRecording();
    } else {
      this.startArVideoRecording(params);
    }
  }

  public startArVideoRecording(params: {
    videoElement: HTMLVideoElement | null;
    arOverlayCanvas: HTMLCanvasElement | null;
    isFrontCamera: boolean;
  }): boolean {
    const { videoElement, arOverlayCanvas, isFrontCamera } = params;
    if (
      this.recorderState.isRecording ||
      !videoElement ||
      !arOverlayCanvas ||
      typeof window === 'undefined' ||
      typeof MediaRecorder === 'undefined'
    ) {
      return false;
    }

    if (!this.compositorCanvas) {
      this.compositorCanvas = document.createElement('canvas');
    }
    const compCanvas = this.compositorCanvas;

    // Resolución par optimizada para codificación hardware móvil (720p proporcional)
    const targetW = Math.min(1280, Math.max(480, Math.floor(window.innerWidth / 2) * 2));
    const targetH = Math.min(1280, Math.max(480, Math.floor(window.innerHeight / 2) * 2));
    compCanvas.width = targetW;
    compCanvas.height = targetH;

    const ctx = compCanvas.getContext('2d');
    if (!ctx) return false;

    const candidateMimeTypes = [
      'video/mp4;codecs=avc1',
      'video/mp4',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
    ];
    let selectedMimeType = '';
    for (const mime of candidateMimeTypes) {
      if (typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(mime)) {
        selectedMimeType = mime;
        break;
      }
    }

    try {
      const stream = compCanvas.captureStream(30);
      const recorder = selectedMimeType
        ? new MediaRecorder(stream, {
            mimeType: selectedMimeType,
            videoBitsPerSecond: 4_500_000,
          })
        : new MediaRecorder(stream);

      this.recordedChunks = [];
      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) {
          this.recordedChunks.push(ev.data);
        }
      };

      recorder.onstop = () => {
        if (this.recordedChunks.length === 0) return;
        const finalMime = selectedMimeType || recorder.mimeType || 'video/webm';
        const ext = finalMime.includes('mp4') ? 'mp4' : 'webm';
        const blob = new Blob(this.recordedChunks, { type: finalMime });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
        link.href = url;
        link.download = `VisionEdge_AR_Clip_${stamp}.${ext}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.setTimeout(() => URL.revokeObjectURL(url), 5000);
        this.recordedChunks = [];
      };

      // Bucle compositor a 30/60 FPS que funde <video> (con object-fit: cover) + <canvas> AR
      const renderCompositeFrame = () => {
        if (!this.mediaRecorder || this.mediaRecorder.state !== 'recording') return;

        const vw = videoElement.videoWidth || targetW;
        const vh = videoElement.videoHeight || targetH;
        const coverScale = Math.max(targetW / Math.max(1, vw), targetH / Math.max(1, vh));
        const drawW = vw * coverScale;
        const drawH = vh * coverScale;
        const offsetX = (targetW - drawW) * 0.5;
        const offsetY = (targetH - drawH) * 0.5;

        ctx.fillStyle = '#06090F';
        ctx.fillRect(0, 0, targetW, targetH);

        ctx.save();
        if (isFrontCamera) {
          ctx.translate(targetW, 0);
          ctx.scale(-1, 1);
        }
        ctx.drawImage(videoElement, offsetX, offsetY, drawW, drawH);
        ctx.restore();

        // Superponer todas las capas gráficas del HUD AR Canvas
        ctx.drawImage(arOverlayCanvas, 0, 0, targetW, targetH);

        // Sello REC discreto en la esquina superior izquierda del video grabado
        ctx.fillStyle = 'rgba(11, 15, 23, 0.72)';
        ctx.fillRect(14, 14, 196, 26);
        ctx.fillStyle = '#EF4444';
        ctx.beginPath();
        ctx.arc(27, 27, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = '700 11px "JetBrains Mono", monospace';
        ctx.fillStyle = '#F8FAFC';
        ctx.textBaseline = 'middle';
        ctx.fillText(`REC AR • ${new Date().toLocaleTimeString()}`, 38, 27);

        this.compositorRafId = window.requestAnimationFrame(renderCompositeFrame);
      };

      this.mediaRecorder = recorder;
      this.recordStartMs = performance.now();
      recorder.start(250);

      this.recorderState = {
        isRecording: true,
        durationSec: 0,
        mimeType: selectedMimeType || 'video/webm',
      };
      this.notifyRecorder();

      this.compositorRafId = window.requestAnimationFrame(renderCompositeFrame);
      this.recordTimerId = window.setInterval(() => {
        const elapsedSec = Math.max(
          0,
          Math.floor((performance.now() - this.recordStartMs) / 1000)
        );
        this.recorderState = {
          ...this.recorderState,
          durationSec: elapsedSec,
        };
        this.notifyRecorder();
      }, 500);

      return true;
    } catch {
      return false;
    }
  }

  public stopArVideoRecording(): void {
    if (this.compositorRafId !== null && typeof window !== 'undefined') {
      window.cancelAnimationFrame(this.compositorRafId);
      this.compositorRafId = null;
    }
    if (this.recordTimerId !== null && typeof window !== 'undefined') {
      window.clearInterval(this.recordTimerId);
      this.recordTimerId = null;
    }
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        this.mediaRecorder.stop();
      } catch {
        // Silencioso
      }
    }
    this.mediaRecorder = null;
    this.recorderState = {
      ...this.recorderState,
      isRecording: false,
      durationSec: 0,
    };
    this.notifyRecorder();
  }
}

export const globalMobileAssistantSuite = new MobileAssistantSuiteManager();
