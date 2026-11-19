/**
 * Sistema de Alerta Visual y Sonora de Exceso de Velocidad (SpeedAlertSystem.ts)
 * - Evalúa si un vehículo supera el umbral configurable speedLimitKmh (20 - 140 km/h).
 * - Genera un pitido táctico doble mediante Web Audio API nativa (OscillatorNode) sin archivos externos.
 * - Aplica un cooldown de 3,000 ms (3 segundos) por trackId para no saturar el canal de audio.
 * - Mantiene el estado del banner flotante superior: "⚠️ EXCESO DETECTADO: 78 km/h (Límite 60)".
 */

export interface OverspeedBannerAlert {
  trackId: number;
  className: string;
  speedKmh: number;
  limitKmh: number;
  triggeredAtMs: number;
}

const SOUND_COOLDOWN_MS = 3000;
const BANNER_HOLD_MS = 2800;

export class SpeedAlertSystem {
  private audioCtx: AudioContext | null = null;
  private readonly lastSoundByTrack = new Map<number, number>();
  private readonly countedOverspeedTracks = new Set<number>();
  private totalOverspeedCount = 0;
  private activeBanner: OverspeedBannerAlert | null = null;

  /**
   * Desbloquea el AudioContext dentro del gesto onClick del botón "Iniciar Cámara AR".
   * Obligatorio en iOS Safari y Chrome Android para permitir síntesis de audio posterior.
   */
  public unlockAudioContext(): void {
    try {
      if (typeof window === 'undefined') return;
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextClass) return;

      if (!this.audioCtx) {
        this.audioCtx = new AudioContextClass();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume().catch(() => {});
      }

      // Pulso inaudible de 4ms para desbloquear el bus de audio en iOS WebKit
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      gain.gain.setValueAtTime(0.0001, this.audioCtx.currentTime);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start(this.audioCtx.currentTime);
      osc.stop(this.audioCtx.currentTime + 0.004);
    } catch {
      // Ignorar si el navegador restringe Web Audio
    }
  }

  /**
   * Sintetiza un pitido táctico doble (doble beep de radar) usando únicamente Web Audio API.
   * Pulso 1: 1,380 Hz (70 ms) -> Pausa (45 ms) -> Pulso 2: 1,760 Hz (90 ms).
   */
  public playTacticalDoubleBeep(): void {
    try {
      if (typeof window === 'undefined') return;
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextClass) return;

      if (!this.audioCtx) {
        this.audioCtx = new AudioContextClass();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume().catch(() => {});
      }

      const ctx = this.audioCtx;
      const now = ctx.currentTime;

      // Primer pulso (1380 Hz)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = 'triangle';
      osc1.frequency.setValueAtTime(1380, now);
      gain1.gain.setValueAtTime(0.001, now);
      gain1.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.075);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.08);

      // Segundo pulso táctico agudo (1760 Hz)
      const t2 = now + 0.115;
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(1760, t2);
      gain2.gain.setValueAtTime(0.001, t2);
      gain2.gain.exponentialRampToValueAtTime(0.22, t2 + 0.01);
      gain2.gain.exponentialRampToValueAtTime(0.001, t2 + 0.1);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(t2);
      osc2.stop(t2 + 0.105);
    } catch {
      // Silenciar si el dispositivo está en bloqueo de audio
    }
  }

  /**
   * Evalúa un vehículo en el fotograma actual:
   * - Devuelve true si supera speedLimitKmh.
   * - Dispara el doble pitido táctico si han pasado >= 3 segundos desde la última alerta de ese trackId.
   * - Actualiza el banner superior de exceso de velocidad.
   */
  public evaluateVehicleSpeed(params: {
    trackId: number;
    className: string;
    speedKmh: number;
    speedLimitKmh: number;
    soundEnabled: boolean;
    nowMs: number;
  }): boolean {
    const { trackId, className, speedKmh, speedLimitKmh, soundEnabled, nowMs } = params;

    if (speedKmh <= speedLimitKmh) {
      return false;
    }

    if (!this.countedOverspeedTracks.has(trackId)) {
      this.countedOverspeedTracks.add(trackId);
      this.totalOverspeedCount += 1;
    }

    // Actualizar banner flotante con el exceso más reciente o más alto
    if (
      !this.activeBanner ||
      nowMs - this.activeBanner.triggeredAtMs > 1200 ||
      speedKmh >= this.activeBanner.speedKmh
    ) {
      this.activeBanner = {
        trackId,
        className: className.toUpperCase(),
        speedKmh: Math.round(speedKmh),
        limitKmh: Math.round(speedLimitKmh),
        triggeredAtMs: nowMs,
      };
    }

    // Cooldown de 3 segundos por trackId para el pitido táctico doble
    const lastBeepMs = this.lastSoundByTrack.get(trackId) ?? 0;
    if (soundEnabled && nowMs - lastBeepMs >= SOUND_COOLDOWN_MS) {
      this.lastSoundByTrack.set(trackId, nowMs);
      this.playTacticalDoubleBeep();
    }

    return true;
  }

  /**
   * Obtiene la alerta activa para el banner flotante superior si sigue vigente.
   */
  public getActiveBanner(nowMs: number): OverspeedBannerAlert | null {
    if (!this.activeBanner) return null;
    if (nowMs - this.activeBanner.triggeredAtMs > BANNER_HOLD_MS) {
      this.activeBanner = null;
      return null;
    }
    return this.activeBanner;
  }

  public getTotalOverspeedCount(): number {
    return this.totalOverspeedCount;
  }

  public resetCounts(): void {
    this.countedOverspeedTracks.clear();
    this.lastSoundByTrack.clear();
    this.totalOverspeedCount = 0;
    this.activeBanner = null;
  }
}

export const globalSpeedAlertSystem = new SpeedAlertSystem();
