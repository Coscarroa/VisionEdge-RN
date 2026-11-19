/**
 * VisionEdge RN — Sub-millisecond Pipeline Profiler & Adaptive Thermal Governor
 * Runs inside JSI Worklets ('worklet') with preallocated ring buffers (zero GC).
 */

export type ThermalMode = 'Ultra' | 'Balanced' | 'Thermal Safe';

export interface StageLatencyEma {
  captureMs: number;
  resizeMs: number;
  yoloMs: number;
  poseMs: number;
  nmsTrackMs: number;
  skiaRenderMs: number;
  totalMs: number;
  theoreticalFps: number;
  realFps: number;
}

export interface ThermalDecision {
  mode: ThermalMode;
  rollingAvgMs: number;
  shouldRunPose: boolean;
  shouldUpdateHeatmap: boolean;
  poseStride: number;
  heatmapStride: number;
}

/**
 * 1. Instrumentación de Latencia por Etapa (PipelineProfiler)
 * Suaviza las 6 etapas mediante Media Móvil Exponencial (EMA) y mide
 * Throughput teórico (1000 / totalMs) vs Throughput real entre fotogramas.
 */
export class PipelineProfiler {
  private readonly alpha: number;
  private initialized = false;
  private lastFrameTimestamp = 0;

  private emaCapture = 0;
  private emaResize = 0;
  private emaYolo = 0;
  private emaPose = 0;
  private emaNmsTrack = 0;
  private emaSkia = 0;
  private emaRealFps = 30;

  constructor(emaAlpha = 0.18) {
    this.alpha = emaAlpha;
  }

  public reset(): void {
    'worklet';
    this.initialized = false;
    this.lastFrameTimestamp = 0;
  }

  public recordFrame(
    captureMs: number,
    resizeMs: number,
    yoloMs: number,
    poseMs: number,
    nmsTrackMs: number,
    skiaRenderMs: number,
    nowTimestampMs: number
  ): StageLatencyEma {
    'worklet';
    const dtFrame =
      this.lastFrameTimestamp > 0 ? Math.max(1, nowTimestampMs - this.lastFrameTimestamp) : 33.3;
    this.lastFrameTimestamp = nowTimestampMs;
    const instantRealFps = Math.min(60, 1000 / dtFrame);

    if (!this.initialized) {
      this.emaCapture = captureMs;
      this.emaResize = resizeMs;
      this.emaYolo = yoloMs;
      this.emaPose = poseMs;
      this.emaNmsTrack = nmsTrackMs;
      this.emaSkia = skiaRenderMs;
      this.emaRealFps = instantRealFps;
      this.initialized = true;
    } else {
      const a = this.alpha;
      const b = 1 - a;
      this.emaCapture = a * captureMs + b * this.emaCapture;
      this.emaResize = a * resizeMs + b * this.emaResize;
      this.emaYolo = a * yoloMs + b * this.emaYolo;
      this.emaPose = a * poseMs + b * this.emaPose;
      this.emaNmsTrack = a * nmsTrackMs + b * this.emaNmsTrack;
      this.emaSkia = a * skiaRenderMs + b * this.emaSkia;
      this.emaRealFps = a * instantRealFps + b * this.emaRealFps;
    }

    const totalMs =
      this.emaCapture +
      this.emaResize +
      this.emaYolo +
      this.emaPose +
      this.emaNmsTrack +
      this.emaSkia;

    const theoreticalFps = Math.min(120, 1000 / Math.max(0.5, totalMs));

    return {
      captureMs: this.emaCapture,
      resizeMs: this.emaResize,
      yoloMs: this.emaYolo,
      poseMs: this.emaPose,
      nmsTrackMs: this.emaNmsTrack,
      skiaRenderMs: this.emaSkia,
      totalMs,
      theoreticalFps,
      realFps: this.emaRealFps,
    };
  }
}

/**
 * 2. Control Automático de Estrangulamiento Térmico (AdaptiveThermalController)
 * Monitorea un buffer circular de los últimos 30 fotogramas y aplica degradación elegante:
 *  - < 33 ms:      Ultra (Detección + Pose cada fotograma)
 *  - 33 ms..55 ms: Balanced (Detección cada fotograma, Pose cada 2 fotogramas)
 *  - > 55 ms:      Thermal Safe (Pose pausada, Heatmap cada 4 fotogramas)
 */
export class AdaptiveThermalController {
  private readonly windowSize = 30;
  private readonly ringBuffer = new Float32Array(30);
  private writeIndex = 0;
  private sampleCount = 0;
  private rollingSum = 0;
  private frameCounter = 0;
  private currentMode: ThermalMode = 'Ultra';

  public reset(): void {
    'worklet';
    this.ringBuffer.fill(0);
    this.writeIndex = 0;
    this.sampleCount = 0;
    this.rollingSum = 0;
    this.frameCounter = 0;
    this.currentMode = 'Ultra';
  }

  public evaluate(totalFrameLatencyMs: number): ThermalDecision {
    'worklet';
    this.frameCounter++;

    if (this.sampleCount < this.windowSize) {
      this.ringBuffer[this.writeIndex] = totalFrameLatencyMs;
      this.rollingSum += totalFrameLatencyMs;
      this.sampleCount++;
    } else {
      const oldest = this.ringBuffer[this.writeIndex];
      this.ringBuffer[this.writeIndex] = totalFrameLatencyMs;
      this.rollingSum += totalFrameLatencyMs - oldest;
    }

    this.writeIndex = (this.writeIndex + 1) % this.windowSize;
    const avgMs = this.rollingSum / Math.max(1, this.sampleCount);

    // Transiciones con histéresis de 2ms para evitar oscilaciones en el borde
    if (this.currentMode === 'Ultra') {
      if (avgMs > 55) this.currentMode = 'Thermal Safe';
      else if (avgMs >= 33) this.currentMode = 'Balanced';
    } else if (this.currentMode === 'Balanced') {
      if (avgMs > 55) this.currentMode = 'Thermal Safe';
      else if (avgMs < 31) this.currentMode = 'Ultra';
    } else {
      // Thermal Safe
      if (avgMs < 31) this.currentMode = 'Ultra';
      else if (avgMs <= 52) this.currentMode = 'Balanced';
    }

    if (this.currentMode === 'Ultra') {
      return {
        mode: 'Ultra',
        rollingAvgMs: avgMs,
        shouldRunPose: true,
        shouldUpdateHeatmap: true,
        poseStride: 1,
        heatmapStride: 1,
      };
    }

    if (this.currentMode === 'Balanced') {
      return {
        mode: 'Balanced',
        rollingAvgMs: avgMs,
        shouldRunPose: this.frameCounter % 2 === 0,
        shouldUpdateHeatmap: true,
        poseStride: 2,
        heatmapStride: 1,
      };
    }

    return {
      mode: 'Thermal Safe',
      rollingAvgMs: avgMs,
      shouldRunPose: false,
      shouldUpdateHeatmap: this.frameCounter % 4 === 0,
      poseStride: 0,
      heatmapStride: 4,
    };
  }
}
