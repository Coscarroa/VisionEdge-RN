import React, { useState } from 'react';
import {
  X,
  Pause,
  Play,
  RotateCcw,
  Code2,
  Activity,
  Compass,
  Sparkles,
  Car,
  Volume2,
  VolumeX,
  AlertTriangle,
  GitCommitHorizontal,
} from 'lucide-react';
import {
  ARVisualTheme,
  CLASS_COLOR_MAP,
  GyroStabilizationTelemetry,
  PipelineMetrics,
  PipelineSettings,
  ThermalControlMode,
  TimeSeriesPoint,
  TrackedObject,
} from '../engine/cvTypes';
import { AR_THEMES, AR_THEME_ORDER } from '../config/ARThemeConfig';
import { CodeArchitectureExplorer } from './CodeArchitectureExplorer';
import { DeliverableCategory, RN_ARCHITECTURE_FILES } from '../data/rnArchitectureFiles';
import { OnnxTrafficLiveTelemetry } from '../hooks/useOnnxVisionEngine';
import { globalSpeedAlertSystem } from '../utils/SpeedAlertSystem';

interface DiagnosticsModalProps {
  visible: boolean;
  settings: PipelineSettings;
  metrics: PipelineMetrics;
  tracks: TrackedObject[];
  timeSeries: TimeSeriesPoint[];
  latencyHistory: number[];
  gyroTelemetry: GyroStabilizationTelemetry;
  trafficTelemetry?: OnnxTrafficLiveTelemetry;
  onUpdate: (patch: Partial<PipelineSettings>) => void;
  onResetHeatmap: () => void;
  onResetTrafficCounts?: () => void;
  onToggleLineOrientation?: () => void;
  onClose: () => void;
}

export const DiagnosticsModal: React.FC<DiagnosticsModalProps> = ({
  visible,
  settings,
  metrics,
  tracks,
  timeSeries,
  latencyHistory,
  gyroTelemetry,
  trafficTelemetry,
  onUpdate,
  onResetHeatmap,
  onResetTrafficCounts,
  onToggleLineOrientation,
  onClose,
}) => {
  const [showCodeExplorer, setShowCodeExplorer] = useState(false);
  const [selectedFileId, setSelectedFileId] = useState<string>('vehicle-speed-estimator');
  const [categoryFilter, setCategoryFilter] = useState<DeliverableCategory | 'all'>('all');

  if (!visible) return null;

  const onlyVehicles = Boolean(settings.onlyVehiclesMode);
  const speedLimitKmh = settings.speedLimitKmh ?? 60;
  const soundAlertOn = settings.enableSpeedSoundAlert ?? true;
  const distCalib = settings.distanceCalibrationFactor ?? 1.0;
  const showCountingLine = settings.showCountingLine ?? true;

  const countingStats = trafficTelemetry?.countingStats ?? {
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

  const thermalOptions: Array<{ id: ThermalControlMode; label: string }> = [
    { id: 'auto', label: 'Auto' },
    { id: 'Ultra', label: 'Ultra' },
    { id: 'Balanced', label: 'Balanceado' },
    { id: 'Thermal Safe', label: 'Ahorro Térmico' },
  ];

  const stageRows = [
    {
      name: '1. WebRTC Frame Capture (rvfc)',
      ms: metrics.timings.captureMs,
      color: '#38BDF8',
    },
    {
      name: '2. Letterbox #727272 + Planar NCHW',
      ms: metrics.timings.preprocessMs,
      color: '#818CF8',
    },
    {
      name: '3. YOLO11n ONNX Inference (FP32)',
      ms: metrics.timings.yoloMs,
      color: '#F97316',
    },
    {
      name: onlyVehicles
        ? '4. Pose 17kp (PAUSADO - Modo Vehículos)'
        : '4. Pose Estimation 17 Keypoints',
      ms: onlyVehicles ? 0 : metrics.timings.poseMs,
      color: '#22C55E',
    },
    {
      name: '5. Class-NMS + ByteTrack + Cinemática',
      ms: metrics.timings.nmsTrackMs,
      color: '#10B981',
    },
    {
      name: '6. Canvas HiDPI + Kalman + Tripwire AR',
      ms: metrics.timings.renderMs,
      color: '#A78BFA',
    },
  ];

  const maxStageMs = Math.max(12, ...stageRows.map((s) => s.ms));

  const handleOpenCodeFile = (fileId: string) => {
    const match = RN_ARCHITECTURE_FILES.find((f) => f.id === fileId);
    if (match) {
      setCategoryFilter(match.deliverable);
      setSelectedFileId(match.id);
    }
    setShowCodeExplorer(true);
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-end sm:items-center justify-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="diagnostics-modal-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={`w-full ${
          showCodeExplorer ? 'max-w-5xl' : 'max-w-2xl'
        } max-h-[88dvh] bg-[#0F172A]/95 backdrop-blur-xl border-t border-x sm:border border-[#1E293B] rounded-t-[24px] sm:rounded-[24px] shadow-2xl flex flex-col overflow-hidden`}
      >
        {/* Drag Handle + Header */}
        <div className="px-5 pt-2.5 pb-3 border-b border-[#1E293B] flex flex-col gap-2.5 shrink-0">
          <div className="w-10 h-1 bg-slate-600/80 rounded-full mx-auto" />

          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <h2 id="diagnostics-modal-title" className="text-[15px] font-semibold text-slate-100 truncate">
                Radar de Tráfico AR, Aforo Vehicular y Telemetría ONNX
              </h2>
              <p className="text-[11px] text-slate-400 truncate">
                Cinemática Monocular (km/h), Alerta de Exceso, Línea de Aforo Bidireccional y Kalman IMU
              </p>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => setShowCodeExplorer((prev) => !prev)}
                className={`h-[34px] px-2.5 rounded-lg border text-[11px] font-medium flex items-center gap-1.5 transition-colors cursor-pointer ${
                  showCodeExplorer
                    ? 'bg-sky-500/15 border-sky-400/50 text-sky-300'
                    : 'bg-[#121826] border-[#1E293B] text-slate-300 hover:bg-slate-800'
                }`}
              >
                {showCodeExplorer ? <Activity className="w-3.5 h-3.5" /> : <Code2 className="w-3.5 h-3.5" />}
                <span>{showCodeExplorer ? 'Telemetría' : 'Código TS'}</span>
              </button>

              <button
                type="button"
                onClick={onClose}
                aria-label="Cerrar diagnóstico"
                className="h-[34px] w-[34px] flex items-center justify-center rounded-xl bg-[#121826] hover:bg-slate-800 border border-[#1E293B] text-slate-300 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* ScrollView interno fluido */}
        <div className="flex-1 overflow-y-auto overscroll-contain p-4 sm:p-5 flex flex-col gap-4">
          {showCodeExplorer ? (
            <CodeArchitectureExplorer
              selectedFileId={selectedFileId}
              onSelectFile={setSelectedFileId}
              activeCategoryFilter={categoryFilter}
              onChangeCategoryFilter={setCategoryFilter}
            />
          ) : (
            <>
              {/* Sección 0: Radar de Tráfico, Alerta de Exceso (km/h) y Línea Virtual de Aforo */}
              <section className="bg-[#121826] border border-orange-500/35 rounded-2xl p-4 flex flex-col gap-3.5">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    <Car className="w-4 h-4 text-orange-400" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-100">
                      1. Radar Cinemático Vehicular (km/h) & Aforo Tripwire AR
                    </h3>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => onUpdate({ onlyVehiclesMode: !onlyVehicles })}
                      className={`px-3 py-1.5 rounded-xl text-xs font-bold border transition-colors cursor-pointer flex items-center gap-1.5 ${
                        onlyVehicles
                          ? 'bg-orange-500 text-slate-950 border-orange-400 shadow-md'
                          : 'bg-[#0B0F17] text-slate-300 border-[#1E293B] hover:bg-slate-800'
                      }`}
                    >
                      <Car className="w-3.5 h-3.5" />
                      <span>{onlyVehicles ? 'Modo: Solo Vehículos (8 Clases)' : 'Modo: Universal (80 Clases)'}</span>
                    </button>
                  </div>
                </div>

                {/* Desglose de Aforo Vehicular en Tiempo Real */}
                <div className="rounded-xl bg-[#0B0F17] border border-[#1E293B] p-3 flex flex-col gap-2.5">
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="flex items-center gap-2">
                      <GitCommitHorizontal className="w-4 h-4 text-sky-400" />
                      <span className="text-xs font-semibold text-slate-200">
                        Línea Virtual de Conteo Bidireccional (Tripwire AR)
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => onUpdate({ showCountingLine: !showCountingLine })}
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border cursor-pointer ${
                          showCountingLine
                            ? 'bg-sky-500/20 border-sky-400/50 text-sky-300'
                            : 'bg-slate-800 border-slate-700 text-slate-400'
                        }`}
                      >
                        {showCountingLine ? 'Línea: Visible' : 'Línea: Oculta'}
                      </button>

                      {onToggleLineOrientation && (
                        <button
                          type="button"
                          onClick={onToggleLineOrientation}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[11px] font-medium text-slate-200 cursor-pointer"
                        >
                          Girar H/V
                        </button>
                      )}

                      {onResetTrafficCounts && (
                        <button
                          type="button"
                          onClick={onResetTrafficCounts}
                          className="px-2.5 py-1 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/35 text-[11px] font-semibold text-rose-300 cursor-pointer"
                        >
                          Reiniciar Aforo
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-6 gap-2 text-center font-mono">
                    <div className="p-2 rounded-lg bg-[#121826] border border-[#1E293B]">
                      <div className="text-[10px] text-slate-400">AFORO TOTAL</div>
                      <div className="text-sm font-bold text-sky-400">{countingStats.total}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-[#121826] border border-[#1E293B]">
                      <div className="text-[10px] text-slate-400">↑ ENTRADA</div>
                      <div className="text-sm font-bold text-emerald-400">{countingStats.inbound}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-[#121826] border border-[#1E293B]">
                      <div className="text-[10px] text-slate-400">↓ SALIDA</div>
                      <div className="text-sm font-bold text-amber-400">{countingStats.outbound}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-[#121826] border border-[#1E293B]">
                      <div className="text-[10px] text-slate-400">AUTOS</div>
                      <div className="text-sm font-bold text-slate-100">{countingStats.cars}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-[#121826] border border-[#1E293B]">
                      <div className="text-[10px] text-slate-400">MOTOS/BICIS</div>
                      <div className="text-sm font-bold text-slate-100">{countingStats.twoWheelers}</div>
                    </div>
                    <div className="p-2 rounded-lg bg-[#121826] border border-[#1E293B]">
                      <div className="text-[10px] text-slate-400">BUS/CAMIÓN</div>
                      <div className="text-sm font-bold text-slate-100">{countingStats.heavyVehicles}</div>
                    </div>
                  </div>
                </div>

                {/* Sliders: Límite de Velocidad (20..140 km/h) + Calibración Monocular (0.5x..2.0x) */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-[#0B0F17] border border-[#1E293B] flex flex-col gap-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="speed-limit-slider" className="font-semibold text-slate-200 flex items-center gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                        <span>Límite de Velocidad (Alerta)</span>
                      </label>
                      <span className="font-mono text-rose-400 font-bold tabular-nums">
                        {speedLimitKmh} km/h
                      </span>
                    </div>
                    <input
                      id="speed-limit-slider"
                      type="range"
                      min={20}
                      max={140}
                      step={5}
                      value={speedLimitKmh}
                      onChange={(e) => onUpdate({ speedLimitKmh: parseInt(e.target.value, 10) })}
                      className="w-full accent-rose-500 cursor-pointer"
                    />
                    <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                      <span>20 km/h (Zona Escolar)</span>
                      <span>60 km/h</span>
                      <span>140 km/h (Autopista)</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-[#0B0F17] border border-[#1E293B] flex flex-col gap-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="dist-calib-slider" className="font-semibold text-slate-200">
                        Factor Calibración Distancia Monocular
                      </label>
                      <span className="font-mono text-sky-400 font-bold tabular-nums">
                        {distCalib.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      id="dist-calib-slider"
                      type="range"
                      min={0.5}
                      max={2.0}
                      step={0.05}
                      value={distCalib}
                      onChange={(e) =>
                        onUpdate({ distanceCalibrationFactor: parseFloat(e.target.value) })
                      }
                      className="w-full accent-sky-400 cursor-pointer"
                    />
                    <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                      <span>0.5x (Gran Angular)</span>
                      <span>1.0x (1x Estándar)</span>
                      <span>2.0x (Telefoto)</span>
                    </div>
                  </div>
                </div>

                {/* Interruptor de Alerta Sonora Nativa Web Audio API */}
                <div className="p-3 rounded-xl bg-[#0B0F17] border border-[#1E293B] flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    {soundAlertOn ? (
                      <Volume2 className="w-4 h-4 text-rose-400" />
                    ) : (
                      <VolumeX className="w-4 h-4 text-slate-500" />
                    )}
                    <div>
                      <div className="text-xs font-semibold text-slate-200">
                        Alerta Sonora Táctica de Exceso (Web Audio API OscillatorNode)
                      </div>
                      <div className="text-[10.5px] text-slate-400">
                        Doble pitido sintetizado con cooldown de 3s por vehículo · Excesos acumulados:{' '}
                        <strong className="text-rose-400 font-mono">
                          {trafficTelemetry?.overspeedTotalCount ?? 0}
                        </strong>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => globalSpeedAlertSystem.playTacticalDoubleBeep()}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-[11px] font-medium text-slate-200 cursor-pointer"
                    >
                      Probar Beep
                    </button>
                    <button
                      type="button"
                      onClick={() => onUpdate({ enableSpeedSoundAlert: !soundAlertOn })}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold border cursor-pointer transition-colors ${
                        soundAlertOn
                          ? 'bg-rose-500/20 border-rose-400/50 text-rose-300'
                          : 'bg-slate-800 border-slate-700 text-slate-400'
                      }`}
                    >
                      {soundAlertOn ? 'Sonido: ON' : 'Sonido: OFF'}
                    </button>
                  </div>
                </div>
              </section>

              {/* Sección 1: Selector de Estilo Visual AR (ARThemeConfig.ts) & Giroscopio 60Hz */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-sky-400" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      2. Estilo Visual AR & Estabilización Predictiva IMU (60Hz)
                    </h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleOpenCodeFile('ar-theme-config')}
                    className="text-[11px] font-mono text-sky-400 hover:underline cursor-pointer"
                  >
                    ARThemeConfig.ts
                  </button>
                </div>

                {/* 3 Tarjetas de Tema Visual AR */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {AR_THEME_ORDER.map((themeKey: ARVisualTheme) => {
                    const spec = AR_THEMES[themeKey];
                    const isSelected = settings.arTheme === themeKey;
                    return (
                      <button
                        key={themeKey}
                        type="button"
                        onClick={() => onUpdate({ arTheme: themeKey })}
                        className={`p-3 rounded-xl border text-left transition-all cursor-pointer flex flex-col gap-1 ${
                          isSelected
                            ? 'bg-sky-500/15 border-sky-400 text-slate-100 shadow-md'
                            : 'bg-[#0B0F17] border-[#1E293B] text-slate-300 hover:bg-slate-800/70'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold">{spec.shortLabel}</span>
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{
                              backgroundColor:
                                themeKey === 'tactical'
                                  ? '#38BDF8'
                                  : themeKey === 'industrial'
                                  ? '#FACC15'
                                  : '#FFFFFF',
                            }}
                          />
                        </div>
                        <span className="text-[10.5px] text-slate-400 line-clamp-2 leading-snug">
                          {spec.name}
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Estado del Giroscopio Predictivo */}
                <div className="p-3 rounded-xl bg-[#0B0F17] border border-[#1E293B] flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <Compass
                      className={`w-4 h-4 ${
                        settings.enableGyro ? 'text-emerald-400' : 'text-slate-500'
                      }`}
                    />
                    <div>
                      <div className="text-xs font-semibold text-slate-200">
                        Compensación Inter-Frame con Giroscopio (Kalman + LERP)
                      </div>
                      <div className="text-[10.5px] font-mono text-slate-400 tabular-nums">
                        ω=({gyroTelemetry.omegaX.toFixed(2)}, {gyroTelemetry.omegaY.toFixed(2)} rad/s) •
                        Offset=({gyroTelemetry.offsetX.toFixed(1)}px, {gyroTelemetry.offsetY.toFixed(1)}px)
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleOpenCodeFile('use-inertial-ar-tracker')}
                      className="text-[10.5px] font-mono text-sky-400 hover:underline cursor-pointer"
                    >
                      Ver Hook
                    </button>
                    <button
                      type="button"
                      onClick={() => onUpdate({ enableGyro: !settings.enableGyro })}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold border cursor-pointer transition-colors ${
                        settings.enableGyro
                          ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-300'
                          : 'bg-slate-800 border-slate-700 text-slate-400'
                      }`}
                    >
                      {settings.enableGyro ? 'Gyro 60Hz: ON' : 'Gyro: OFF'}
                    </button>
                  </div>
                </div>
              </section>

              {/* Sección 2: Gráfica de Seguimiento 20s & Onda de Velocidad */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      3. Historial de Seguimiento (20s) & Velocidad de Inferencia
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Series temporales desacopladas del visor AR principal
                    </p>
                  </div>
                  <span className="font-mono text-xs font-bold text-sky-400 tabular-nums">
                    {metrics.realFps.toFixed(1)} FPS ({metrics.timings.totalMs.toFixed(1)} ms)
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-[#0B0F17] border border-[#1E293B] flex flex-col gap-2">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold text-slate-300">Object Tracking (20s)</span>
                      <div className="flex items-center gap-2 font-mono text-[10px]">
                        <span className="text-[#38BDF8]">● Per</span>
                        <span className="text-[#F97316]">● Veh</span>
                        <span className="text-[#10B981]">● Otros</span>
                      </div>
                    </div>
                    <TrackingSvgChart data={timeSeries} />
                  </div>

                  <div className="p-3 rounded-xl bg-[#0B0F17] border border-[#1E293B] flex flex-col gap-2">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold text-slate-300">Processing Speed Wave</span>
                      <span className="font-mono text-[10px] text-emerald-400 tabular-nums">
                        Avg 30f: {metrics.rollingAvgMs.toFixed(1)} ms
                      </span>
                    </div>
                    <LatencyWaveSvg values={latencyHistory} />
                  </div>
                </div>
              </section>

              {/* Sección 3: Controles Rápidos y Gobernanza Térmica */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    4. Controles del Pipeline y Umbrales YOLO ONNX
                  </h3>
                  <button
                    type="button"
                    onClick={() => handleOpenCodeFile('use-onnx-vision-engine')}
                    className="text-[11px] font-mono text-sky-400 hover:underline cursor-pointer"
                  >
                    Ver useOnnxVisionEngine.ts
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <button
                    type="button"
                    onClick={() => onUpdate({ isRunning: !settings.isRunning })}
                    className="min-h-[40px] px-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer whitespace-nowrap"
                  >
                    {settings.isRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    <span>{settings.isRunning ? 'Pausar Pipeline' : 'Reanudar Pipeline'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={onResetHeatmap}
                    className="min-h-[40px] px-3 rounded-xl bg-[#1E293B]/70 hover:bg-slate-700 border border-[#334155] text-slate-100 text-xs font-medium flex items-center justify-center gap-2 transition-colors cursor-pointer whitespace-nowrap"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Reiniciar Radar/Heatmap</span>
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-1 bg-[#0B0F17] p-1 rounded-xl border border-[#1E293B]">
                  {thermalOptions.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => onUpdate({ thermalControlMode: opt.id })}
                      className={`min-h-[32px] px-1 rounded-lg text-[10.5px] font-medium transition-colors cursor-pointer truncate ${
                        settings.thermalControlMode === opt.id
                          ? 'bg-emerald-500 text-slate-950 font-semibold'
                          : 'text-slate-400 hover:text-slate-100'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="diag-conf-slider" className="font-medium text-slate-300">
                        Confidence Threshold
                      </label>
                      <span className="font-mono text-sky-400 font-semibold tabular-nums">
                        {settings.confidenceThreshold.toFixed(2)}
                      </span>
                    </div>
                    <input
                      id="diag-conf-slider"
                      type="range"
                      min={0.1}
                      max={0.9}
                      step={0.01}
                      value={settings.confidenceThreshold}
                      onChange={(e) => onUpdate({ confidenceThreshold: parseFloat(e.target.value) })}
                      className="w-full accent-sky-400 cursor-pointer"
                    />
                  </div>

                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="diag-iou-slider" className="font-medium text-slate-300">
                        NMS IoU Threshold
                      </label>
                      <span className="font-mono text-sky-400 font-semibold tabular-nums">
                        {settings.nmsIouThreshold.toFixed(2)}
                      </span>
                    </div>
                    <input
                      id="diag-iou-slider"
                      type="range"
                      min={0.1}
                      max={0.9}
                      step={0.01}
                      value={settings.nmsIouThreshold}
                      onChange={(e) => onUpdate({ nmsIouThreshold: parseFloat(e.target.value) })}
                      className="w-full accent-sky-400 cursor-pointer"
                    />
                  </div>
                </div>
              </section>

              {/* Sección 4: Desglose de Latencia por Etapa (EMA) */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      5. Desglose de Latencia por Etapa (EMA)
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      ONNX Runtime Web (WebGPU / WASM SIMD) + Cinemática Monocular
                    </p>
                  </div>
                  <div className="text-right font-mono tabular-nums shrink-0">
                    <div className="text-sm font-bold text-sky-400">
                      {metrics.timings.totalMs.toFixed(1)} ms
                    </div>
                    <div className="text-[10.5px] text-slate-400">
                      Real {metrics.realFps.toFixed(1)} vs. Teórico {metrics.theoreticalFps.toFixed(1)} FPS
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  {stageRows.map((stage) => (
                    <div key={stage.name} className="flex flex-col gap-1">
                      <div className="flex items-center justify-between text-xs gap-2">
                        <span className="text-slate-200 font-medium truncate">{stage.name}</span>
                        <span className="font-mono text-slate-100 font-semibold tabular-nums shrink-0">
                          {stage.ms.toFixed(1)} ms
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-[#0B0F17] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-150"
                          style={{
                            width: `${Math.min(100, (stage.ms / maxStageMs) * 100)}%`,
                            backgroundColor: stage.color,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Sección 5: Tabla de Objetos Rastreados con Velocidad / Estado y Aforo */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    6. Tabla de Objetos Rastreados (ByteTrack + Velocidad km/h + Aforo)
                  </h3>
                  <span className="font-mono text-[11px] px-2 py-0.5 rounded-md bg-[#0B0F17] border border-[#1E293B] text-sky-400 tabular-nums shrink-0">
                    {tracks.length} activos
                  </span>
                </div>

                {tracks.length === 0 ? (
                  <div className="py-6 text-center text-xs text-slate-500 border border-dashed border-[#1E293B] rounded-xl">
                    Sin detecciones por encima del umbral {settings.confidenceThreshold.toFixed(2)}.
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-[#1E293B] text-[10.5px] font-medium text-slate-400">
                          <th className="py-2 pr-2">ID</th>
                          <th className="py-2 pr-2">Clase</th>
                          <th className="py-2 pr-2 text-right">Conf</th>
                          <th className="py-2 pr-2 text-right">Velocidad / Estado</th>
                          <th className="py-2 pr-2 text-right">Aforo</th>
                          <th className="py-2 pr-2 text-right">Screen BBox [x, y, w, h]</th>
                          <th className="py-2 text-right">Edad</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1E293B]/60 text-[11px] font-mono tabular-nums">
                        {tracks.map((tr) => {
                          const palette = CLASS_COLOR_MAP[tr.className];
                          const speedVal = tr.speedKmh ?? 0;
                          const isOverspeed = Boolean(tr.isOverspeed);
                          const status = tr.motionStatus ?? 'DETENIDO';
                          return (
                            <tr
                              key={tr.trackId}
                              className={`hover:bg-white/[0.02] ${
                                isOverspeed ? 'bg-rose-950/30' : ''
                              }`}
                            >
                              <td className="py-2 pr-2 text-slate-300">#{tr.trackId}</td>
                              <td className="py-2 pr-2 font-sans">
                                <span
                                  className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border"
                                  style={{
                                    color: isOverspeed ? '#FCA5A5' : palette.stroke,
                                    backgroundColor: isOverspeed
                                      ? 'rgba(239, 68, 68, 0.2)'
                                      : palette.fill,
                                    borderColor: isOverspeed
                                      ? 'rgba(239, 68, 68, 0.5)'
                                      : `${palette.stroke}55`,
                                  }}
                                >
                                  {tr.rawClassName ?? tr.className}
                                </span>
                              </td>
                              <td className="py-2 pr-2 text-right text-slate-200">
                                {tr.confidence.toFixed(2)}
                              </td>
                              <td className="py-2 pr-2 text-right whitespace-nowrap">
                                {isOverspeed ? (
                                  <span className="text-rose-400 font-bold">
                                    ⚠️ {speedVal.toFixed(0)} km/h · EXCESO
                                  </span>
                                ) : status === 'EN MOVIMIENTO' ? (
                                  <span className="text-emerald-400 font-semibold">
                                    {speedVal.toFixed(0)} km/h · MOVIMIENTO
                                  </span>
                                ) : (
                                  <span className="text-slate-400">DETENIDO (0 km/h)</span>
                                )}
                              </td>
                              <td className="py-2 pr-2 text-right whitespace-nowrap">
                                {tr.hasCrossedLine ? (
                                  <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-[10px] font-semibold">
                                    ✓ {tr.crossingDirection ?? 'CRUZÓ'}
                                  </span>
                                ) : (
                                  <span className="text-slate-500">—</span>
                                )}
                              </td>
                              <td className="py-2 pr-2 text-right text-slate-400 whitespace-nowrap">
                                [{Math.round(tr.screenBox.x)}, {Math.round(tr.screenBox.y)},{' '}
                                {Math.round(tr.screenBox.width)}, {Math.round(tr.screenBox.height)}]
                              </td>
                              <td className="py-2 text-right text-slate-400">{tr.age}f</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              {/* Sección 6: Acceso Directo a los Archivos de Arquitectura de Tráfico y ONNX Web AR */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      7. Módulos de Radar Vehicular, Aforo AR y Motor ONNX
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Inspecciona el código completo tipado en TypeScript
                    </p>
                  </div>
                  <span className="font-mono text-[10.5px] px-2 py-0.5 rounded-md bg-orange-500/15 border border-orange-400/35 text-orange-300 shrink-0">
                    Radar + Tripwire AR
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => handleOpenCodeFile('vehicle-speed-estimator')}
                    className="px-3 py-2 rounded-xl bg-[#0B0F17] hover:bg-slate-900 border border-[#1E293B] text-left transition-colors cursor-pointer"
                  >
                    <div className="text-[11px] font-mono font-semibold text-orange-400 truncate">
                      VehicleSpeedEstimator.ts
                    </div>
                    <div className="text-[10px] text-slate-400 truncate">
                      Escala Monocular + EMA 500ms + Histéresis
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleOpenCodeFile('speed-alert-system')}
                    className="px-3 py-2 rounded-xl bg-[#0B0F17] hover:bg-slate-900 border border-[#1E293B] text-left transition-colors cursor-pointer"
                  >
                    <div className="text-[11px] font-mono font-semibold text-rose-400 truncate">
                      SpeedAlertSystem.ts
                    </div>
                    <div className="text-[10px] text-slate-400 truncate">
                      Alerta #EF4444 + Doble Beep Web Audio API
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleOpenCodeFile('virtual-counting-line')}
                    className="px-3 py-2 rounded-xl bg-[#0B0F17] hover:bg-slate-900 border border-[#1E293B] text-left transition-colors cursor-pointer"
                  >
                    <div className="text-[11px] font-mono font-semibold text-sky-400 truncate">
                      VirtualCountingLine.ts
                    </div>
                    <div className="text-[10px] text-slate-400 truncate">
                      Intersección 2D + Set&lt;number&gt; + Aforo ↑↓
                    </div>
                  </button>
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const TrackingSvgChart: React.FC<{ data: TimeSeriesPoint[] }> = ({ data }) => {
  const w = 260;
  const h = 68;
  const maxVal = Math.max(
    6,
    ...data.map((d) => Math.max(d.persons, d.vehicles, d.bicycles))
  );

  const buildPolyline = (getter: (pt: TimeSeriesPoint) => number) =>
    data
      .map((pt, i) => {
        const x = (i / Math.max(1, data.length - 1)) * (w - 8) + 4;
        const y = h - 6 - (getter(pt) / maxVal) * (h - 14);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[68px] overflow-visible">
      <line x1="0" y1={h - 6} x2={w} y2={h - 6} stroke="#1E293B" strokeWidth="1" />
      <line x1="0" y1={h * 0.5} x2={w} y2={h * 0.5} stroke="#1E293B" strokeDasharray="3 3" strokeWidth="1" />
      <polyline
        fill="none"
        stroke="#38BDF8"
        strokeWidth="2"
        points={buildPolyline((d) => d.persons)}
      />
      <polyline
        fill="none"
        stroke="#F97316"
        strokeWidth="1.8"
        points={buildPolyline((d) => d.vehicles)}
      />
      <polyline
        fill="none"
        stroke="#10B981"
        strokeWidth="1.6"
        points={buildPolyline((d) => d.bicycles)}
      />
    </svg>
  );
};

const LatencyWaveSvg: React.FC<{ values: number[] }> = ({ values }) => {
  const w = 260;
  const h = 68;
  const maxMs = Math.max(45, ...values);
  const minMs = Math.min(12, ...values);

  const pts = values
    .map((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * (w - 8) + 4;
      const norm = (v - minMs) / Math.max(1, maxMs - minMs);
      const y = h - 6 - norm * (h - 14);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[68px] overflow-visible">
      <line x1="0" y1={h - 6} x2={w} y2={h - 6} stroke="#1E293B" strokeWidth="1" />
      <polyline fill="none" stroke="#22C55E" strokeWidth="2" points={pts} />
    </svg>
  );
};
