import React, { useState } from 'react';
import { X, Pause, Play, RotateCcw, Code2, Activity } from 'lucide-react';
import {
  CLASS_COLOR_MAP,
  PipelineMetrics,
  PipelineSettings,
  ThermalControlMode,
  TrackedObject,
} from '../engine/cvTypes';
import { CodeArchitectureExplorer } from './CodeArchitectureExplorer';
import { DeliverableCategory, RN_ARCHITECTURE_FILES } from '../data/rnArchitectureFiles';

interface DiagnosticsModalProps {
  visible: boolean;
  settings: PipelineSettings;
  metrics: PipelineMetrics;
  tracks: TrackedObject[];
  onUpdate: (patch: Partial<PipelineSettings>) => void;
  onResetHeatmap: () => void;
  onClose: () => void;
}

export const DiagnosticsModal: React.FC<DiagnosticsModalProps> = ({
  visible,
  settings,
  metrics,
  tracks,
  onUpdate,
  onResetHeatmap,
  onClose,
}) => {
  const [showCodeExplorer, setShowCodeExplorer] = useState(false);
  const [selectedFileId, setSelectedFileId] = useState<string>('diagnostics-modal');
  const [categoryFilter, setCategoryFilter] = useState<DeliverableCategory | 'all'>('all');

  // Evita trabajo de render y cálculos mientras el modal esté cerrado
  if (!visible) return null;

  const thermalOptions: Array<{ id: ThermalControlMode; label: string }> = [
    { id: 'auto', label: 'Auto' },
    { id: 'Ultra', label: 'Ultra' },
    { id: 'Balanced', label: 'Balanceado' },
    { id: 'Thermal Safe', label: 'Ahorro Térmico' },
  ];

  const stageRows = [
    {
      name: '1. VisionCamera JSI Capture',
      ms: metrics.timings.captureMs,
      color: '#38BDF8',
    },
    {
      name: '2. Letterbox Resize 640x640',
      ms: metrics.timings.preprocessMs,
      color: '#818CF8',
    },
    {
      name: '3. YOLO Detection Inference',
      ms: metrics.timings.yoloMs,
      color: '#F97316',
    },
    {
      name: '4. Pose Estimation 17 Keypoints',
      ms: metrics.timings.poseMs,
      color: '#22C55E',
    },
    {
      name: '5. Class-Aware NMS + ByteTrack',
      ms: metrics.timings.nmsTrackMs,
      color: '#10B981',
    },
    {
      name: '6. React Native Skia Render',
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
          showCodeExplorer ? 'max-w-5xl' : 'max-w-xl'
        } max-h-[85dvh] bg-[#0F172A] border-t border-x sm:border border-[#1E293B] rounded-t-[24px] sm:rounded-[24px] shadow-2xl flex flex-col overflow-hidden`}
      >
        {/* Drag Handle + Header */}
        <div className="px-5 pt-2.5 pb-3 border-b border-[#1E293B] flex flex-col gap-2.5 shrink-0">
          <div className="w-10 h-1 bg-slate-600/80 rounded-full mx-auto" />

          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <h2 id="diagnostics-modal-title" className="text-[15px] font-semibold text-slate-100 truncate">
                Ajustes y Telemetría Avanzada
              </h2>
              <p className="text-[11px] text-slate-400 truncate">
                Control térmico, calibración YOLO/NMS, latencia EMA y ByteTrack
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
                <span>{showCodeExplorer ? 'Telemetría' : 'Código RN'}</span>
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
              {/* Sección 1: Controles Rápidos y Gobernanza Térmica */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    1. Controles Rápidos y Gobernanza Térmica
                  </h3>
                  <button
                    type="button"
                    onClick={() => handleOpenCodeFile('diagnostics-modal')}
                    className="text-[11px] font-mono text-sky-400 hover:underline cursor-pointer"
                  >
                    Ver RN
                  </button>
                </div>

                {/* Fila principal de acciones: Pausar/Reanudar + Reiniciar Heatmap */}
                <div className="grid grid-cols-2 gap-2.5">
                  <button
                    type="button"
                    onClick={() => onUpdate({ isRunning: !settings.isRunning })}
                    className="min-h-[42px] px-3 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 text-xs font-semibold flex items-center justify-center gap-2 transition-colors cursor-pointer whitespace-nowrap"
                  >
                    {settings.isRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    <span>{settings.isRunning ? 'Pausar Pipeline' : 'Reanudar Pipeline'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={onResetHeatmap}
                    className="min-h-[42px] px-3 rounded-xl bg-[#1E293B]/70 hover:bg-slate-700 border border-[#334155] text-slate-100 text-xs font-medium flex items-center justify-center gap-2 transition-colors cursor-pointer whitespace-nowrap"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Reiniciar Heatmap</span>
                  </button>
                </div>

                {/* Modificadores secundarios de escenario y aceleración */}
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => onUpdate({ crowdBoost: !settings.crowdBoost })}
                    className={`min-h-[34px] px-2 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer truncate ${
                      settings.crowdBoost
                        ? 'bg-amber-500/15 border-amber-400/40 text-amber-300'
                        : 'bg-[#0B0F17] border-[#1E293B] text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    {settings.crowdBoost ? 'Multitud: Alta' : 'Densidad Alta'}
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      onUpdate({
                        delegate: settings.delegate === 'cpu-xnnpack' ? 'coreml-metal' : 'cpu-xnnpack',
                      })
                    }
                    className={`min-h-[34px] px-2 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer truncate ${
                      settings.delegate === 'cpu-xnnpack'
                        ? 'bg-rose-500/15 border-rose-400/40 text-rose-300'
                        : 'bg-[#0B0F17] border-[#1E293B] text-slate-300 hover:bg-slate-800'
                    }`}
                  >
                    {settings.delegate === 'cpu-xnnpack' ? 'CPU XNNPACK' : 'NPU / Metal'}
                  </button>

                  <button
                    type="button"
                    onClick={() => onUpdate({ enablePose: !settings.enablePose })}
                    className={`min-h-[34px] px-2 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer truncate ${
                      settings.enablePose
                        ? 'bg-emerald-500/15 border-emerald-400/40 text-emerald-300'
                        : 'bg-[#0B0F17] border-[#1E293B] text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {settings.enablePose ? 'Pose 17kp: ON' : 'Pose 17kp: OFF'}
                  </button>
                </div>

                {/* Selector segmentado de modo térmico */}
                <div className="flex flex-col gap-1.5">
                  <div className="grid grid-cols-4 gap-1 bg-[#0B0F17] p-1 rounded-xl border border-[#1E293B]">
                    {thermalOptions.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => onUpdate({ thermalControlMode: opt.id })}
                        className={`min-h-[34px] px-1 rounded-lg text-[10.5px] font-medium transition-colors cursor-pointer truncate ${
                          settings.thermalControlMode === opt.id
                            ? 'bg-emerald-500 text-slate-950 font-semibold'
                            : 'text-slate-400 hover:text-slate-100'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 tabular-nums px-0.5">
                    <span>
                      Activo: <strong className="text-slate-200">{metrics.thermalMode}</strong>
                    </span>
                    <span>Avg 30f: {metrics.rollingAvgMs.toFixed(1)} ms</span>
                  </div>
                </div>
              </section>

              {/* Sección 2: Sliders de Calibración en Tiempo Real */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3.5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  2. Sliders de Calibración en Tiempo Real
                </h3>

                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="diag-conf-slider" className="font-medium text-slate-200">
                      Confidence Threshold
                    </label>
                    <span className="font-mono text-sky-400 font-semibold tabular-nums">
                      {settings.confidenceThreshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    id="diag-conf-slider"
                    type="range"
                    min={0.10}
                    max={0.90}
                    step={0.01}
                    value={settings.confidenceThreshold}
                    onChange={(e) => onUpdate({ confidenceThreshold: parseFloat(e.target.value) })}
                    className="w-full accent-sky-400 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>0.10</span>
                    <span>0.90</span>
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="diag-iou-slider" className="font-medium text-slate-200">
                      NMS IoU Threshold
                    </label>
                    <span className="font-mono text-sky-400 font-semibold tabular-nums">
                      {settings.nmsIouThreshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    id="diag-iou-slider"
                    type="range"
                    min={0.10}
                    max={0.90}
                    step={0.01}
                    value={settings.nmsIouThreshold}
                    onChange={(e) => onUpdate({ nmsIouThreshold: parseFloat(e.target.value) })}
                    className="w-full accent-sky-400 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>0.10</span>
                    <span>0.90</span>
                  </div>
                </div>
              </section>

              {/* Sección 3: Panel de Latencia por Etapa (EMA) */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      3. Panel de Latencia por Etapa (EMA)
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Instrumentación JSI Worklet en tiempo real
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

                <div className="flex flex-col gap-2.5">
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

              {/* Sección 4: Tabla de Objetos Rastreados (ByteTrack + NMS) */}
              <section className="bg-[#121826] border border-[#1E293B] rounded-2xl p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    4. Tabla de Objetos Rastreados (ByteTrack + NMS)
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
                          <th className="py-2 pr-2 text-right">Screen BBox [x, y, w, h]</th>
                          <th className="py-2 pr-2 text-right">Keypoints</th>
                          <th className="py-2 text-right">Edad</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1E293B]/60 text-[11px] font-mono tabular-nums">
                        {tracks.map((tr) => {
                          const palette = CLASS_COLOR_MAP[tr.className];
                          return (
                            <tr key={tr.trackId} className="hover:bg-white/[0.02]">
                              <td className="py-2 pr-2 text-slate-300">#{tr.trackId}</td>
                              <td className="py-2 pr-2 font-sans">
                                <span
                                  className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border"
                                  style={{
                                    color: palette.stroke,
                                    backgroundColor: palette.fill,
                                    borderColor: `${palette.stroke}55`,
                                  }}
                                >
                                  {tr.className}
                                </span>
                              </td>
                              <td className="py-2 pr-2 text-right text-slate-200">
                                {tr.confidence.toFixed(2)}
                              </td>
                              <td className="py-2 pr-2 text-right text-slate-400 whitespace-nowrap">
                                [{Math.round(tr.screenBox.x)}, {Math.round(tr.screenBox.y)},{' '}
                                {Math.round(tr.screenBox.width)}, {Math.round(tr.screenBox.height)}]
                              </td>
                              <td className="py-2 pr-2 text-right text-emerald-400">
                                {tr.keypoints.length > 0 ? `${tr.keypoints.length}/17` : '-'}
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
            </>
          )}
        </div>
      </div>
    </div>
  );
};
