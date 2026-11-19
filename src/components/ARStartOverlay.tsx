import React, { useState } from 'react';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Compass,
  Cpu,
  ExternalLink,
  Loader2,
  RefreshCw,
  ShieldAlert,
  Smartphone,
} from 'lucide-react';

export interface PermissionStepStatus {
  imu: 'idle' | 'requesting' | 'granted' | 'denied' | 'unsupported';
  camera: 'idle' | 'requesting' | 'granted' | 'denied';
  engine: 'idle' | 'loading' | 'ready' | 'error';
}

export interface CameraErrorDetails {
  code: 'INSECURE_CONTEXT' | 'PERMISSION_DENIED' | 'NO_CAMERA_FOUND' | 'HARDWARE_LOCKED' | 'UNKNOWN';
  title: string;
  message: string;
  browserHint: string;
}

export interface ARStartOverlayProps {
  visible: boolean;
  engineStatusText: string;
  activeBackend: string;
  permissionStatus: PermissionStepStatus;
  errorDetails: CameraErrorDetails | null;
  onStartARCamera: () => Promise<void>;
}

/**
 * Solicita explícitamente permiso de sensores inerciales (Giroscopio/Acelerómetro) en iOS 13+ Safari.
 * DEBE invocarse síncronamente dentro de la pila del evento onClick del usuario.
 */
export async function requestMobileInertialSensorsPermission(): Promise<'granted' | 'denied' | 'unsupported'> {
  try {
    let granted = false;
    let supported = false;

    const DM = typeof DeviceMotionEvent !== 'undefined'
      ? (DeviceMotionEvent as unknown as { requestPermission?: () => Promise<'granted' | 'denied'> })
      : null;

    if (DM && typeof DM.requestPermission === 'function') {
      supported = true;
      const res = await DM.requestPermission();
      if (res === 'granted') granted = true;
    }

    const DO = typeof DeviceOrientationEvent !== 'undefined'
      ? (DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<'granted' | 'denied'> })
      : null;

    if (DO && typeof DO.requestPermission === 'function') {
      supported = true;
      const res = await DO.requestPermission();
      if (res === 'granted') granted = true;
    }

    if (!supported) {
      // En Android Chrome / Firefox / Samsung Internet los eventos devicemotion/deviceorientation no requieren prompt explícito en HTTPS
      return typeof window !== 'undefined' && ('DeviceMotionEvent' in window || 'DeviceOrientationEvent' in window)
        ? 'granted'
        : 'unsupported';
    }

    return granted ? 'granted' : 'denied';
  } catch {
    // Capturamos rechazos de iOS Safari sin romper el flujo hacia la cámara
    return 'denied';
  }
}

/**
 * Desbloquea el contexto de audio y vibración háptica dentro del gesto del usuario.
 */
export function unlockAudioAndHapticsOnGesture(): void {
  try {
    if (typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function') {
      navigator.vibrate(18);
    }
    const AudioCtx =
      typeof window !== 'undefined'
        ? window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
        : undefined;
    if (AudioCtx) {
      const ctx = new AudioCtx();
      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
      }
      // Oscilador silencioso de 1ms para desbloquear WebAudio en iOS Safari
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      gain.gain.value = 0.0001;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(0);
      osc.stop(ctx.currentTime + 0.005);
    }
  } catch {
    // Ignorar si el navegador bloquea vibración o WebAudio
  }
}

export const ARStartOverlay: React.FC<ARStartOverlayProps> = ({
  visible,
  engineStatusText,
  activeBackend,
  permissionStatus,
  errorDetails,
  onStartARCamera,
}) => {
  const [isStarting, setIsStarting] = useState(false);

  if (!visible) return null;

  const isSecure = typeof window !== 'undefined' ? window.isSecureContext : true;

  const handlePressStart = async () => {
    if (isStarting) return;
    setIsStarting(true);
    try {
      await onStartARCamera();
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-[#070A10]/95 backdrop-blur-xl flex flex-col items-center justify-between p-5 sm:p-8 text-slate-100 overflow-y-auto">
      {/* Cabecera minimalista de estado del motor */}
      <div className="w-full max-w-md flex items-center justify-between text-xs text-slate-400 pt-1">
        <span className="font-semibold tracking-tight text-slate-200">VisionEdge Web AR</span>
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-300">
          <span>80 Clases COCO</span>
          <span aria-hidden="true">·</span>
          <span className="text-sky-400 uppercase">{activeBackend || 'WebGPU / WASM'}</span>
        </div>
      </div>

      {/* Contenedor central con botón flotante de alto contraste "Iniciar Cámara AR" */}
      <div className="w-full max-w-md my-auto flex flex-col items-center text-center gap-6 py-4">
        {/* Visor óptico central */}
        <div className="relative w-24 h-24 rounded-3xl bg-[#0F172A] border border-sky-400/35 flex items-center justify-center shadow-[0_0_40px_rgba(56,189,248,0.18)]">
          {/* Esquinas tácticas decorativas del icono */}
          <span className="absolute top-2 left-2 w-3 h-3 border-t-2 border-l-2 border-sky-400" />
          <span className="absolute top-2 right-2 w-3 h-3 border-t-2 border-r-2 border-sky-400" />
          <span className="absolute bottom-2 left-2 w-3 h-3 border-b-2 border-l-2 border-sky-400" />
          <span className="absolute bottom-2 right-2 w-3 h-3 border-b-2 border-r-2 border-sky-400" />
          <Camera className="w-10 h-10 text-sky-400" />
        </div>

        <div className="flex flex-col gap-2">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Visor de Realidad Aumentada en Tiempo Real
          </h1>
          <p className="text-sm text-slate-300 leading-relaxed">
            Detección neuronal de las 80 clases de COCO (personas, vehículos, animales y objetos cotidianos), esqueleto de 17 puntos y estabilización inercial por giroscopio directamente en tu navegador móvil.
          </p>
        </div>

        {/* Alerta de contexto no seguro (HTTP en lugar de HTTPS) */}
        {!isSecure && (
          <div className="w-full rounded-2xl bg-rose-950/60 border border-rose-500/40 p-4 text-left flex flex-col gap-2">
            <div className="flex items-center gap-2 text-rose-300 font-semibold text-xs">
              <ShieldAlert className="w-4 h-4 shrink-0" />
              <span>Conexión HTTP Detectada (Se requiere HTTPS)</span>
            </div>
            <p className="text-xs text-rose-200/90 leading-relaxed">
              Los navegadores móviles (iOS Safari y Chrome Android) bloquean <code className="font-mono">navigator.mediaDevices.getUserMedia</code> y el giroscopio en conexiones HTTP no cifradas. Abre esta URL con <strong>https://</strong> o desde <strong>localhost</strong>.
            </p>
          </div>
        )}

        {/* Alerta detallada si ocurrió un error de permisos o hardware de cámara */}
        {errorDetails && (
          <div className="w-full rounded-2xl bg-amber-950/60 border border-amber-500/40 p-4 text-left flex flex-col gap-2.5">
            <div className="flex items-center gap-2 text-amber-300 font-semibold text-xs">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{errorDetails.title}</span>
            </div>
            <p className="text-xs text-amber-100/90 leading-relaxed">
              {errorDetails.message}
            </p>
            <div className="rounded-xl bg-[#090D16]/80 border border-amber-500/25 p-3 text-[11px] text-slate-300 leading-relaxed">
              <strong className="text-amber-300 block mb-1">Cómo habilitar la cámara:</strong>
              {errorDetails.browserHint}
            </div>
          </div>
        )}

        {/* BOTÓN FLOTANTE CENTRAL DE ALTO CONTRASTE: "Iniciar Cámara AR" */}
        <button
          type="button"
          onClick={handlePressStart}
          disabled={isStarting}
          className="w-full min-h-[56px] px-6 py-4 rounded-2xl bg-sky-400 hover:bg-sky-300 active:scale-[0.99] disabled:opacity-60 text-slate-950 font-bold text-base flex items-center justify-center gap-3 shadow-[0_0_32px_rgba(56,189,248,0.38)] transition-all cursor-pointer"
        >
          {isStarting ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin text-slate-950" />
              <span>Inicializando Cámara y Sensores AR...</span>
            </>
          ) : errorDetails ? (
            <>
              <RefreshCw className="w-5 h-5 text-slate-950" />
              <span>Reintentar Iniciar Cámara AR</span>
            </>
          ) : (
            <>
              <Camera className="w-5 h-5 text-slate-950" />
              <span>Iniciar Cámara AR</span>
            </>
          )}
        </button>

        {/* Indicador de estado listo y checklist de compatibilidad */}
        <div className="w-full rounded-2xl bg-[#0F172A]/85 border border-slate-800 p-4 flex flex-col gap-3 text-left">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-300 font-medium">Estado del Motor Neuronal</span>
            <span className="font-mono text-[11px] text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{engineStatusText}</span>
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 pt-1 border-t border-slate-800/80 text-[11px] text-slate-400">
            <div className="flex items-center gap-1.5">
              <Smartphone className="w-3.5 h-3.5 text-sky-400 shrink-0" />
              <span className="truncate">
                Cámara: {permissionStatus.camera === 'granted' ? 'Activa' : 'Lista'}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span className="truncate">
                IMU: {permissionStatus.imu === 'granted' ? '60Hz OK' : 'iOS/Android'}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="truncate">WebGPU / WASM</span>
            </div>
          </div>
        </div>
      </div>

      {/* Pie de compatibilidad multi-navegador */}
      <div className="w-full max-w-md flex items-center justify-center gap-2 text-[11px] text-slate-400 pb-1 text-center">
        <span>Compatible con iOS Safari 15.4+, Chrome Android, Firefox, Edge y Samsung Internet</span>
        <ExternalLink className="w-3 h-3 shrink-0 opacity-60" />
      </div>
    </div>
  );
};
