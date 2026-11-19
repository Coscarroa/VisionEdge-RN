import React, { useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import { CameraErrorDetails } from './ARStartOverlay';

export type WebCameraFacingMode = 'environment' | 'user';

export interface WebCameraStreamHandle {
  startStream: (facing?: WebCameraFacingMode) => Promise<HTMLVideoElement | null>;
  stopStream: () => void;
  switchCamera: () => Promise<WebCameraFacingMode>;
  getVideoElement: () => HTMLVideoElement | null;
}

export interface WebCameraStreamProps {
  facingMode: WebCameraFacingMode;
  onStreamReady: (video: HTMLVideoElement, actualFacing: WebCameraFacingMode) => void;
  onStreamError: (err: CameraErrorDetails) => void;
}

/**
 * Clasifica errores de WebRTC getUserMedia en diagnósticos accionables para iOS Safari,
 * Chrome Android, Firefox, Edge y Samsung Internet.
 */
export function classifyGetUserMediaError(err: unknown): CameraErrorDetails {
  if (typeof window !== 'undefined' && !window.isSecureContext) {
    return {
      code: 'INSECURE_CONTEXT',
      title: 'Se requiere conexión segura (HTTPS)',
      message:
        'El navegador bloqueó el acceso a la cámara porque la página no se cargó mediante HTTPS.',
      browserHint:
        'Abre la aplicación usando una dirección https:// válida o mediante http://localhost durante desarrollo.',
    };
  }

  const domErr = err as { name?: string; message?: string };
  const errName = domErr?.name ?? '';

  if (
    errName === 'NotAllowedError' ||
    errName === 'PermissionDeniedError' ||
    errName === 'SecurityError'
  ) {
    return {
      code: 'PERMISSION_DENIED',
      title: 'Permiso de Cámara Denegado',
      message:
        'El navegador rechazó el acceso a la cámara. Debes habilitar el permiso para este sitio.',
      browserHint:
        'En iOS Safari: toca el icono "aA" o el icono de ajustes en la barra de direcciones → Configuración del sitio web → Cámara → Permitir. En Chrome Android: toca el icono de candado/ajustes junto a la URL → Permisos → Cámara → Permitir.',
    };
  }

  if (errName === 'NotFoundError' || errName === 'DevicesNotFoundError') {
    return {
      code: 'NO_CAMERA_FOUND',
      title: 'No se detectó ninguna cámara física',
      message:
        'El dispositivo no reportó ningún sensor de video disponible o está deshabilitado.',
      browserHint:
        'Verifica que tu dispositivo móvil o computador tenga una cámara conectada y que no esté bloqueada por políticas del sistema.',
    };
  }

  if (errName === 'NotReadableError' || errName === 'TrackStartError') {
    return {
      code: 'HARDWARE_LOCKED',
      title: 'Cámara en uso por otra aplicación',
      message:
        'El hardware de la cámara está bloqueado temporalmente por otra pestaña o aplicación del sistema.',
      browserHint:
        'Cierra otras aplicaciones o pestañas que estén usando la cámara y presiona "Reintentar Iniciar Cámara AR".',
    };
  }

  return {
    code: 'UNKNOWN',
    title: 'No se pudo inicializar el flujo de cámara',
    message: domErr?.message || 'Error inesperado al abrir MediaDevices.getUserMedia.',
    browserHint:
      'Verifica los permisos de cámara del navegador y asegúrate de usar Safari 15.4+, Chrome Android, Firefox o Samsung Internet.',
  };
}

export const WebCameraStream = React.forwardRef<WebCameraStreamHandle, WebCameraStreamProps>(
  ({ facingMode, onStreamReady, onStreamError }, ref) => {
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const activeStreamRef = useRef<MediaStream | null>(null);
    const currentFacingRef = useRef<WebCameraFacingMode>(facingMode);
    currentFacingRef.current = facingMode;

    /**
     * Detiene todos los tracks de video activos y limpia srcObject.
     * Requisito crítico en Android para liberar el hardware ISP antes de abrir la otra lente.
     */
    const stopStream = useCallback(() => {
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach((track) => {
          try {
            track.stop();
          } catch {
            // Ignorar error al detener track ya cerrado
          }
        });
        activeStreamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }
    }, []);

    /**
     * Configura atributos nativos en el nodo DOM <video> requeridos por Safari iOS
     * y Samsung Internet para reproducción inline sin abrir reproductor a pantalla completa.
     */
    const enforceInlineVideoAttributes = (videoEl: HTMLVideoElement) => {
      videoEl.playsInline = true;
      videoEl.muted = true;
      videoEl.autoplay = true;
      videoEl.setAttribute('playsinline', 'true');
      videoEl.setAttribute('webkit-playsinline', 'true');
      videoEl.setAttribute('muted', 'true');
      videoEl.setAttribute('autoplay', 'true');
    };

    /**
     * Abre la cámara con restricciones adaptables y fallback progresivo en 3 niveles
     * para evitar OverconstrainedError en dispositivos Android antiguos, iPads o laptops.
     */
    const startStream = useCallback(
      async (requestedFacing?: WebCameraFacingMode): Promise<HTMLVideoElement | null> => {
        const targetFacing = requestedFacing ?? currentFacingRef.current;

        if (
          typeof navigator === 'undefined' ||
          !navigator.mediaDevices ||
          typeof navigator.mediaDevices.getUserMedia !== 'function'
        ) {
          onStreamError(
            classifyGetUserMediaError(
              new Error('API navigator.mediaDevices.getUserMedia no disponible en este contexto.')
            )
          );
          return null;
        }

        // 1. Liberar cualquier stream previo antes de solicitar el nuevo hardware
        stopStream();

        // Pequeña pausa de 40ms para que el driver de cámara de Android libere el descriptor
        await new Promise((r) => setTimeout(r, 40));

        const progressiveConstraints: MediaStreamConstraints[] = [
          // Nivel 1: 720p HD con facingMode ideal y 30 FPS ideales
          {
            video: {
              facingMode: { ideal: targetFacing },
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30, max: 60 },
            },
            audio: false,
          },
          // Nivel 2: Fallback VGA 640x480 con facingMode flexible (evita OverconstrainedError)
          {
            video: {
              facingMode: targetFacing,
              width: { ideal: 640 },
              height: { ideal: 480 },
            },
            audio: false,
          },
          // Nivel 3: Fallback universal sin restricciones de resolución ni orientación
          {
            video: true,
            audio: false,
          },
        ];

        let acquiredStream: MediaStream | null = null;
        let lastError: unknown = null;

        for (const constraints of progressiveConstraints) {
          try {
            acquiredStream = await navigator.mediaDevices.getUserMedia(constraints);
            if (acquiredStream) break;
          } catch (err) {
            lastError = err;
            const errName = (err as { name?: string })?.name ?? '';
            // Si el usuario denegó el permiso explícitamente, no reintentar con otros constraints
            if (
              errName === 'NotAllowedError' ||
              errName === 'PermissionDeniedError' ||
              errName === 'SecurityError'
            ) {
              break;
            }
          }
        }

        if (!acquiredStream) {
          onStreamError(classifyGetUserMediaError(lastError));
          return null;
        }

        activeStreamRef.current = acquiredStream;
        const videoEl = videoRef.current;
        if (!videoEl) return null;

        enforceInlineVideoAttributes(videoEl);
        videoEl.srcObject = acquiredStream;

        // Esperar a que los metadatos del stream (videoWidth, videoHeight) estén listos y reproducir
        await new Promise<void>((resolve, reject) => {
          const onLoadedMeta = () => {
            videoEl.removeEventListener('loadedmetadata', onLoadedMeta);
            videoEl
              .play()
              .then(() => resolve())
              .catch((playErr) => reject(playErr));
          };

          if (videoEl.readyState >= 1 && videoEl.videoWidth > 0) {
            videoEl
              .play()
              .then(() => resolve())
              .catch((playErr) => reject(playErr));
          } else {
            videoEl.addEventListener('loadedmetadata', onLoadedMeta, { once: true });
          }
        }).catch((playErr) => {
          onStreamError(classifyGetUserMediaError(playErr));
        });

        currentFacingRef.current = targetFacing;
        onStreamReady(videoEl, targetFacing);
        return videoEl;
      },
      [onStreamError, onStreamReady, stopStream]
    );

    const switchCamera = useCallback(async (): Promise<WebCameraFacingMode> => {
      const nextFacing: WebCameraFacingMode =
        currentFacingRef.current === 'environment' ? 'user' : 'environment';
      await startStream(nextFacing);
      return nextFacing;
    }, [startStream]);

    useImperativeHandle(
      ref,
      () => ({
        startStream,
        stopStream,
        switchCamera,
        getVideoElement: () => videoRef.current,
      }),
      [startStream, stopStream, switchCamera]
    );

    useEffect(() => {
      if (videoRef.current) {
        enforceInlineVideoAttributes(videoRef.current);
      }
      return () => {
        stopStream();
      };
    }, [stopStream]);

    return (
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`fixed inset-0 w-screen h-dvh object-cover pointer-events-none select-none bg-[#06090F] ${
          facingMode === 'user' ? '-scale-x-100' : ''
        }`}
      />
    );
  }
);

WebCameraStream.displayName = 'WebCameraStream';
