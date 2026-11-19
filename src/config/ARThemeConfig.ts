import { SemanticCategory } from '../utils/visionPostProcessor';

export type ARThemeId = 'tactical' | 'native' | 'industrial';

export interface ARCategoryStyle {
  stroke: string;
  fill: string;
  glow: string;
  badgeBg: string;
  badgeText: string;
}

export interface ARThemeSpec {
  id: ARThemeId;
  name: string;
  shortLabel: string;
  description: string;
  useCornerBrackets: boolean;
  cornerBracketLength: number;
  cornerRadius: number;
  boxStrokeWidth: number;
  showHorizonReticle: boolean;
  showLeaderLinePins: boolean;
  showDistanceEstimate: boolean;
  useMonospaceFont: boolean;
  skeletonBoneColor: string;
  skeletonJointColor: string;
  hudGlassBg: string;
  hudBorderColor: string;
  pillBackground: string;
  pillTextColor: string;
  categories: Record<SemanticCategory, ARCategoryStyle>;
}

/**
 * 3 Modos Visuales AR intercambiables en vivo:
 * 1. tactical: Táctico / Futurista Cian (Por defecto)
 * 2. native: Minimalista Cámara Nativa (iOS/Android Camera AI)
 * 3. industrial: Alto Contraste Industrial (Luz solar directa / Ingeniería)
 */
export const AR_THEMES: Record<ARThemeId, ARThemeSpec> = {
  tactical: {
    id: 'tactical',
    name: 'Táctico / Futurista Cian',
    shortLabel: 'Táctico',
    description:
      'Esquinas de enfoque óptico (corner brackets) con brillo neón (#38BDF8), retícula central de horizonte, etiquetas con línea guía, distancia estimada (~3.2m) y esqueleto biomecánico verde neón (#22C55E).',
    useCornerBrackets: true,
    cornerBracketLength: 16,
    cornerRadius: 4,
    boxStrokeWidth: 2.0,
    showHorizonReticle: true,
    showLeaderLinePins: true,
    showDistanceEstimate: true,
    useMonospaceFont: true,
    skeletonBoneColor: '#22C55E',
    skeletonJointColor: '#86EFAC',
    hudGlassBg: 'rgba(11, 15, 23, 0.55)',
    hudBorderColor: 'rgba(56, 189, 248, 0.35)',
    pillBackground: 'rgba(9, 13, 22, 0.84)',
    pillTextColor: '#F8FAFC',
    categories: {
      person: {
        stroke: '#38BDF8',
        fill: 'rgba(56, 189, 248, 0.11)',
        glow: 'rgba(56, 189, 248, 0.48)',
        badgeBg: 'rgba(9, 13, 22, 0.88)',
        badgeText: '#38BDF8',
      },
      vehicle: {
        stroke: '#F97316',
        fill: 'rgba(249, 115, 22, 0.12)',
        glow: 'rgba(249, 115, 22, 0.48)',
        badgeBg: 'rgba(9, 13, 22, 0.88)',
        badgeText: '#FB923C',
      },
      bicycle: {
        stroke: '#F97316',
        fill: 'rgba(249, 115, 22, 0.12)',
        glow: 'rgba(249, 115, 22, 0.48)',
        badgeBg: 'rgba(9, 13, 22, 0.88)',
        badgeText: '#FB923C',
      },
      animal: {
        stroke: '#10B981',
        fill: 'rgba(16, 185, 129, 0.13)',
        glow: 'rgba(16, 185, 129, 0.48)',
        badgeBg: 'rgba(9, 13, 22, 0.88)',
        badgeText: '#34D399',
      },
      object: {
        stroke: '#A855F7',
        fill: 'rgba(168, 85, 247, 0.13)',
        glow: 'rgba(168, 85, 247, 0.48)',
        badgeBg: 'rgba(9, 13, 22, 0.88)',
        badgeText: '#C084FC',
      },
    },
  },
  native: {
    id: 'native',
    name: 'Minimalista Cámara Nativa',
    shortLabel: 'Nativo',
    description:
      'Estética limpia tipo iOS/Android Camera AI con esquinas delgadas redondeadas en blanco translúcido (rgba(255, 255, 255, 0.85)), píldoras frosted glass ultracompactas sin líneas guía y puntos articulares discretos.',
    useCornerBrackets: false,
    cornerBracketLength: 10,
    cornerRadius: 12,
    boxStrokeWidth: 1.4,
    showHorizonReticle: false,
    showLeaderLinePins: false,
    showDistanceEstimate: false,
    useMonospaceFont: false,
    skeletonBoneColor: 'rgba(255, 255, 255, 0.72)',
    skeletonJointColor: '#FFFFFF',
    hudGlassBg: 'rgba(11, 15, 23, 0.55)',
    hudBorderColor: 'rgba(255, 255, 255, 0.22)',
    pillBackground: 'rgba(20, 24, 33, 0.68)',
    pillTextColor: '#FFFFFF',
    categories: {
      person: {
        stroke: 'rgba(255, 255, 255, 0.85)',
        fill: 'rgba(255, 255, 255, 0.05)',
        glow: 'rgba(255, 255, 255, 0.15)',
        badgeBg: 'rgba(255, 255, 255, 0.22)',
        badgeText: '#FFFFFF',
      },
      vehicle: {
        stroke: 'rgba(255, 255, 255, 0.85)',
        fill: 'rgba(255, 255, 255, 0.05)',
        glow: 'rgba(255, 255, 255, 0.15)',
        badgeBg: 'rgba(255, 255, 255, 0.22)',
        badgeText: '#FFFFFF',
      },
      bicycle: {
        stroke: 'rgba(255, 255, 255, 0.85)',
        fill: 'rgba(255, 255, 255, 0.05)',
        glow: 'rgba(255, 255, 255, 0.15)',
        badgeBg: 'rgba(255, 255, 255, 0.22)',
        badgeText: '#FFFFFF',
      },
      animal: {
        stroke: 'rgba(255, 255, 255, 0.85)',
        fill: 'rgba(255, 255, 255, 0.05)',
        glow: 'rgba(255, 255, 255, 0.15)',
        badgeBg: 'rgba(255, 255, 255, 0.22)',
        badgeText: '#FFFFFF',
      },
      object: {
        stroke: 'rgba(255, 255, 255, 0.85)',
        fill: 'rgba(255, 255, 255, 0.05)',
        glow: 'rgba(255, 255, 255, 0.15)',
        badgeBg: 'rgba(255, 255, 255, 0.22)',
        badgeText: '#FFFFFF',
      },
    },
  },
  industrial: {
    id: 'industrial',
    name: 'Alto Contraste Industrial',
    shortLabel: 'Industrial',
    description:
      'Diseñado para exteriores con luz solar directa o entornos de ingeniería; bordes gruesos en amarillo/ámbar de seguridad (#FACC15) y naranja (#F97316), fondos de etiqueta sólidos (#090D16) y tipografía monoespaciada.',
    useCornerBrackets: false,
    cornerBracketLength: 14,
    cornerRadius: 2,
    boxStrokeWidth: 3.0,
    showHorizonReticle: true,
    showLeaderLinePins: false,
    showDistanceEstimate: true,
    useMonospaceFont: true,
    skeletonBoneColor: '#FACC15',
    skeletonJointColor: '#F97316',
    hudGlassBg: 'rgba(11, 15, 23, 0.55)',
    hudBorderColor: 'rgba(250, 204, 21, 0.55)',
    pillBackground: '#090D16',
    pillTextColor: '#FACC15',
    categories: {
      person: {
        stroke: '#FACC15',
        fill: 'rgba(250, 204, 21, 0.14)',
        glow: 'rgba(250, 204, 21, 0.35)',
        badgeBg: '#090D16',
        badgeText: '#FACC15',
      },
      vehicle: {
        stroke: '#F97316',
        fill: 'rgba(249, 115, 22, 0.16)',
        glow: 'rgba(249, 115, 22, 0.35)',
        badgeBg: '#090D16',
        badgeText: '#F97316',
      },
      bicycle: {
        stroke: '#FACC15',
        fill: 'rgba(250, 204, 21, 0.14)',
        glow: 'rgba(250, 204, 21, 0.35)',
        badgeBg: '#090D16',
        badgeText: '#FACC15',
      },
      animal: {
        stroke: '#F97316',
        fill: 'rgba(249, 115, 22, 0.16)',
        glow: 'rgba(249, 115, 22, 0.35)',
        badgeBg: '#090D16',
        badgeText: '#F97316',
      },
      object: {
        stroke: '#FACC15',
        fill: 'rgba(250, 204, 21, 0.12)',
        glow: 'rgba(250, 204, 21, 0.30)',
        badgeBg: '#090D16',
        badgeText: '#FACC15',
      },
    },
  },
};

export const AR_THEME_ORDER: ARThemeId[] = ['tactical', 'native', 'industrial'];

/**
 * Estima la distancia aproximada en metros (~1.2m a ~25.0m) mediante relación
 * de altura proyectada de la caja delimitadora respecto a la altura del visor.
 */
export function estimateObjectDistanceMeters(
  category: SemanticCategory,
  boxHeightPx: number,
  screenHeightPx: number
): number {
  'worklet';
  const ratio = Math.max(0.03, Math.min(0.95, boxHeightPx / Math.max(1, screenHeightPx)));
  const referenceHeightMeters =
    category === 'person'
      ? 1.72
      : category === 'vehicle'
      ? 1.58
      : category === 'bicycle'
      ? 1.15
      : category === 'animal'
      ? 0.65
      : 0.55;
  const focalFactor = 0.85;
  const dist = (referenceHeightMeters * focalFactor) / ratio;
  return Math.max(0.8, Math.min(28.0, Math.round(dist * 10) / 10));
}
