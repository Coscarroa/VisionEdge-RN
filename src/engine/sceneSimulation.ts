import { CameraSourceMode, RawProposal, TargetClass } from './cvTypes';
import { classToId, computeLetterboxTransform } from './postprocess';

interface SimAgent {
  id: string;
  className: TargetClass;
  baseConf: number;
  xNorm: number; // 0..1 of frame width
  yNorm: number; // 0..1 of frame height
  wNorm: number;
  hNorm: number;
  vx: number; // normalized units per second
  vy: number;
  phase: number;
  isStatic?: boolean;
}

const CROSSWALK_AGENTS: SimAgent[] = [
  // Static/slow background objects grounded in the crosswalk scene
  { id: 'taxi_left', className: 'car', baseConf: 0.94, xNorm: 0.33, yNorm: 0.47, wNorm: 0.08, hNorm: 0.065, vx: 0.015, vy: 0.002, phase: 0.4 },
  { id: 'suv_mid', className: 'truck', baseConf: 0.95, xNorm: 0.67, yNorm: 0.51, wNorm: 0.11, hNorm: 0.085, vx: -0.055, vy: 0.008, phase: 1.1 },
  { id: 'tram_bus', className: 'bus', baseConf: 0.91, xNorm: 0.71, yNorm: 0.41, wNorm: 0.095, hNorm: 0.095, vx: -0.012, vy: 0.005, phase: 2.0 },
  // Moving crosswalk pedestrians with full 17-keypoint skeletons + pet dog (COCO class 16)
  { id: 'ped_cross_1', className: 'person', baseConf: 0.93, xNorm: 0.24, yNorm: 0.66, wNorm: 0.075, hNorm: 0.22, vx: 0.065, vy: -0.004, phase: 0.0 },
  { id: 'dog_cross_1', className: 'dog', baseConf: 0.89, xNorm: 0.30, yNorm: 0.74, wNorm: 0.055, hNorm: 0.085, vx: 0.065, vy: -0.004, phase: 0.3 },
  { id: 'ped_cross_2', className: 'person', baseConf: 0.89, xNorm: 0.52, yNorm: 0.64, wNorm: 0.068, hNorm: 0.20, vx: -0.052, vy: 0.003, phase: 1.8 },
  { id: 'ped_sidewalk', className: 'person', baseConf: 0.87, xNorm: 0.85, yNorm: 0.63, wNorm: 0.065, hNorm: 0.19, vx: -0.018, vy: 0.0, phase: 3.1 },
  { id: 'bike_lane_1', className: 'bicycle', baseConf: 0.88, xNorm: 0.15, yNorm: 0.69, wNorm: 0.095, hNorm: 0.13, vx: 0.085, vy: -0.005, phase: 0.7 },
];

const BOULEVARD_AGENTS: SimAgent[] = [
  { id: 'cyclist_center', className: 'bicycle', baseConf: 0.94, xNorm: 0.42, yNorm: 0.66, wNorm: 0.11, hNorm: 0.18, vx: 0.006, vy: -0.028, phase: 0.2 },
  { id: 'cyclist_rider', className: 'person', baseConf: 0.95, xNorm: 0.42, yNorm: 0.60, wNorm: 0.085, hNorm: 0.22, vx: 0.006, vy: -0.028, phase: 0.2 },
  { id: 'moto_ahead', className: 'motorcycle', baseConf: 0.88, xNorm: 0.59, yNorm: 0.59, wNorm: 0.07, hNorm: 0.125, vx: 0.003, vy: -0.018, phase: 1.4 },
  { id: 'rider_ahead', className: 'person', baseConf: 0.88, xNorm: 0.59, yNorm: 0.56, wNorm: 0.052, hNorm: 0.13, vx: 0.003, vy: -0.018, phase: 1.4 },
  { id: 'ped_walk_1', className: 'person', baseConf: 0.92, xNorm: 0.79, yNorm: 0.57, wNorm: 0.055, hNorm: 0.16, vx: -0.022, vy: 0.01, phase: 2.3 },
  { id: 'dog_plaza_1', className: 'dog', baseConf: 0.87, xNorm: 0.84, yNorm: 0.65, wNorm: 0.052, hNorm: 0.08, vx: -0.020, vy: 0.012, phase: 0.9 },
  { id: 'bus_yellow', className: 'bus', baseConf: 0.93, xNorm: 0.17, yNorm: 0.51, wNorm: 0.12, hNorm: 0.085, vx: 0.018, vy: 0.002, phase: 0.0 },
  { id: 'car_silver', className: 'car', baseConf: 0.91, xNorm: 0.18, yNorm: 0.55, wNorm: 0.10, hNorm: 0.075, vx: 0.042, vy: 0.003, phase: 1.7 },
];

const CROWD_EXTRA_AGENTS: SimAgent[] = [
  { id: 'crowd_ped_1', className: 'person', baseConf: 0.91, xNorm: 0.37, yNorm: 0.69, wNorm: 0.065, hNorm: 0.19, vx: 0.048, vy: 0.002, phase: 0.5 },
  { id: 'crowd_ped_2', className: 'person', baseConf: 0.84, xNorm: 0.64, yNorm: 0.68, wNorm: 0.065, hNorm: 0.19, vx: -0.044, vy: -0.002, phase: 2.7 },
  { id: 'crowd_bike_2', className: 'bicycle', baseConf: 0.85, xNorm: 0.74, yNorm: 0.72, wNorm: 0.09, hNorm: 0.12, vx: -0.075, vy: 0.0, phase: 1.9 },
  { id: 'crowd_car_2', className: 'car', baseConf: 0.89, xNorm: 0.45, yNorm: 0.50, wNorm: 0.095, hNorm: 0.075, vx: 0.06, vy: 0.001, phase: 3.3 },
];

function wrapNorm(val: number, min = 0.08, max = 0.92): number {
  const span = max - min;
  const shifted = (val - min) % span;
  return min + (shifted < 0 ? shifted + span : shifted);
}

/**
 * Generates realistic COCO 17-point walking skeleton keypoints in normalized [0..1] box space,
 * then maps them into 640x640 tensor coordinates.
 */
function buildTensorKeypoints(
  cxTensor: number,
  cyTensor: number,
  wTensor: number,
  hTensor: number,
  gaitPhase: number,
  directionSign: number
): Array<{ x: number; y: number; score: number }> {
  const stride = Math.sin(gaitPhase) * 0.24;
  const armSwing = Math.sin(gaitPhase + Math.PI) * 0.2;
  const bob = Math.abs(Math.cos(gaitPhase)) * 0.015;
  const dir = directionSign >= 0 ? 1 : -1;

  const left = cxTensor - wTensor * 0.5;
  const top = cyTensor - hTensor * (0.5 + bob);

  const pt = (rx: number, ry: number, score = 0.92) => ({
    x: left + rx * wTensor,
    y: top + ry * hTensor,
    score,
  });

  // 17 COCO keypoints
  return [
    pt(0.5 + dir * 0.04, 0.09, 0.96),               // 0 nose
    pt(0.45 + dir * 0.03, 0.07, 0.93),              // 1 left_eye
    pt(0.55 + dir * 0.03, 0.07, 0.93),              // 2 right_eye
    pt(0.39, 0.085, 0.86),                          // 3 left_ear
    pt(0.61, 0.085, 0.86),                          // 4 right_ear
    pt(0.31, 0.24, 0.95),                           // 5 left_shoulder
    pt(0.69, 0.24, 0.95),                           // 6 right_shoulder
    pt(0.24 + armSwing * 0.45, 0.39, 0.91),         // 7 left_elbow
    pt(0.76 - armSwing * 0.45, 0.39, 0.91),         // 8 right_elbow
    pt(0.22 + armSwing * 0.85, 0.52, 0.88),         // 9 left_wrist
    pt(0.78 - armSwing * 0.85, 0.52, 0.88),         // 10 right_wrist
    pt(0.37, 0.53, 0.94),                           // 11 left_hip
    pt(0.63, 0.53, 0.94),                           // 12 right_hip
    pt(0.36 + stride * 0.5, 0.74, 0.92),            // 13 left_knee
    pt(0.64 - stride * 0.5, 0.74, 0.92),            // 14 right_knee
    pt(0.34 + stride, 0.95, 0.90),                  // 15 left_ankle
    pt(0.66 - stride, 0.95, 0.90),                  // 16 right_ankle
  ];
}

export interface RenderedSimEntity {
  id: string;
  className: TargetClass;
  xNorm: number;
  yNorm: number;
  wNorm: number;
  hNorm: number;
  gaitPhase: number;
  dir: number;
}

/**
 * Generates raw YOLOv8n-style tensor proposals (640x640 letterboxed space),
 * including realistic overlapping duplicate grid anchor candidates so that
 * `applyNMS` actively filters duplicates in real time.
 */
export function generateSimulatedProposals(
  mode: CameraSourceMode,
  elapsedSec: number,
  crowdBoost: boolean,
  srcWidth: number,
  srcHeight: number
): { proposals: RawProposal[]; entities: RenderedSimEntity[] } {
  const baseAgents = mode === 'bike-boulevard' ? BOULEVARD_AGENTS : CROSSWALK_AGENTS;
  const agents = crowdBoost ? [...baseAgents, ...CROWD_EXTRA_AGENTS] : baseAgents;

  const letterbox = computeLetterboxTransform(srcWidth, srcHeight, 640);
  const proposals: RawProposal[] = [];
  const entities: RenderedSimEntity[] = [];

  for (let i = 0; i < agents.length; i++) {
    const a = agents[i];
    const xNorm = wrapNorm(a.xNorm + a.vx * elapsedSec);
    const yNorm = mode === 'bike-boulevard'
      ? wrapNorm(a.yNorm + a.vy * elapsedSec, 0.45, 0.82)
      : Math.max(0.32, Math.min(0.84, a.yNorm + Math.sin(elapsedSec * 0.8 + a.phase) * 0.012));

    const gaitPhase = elapsedSec * 5.2 + a.phase;
    const dir = a.vx >= 0 ? 1 : -1;

    entities.push({
      id: a.id,
      className: a.className,
      xNorm,
      yNorm,
      wNorm: a.wNorm,
      hNorm: a.hNorm,
      gaitPhase,
      dir,
    });

    const origCx = xNorm * srcWidth;
    const origCy = yNorm * srcHeight;
    const origW = a.wNorm * srcWidth;
    const origH = a.hNorm * srcHeight;

    const tensorCx = origCx * letterbox.scale + letterbox.padX;
    const tensorCy = origCy * letterbox.scale + letterbox.padY;
    const tensorW = origW * letterbox.scale;
    const tensorH = origH * letterbox.scale;

    const jitter = Math.sin(elapsedSec * 3.7 + i * 1.9) * 0.025;
    const primaryConf = Math.max(0.35, Math.min(0.98, a.baseConf + jitter));

    const tensorKeypoints =
      a.className === 'person'
        ? buildTensorKeypoints(tensorCx, tensorCy, tensorW, tensorH, gaitPhase, dir)
        : undefined;

    // Primary anchor prediction
    proposals.push({
      id: `${a.id}_p0`,
      classId: classToId(a.className),
      className: a.className,
      confidence: primaryConf,
      tensorBox: {
        cx: tensorCx,
        cy: tensorCy,
        w: tensorW,
        h: tensorH,
      },
      tensorKeypoints,
    });

    // Overlapping secondary anchor proposal (to be suppressed by real NMS)
    proposals.push({
      id: `${a.id}_dup1`,
      classId: classToId(a.className),
      className: a.className,
      confidence: Math.max(0.28, primaryConf - 0.14),
      tensorBox: {
        cx: tensorCx + 4.5,
        cy: tensorCy - 3.0,
        w: tensorW * 1.05,
        h: tensorH * 0.97,
      },
      tensorKeypoints,
    });
  }

  return { proposals, entities };
}

/**
 * Extracts live motion-based region proposals from a real HTMLVideoElement
 * (Live Webcam mode) by computing frame-to-frame luminance delta blocks and
 * mapping active clusters into 640x640 tensor proposals for NMS & ByteTrack.
 */
export class WebcamMotionExtractor {
  private prevFrame: Uint8ClampedArray | null = null;
  private offscreen: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null;
  private readonly gridW = 64;
  private readonly gridH = 48;

  constructor() {
    this.offscreen = document.createElement('canvas');
    this.offscreen.width = this.gridW;
    this.offscreen.height = this.gridH;
    this.ctx = this.offscreen.getContext('2d', { willReadFrequently: true });
  }

  public extractProposals(
    video: HTMLVideoElement,
    elapsedSec: number
  ): RawProposal[] {
    if (!this.ctx || video.readyState < 2 || video.videoWidth === 0) {
      return [];
    }

    const srcW = video.videoWidth;
    const srcH = video.videoHeight;
    const letterbox = computeLetterboxTransform(srcW, srcH, 640);

    this.ctx.drawImage(video, 0, 0, this.gridW, this.gridH);
    const imgData = this.ctx.getImageData(0, 0, this.gridW, this.gridH).data;

    // Fallback centered subject box if static webcam scene has low motion
    let minX = this.gridW;
    let maxX = 0;
    let minY = this.gridH;
    let maxY = 0;
    let activePixels = 0;

    if (this.prevFrame) {
      for (let y = 0; y < this.gridH; y++) {
        for (let x = 0; x < this.gridW; x++) {
          const idx = (y * this.gridW + x) * 4;
          const lumNow = imgData[idx] * 0.299 + imgData[idx + 1] * 0.587 + imgData[idx + 2] * 0.114;
          const lumPrev =
            this.prevFrame[idx] * 0.299 +
            this.prevFrame[idx + 1] * 0.587 +
            this.prevFrame[idx + 2] * 0.114;
          if (Math.abs(lumNow - lumPrev) > 16) {
            activePixels++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }
    }

    this.prevFrame = new Uint8ClampedArray(imgData);

    const cxNorm = activePixels > 18 ? (minX + maxX) / 2 / this.gridW : 0.5;
    const cyNorm = activePixels > 18 ? (minY + maxY) / 2 / this.gridH : 0.54;
    const wNorm = activePixels > 18 ? Math.max(0.24, Math.min(0.65, (maxX - minX) / this.gridW)) : 0.34;
    const hNorm = activePixels > 18 ? Math.max(0.38, Math.min(0.84, (maxY - minY) / this.gridH)) : 0.62;

    const tensorCx = cxNorm * srcW * letterbox.scale + letterbox.padX;
    const tensorCy = cyNorm * srcH * letterbox.scale + letterbox.padY;
    const tensorW = wNorm * srcW * letterbox.scale;
    const tensorH = hNorm * srcH * letterbox.scale;

    const kps = buildTensorKeypoints(tensorCx, tensorCy, tensorW, tensorH, elapsedSec * 1.5, 1);
    const conf = activePixels > 18 ? 0.91 : 0.84;

    return [
      {
        id: 'webcam_person_0',
        classId: 0,
        className: 'person',
        confidence: conf,
        tensorBox: { cx: tensorCx, cy: tensorCy, w: tensorW, h: tensorH },
        tensorKeypoints: kps,
      },
      {
        id: 'webcam_person_dup',
        classId: 0,
        className: 'person',
        confidence: conf - 0.15,
        tensorBox: { cx: tensorCx + 8, cy: tensorCy - 5, w: tensorW * 1.04, h: tensorH * 0.96 },
        tensorKeypoints: kps,
      },
    ];
  }
}
