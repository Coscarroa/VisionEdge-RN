import {
  COCO_KEYPOINT_NAMES,
  DensityLevel,
  DetectionResult,
  Keypoint,
  RawProposal,
  ScreenBox,
  TargetClass,
  TrackedObject,
} from './cvTypes';

export interface LetterboxTransform {
  modelSize: number;
  scale: number;
  padX: number;
  padY: number;
}

/**
 * Computes letterbox scale and padding when fitting a camera frame (srcW x srcH)
 * into a square YOLO tensor (e.g. 640 x 640) while preserving aspect ratio.
 */
export function computeLetterboxTransform(
  srcWidth: number,
  srcHeight: number,
  modelSize = 640
): LetterboxTransform {
  const safeW = Math.max(1, srcWidth);
  const safeH = Math.max(1, srcHeight);
  const scale = Math.min(modelSize / safeW, modelSize / safeH);
  const resizedW = safeW * scale;
  const resizedH = safeH * scale;
  const padX = (modelSize - resizedW) / 2;
  const padY = (modelSize - resizedH) / 2;
  return { modelSize, scale, padX, padY };
}

/**
 * Converts a center-format tensor bounding box [cx, cy, w, h] in 640x640 model space
 * back into screen/canvas pixel coordinates [x, y, width, height].
 */
export function tensorBoxToScreen(
  cx: number,
  cy: number,
  w: number,
  h: number,
  letterbox: LetterboxTransform,
  srcWidth: number,
  srcHeight: number,
  canvasWidth: number,
  canvasHeight: number,
  mirrorX = false
): ScreenBox {
  // 1. Remove letterbox padding & undo model resize scale -> original frame coords
  const origCx = (cx - letterbox.padX) / letterbox.scale;
  const origCy = (cy - letterbox.padY) / letterbox.scale;
  const origW = w / letterbox.scale;
  const origH = h / letterbox.scale;

  // 2. Map original frame coordinates to display canvas (aspect-fill / cover or fit)
  const scaleX = canvasWidth / Math.max(1, srcWidth);
  const scaleY = canvasHeight / Math.max(1, srcHeight);

  const screenW = origW * scaleX;
  const screenH = origH * scaleY;
  let screenX = (origCx - origW / 2) * scaleX;
  const screenY = (origCy - origH / 2) * scaleY;

  if (mirrorX) {
    screenX = canvasWidth - screenX - screenW;
  }

  return {
    x: Math.max(0, Math.min(canvasWidth - 4, screenX)),
    y: Math.max(0, Math.min(canvasHeight - 4, screenY)),
    width: Math.max(4, Math.min(canvasWidth, screenW)),
    height: Math.max(4, Math.min(canvasHeight, screenH)),
  };
}

export function tensorKeypointToScreen(
  kx: number,
  ky: number,
  score: number,
  index: number,
  letterbox: LetterboxTransform,
  srcWidth: number,
  srcHeight: number,
  canvasWidth: number,
  canvasHeight: number,
  mirrorX = false
): Keypoint {
  const origX = (kx - letterbox.padX) / letterbox.scale;
  const origY = (ky - letterbox.padY) / letterbox.scale;
  const scaleX = canvasWidth / Math.max(1, srcWidth);
  const scaleY = canvasHeight / Math.max(1, srcHeight);

  let sx = origX * scaleX;
  const sy = origY * scaleY;
  if (mirrorX) {
    sx = canvasWidth - sx;
  }

  return {
    index,
    name: COCO_KEYPOINT_NAMES[index] ?? `kp_${index}`,
    x: sx,
    y: sy,
    score,
  };
}

export function computeIoU(a: ScreenBox, b: ScreenBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;
  if (interArea <= 0) return 0;

  const unionArea = a.width * a.height + b.width * b.height - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

/**
 * Class-aware Non-Maximum Suppression (NMS) in screen coordinate space.
 * Filters overlapping duplicate detections from raw YOLOv8n grid proposals.
 */
export function applyNMS(
  detections: DetectionResult[],
  iouThreshold = 0.45,
  classAware = true
): DetectionResult[] {
  if (detections.length === 0) return [];

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const selected: DetectionResult[] = [];
  const suppressed = new Uint8Array(sorted.length);

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i] === 1) continue;
    const candidate = sorted[i];
    selected.push(candidate);

    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j] === 1) continue;
      if (classAware && sorted[j].className !== candidate.className) continue;

      const iou = computeIoU(candidate.screenBox, sorted[j].screenBox);
      if (iou > iouThreshold) {
        suppressed[j] = 1;
      }
    }
  }

  return selected;
}

export function decodeProposalsToScreen(
  proposals: RawProposal[],
  confThreshold: number,
  iouThreshold: number,
  srcWidth: number,
  srcHeight: number,
  canvasWidth: number,
  canvasHeight: number,
  enablePose: boolean,
  mirrorX = false
): {
  rawFilteredCount: number;
  detections: DetectionResult[];
  letterbox: LetterboxTransform;
} {
  const letterbox = computeLetterboxTransform(srcWidth, srcHeight, 640);
  const filtered: DetectionResult[] = [];

  for (const prop of proposals) {
    if (prop.confidence < confThreshold) continue;

    const screenBox = tensorBoxToScreen(
      prop.tensorBox.cx,
      prop.tensorBox.cy,
      prop.tensorBox.w,
      prop.tensorBox.h,
      letterbox,
      srcWidth,
      srcHeight,
      canvasWidth,
      canvasHeight,
      mirrorX
    );

    const keypoints: Keypoint[] =
      enablePose && prop.className === 'person' && prop.tensorKeypoints
        ? prop.tensorKeypoints.map((kp, idx) =>
            tensorKeypointToScreen(
              kp.x,
              kp.y,
              kp.score,
              idx,
              letterbox,
              srcWidth,
              srcHeight,
              canvasWidth,
              canvasHeight,
              mirrorX
            )
          )
        : [];

    filtered.push({
      classId: prop.classId,
      className: prop.className,
      confidence: prop.confidence,
      screenBox,
      keypoints,
    });
  }

  const nmsResult = applyNMS(filtered, iouThreshold, true);
  return {
    rawFilteredCount: filtered.length,
    detections: nmsResult,
    letterbox,
  };
}

/**
 * Lightweight SORT / ByteTrack-inspired multi-object tracker.
 * Uses two-stage IoU matching (high confidence first, then low confidence recovery)
 * and exponential moving average (EMA) velocity smoothing.
 */
export class ByteTrackLite {
  private nextId = 1;
  private tracks: TrackedObject[] = [];
  private readonly maxLostFrames: number;
  private readonly highThresh: number;
  private readonly iouMatchThresh: number;

  constructor(maxLostFrames = 18, highThresh = 0.55, iouMatchThresh = 0.25) {
    this.maxLostFrames = maxLostFrames;
    this.highThresh = highThresh;
    this.iouMatchThresh = iouMatchThresh;
  }

  public reset(): void {
    this.nextId = 1;
    this.tracks = [];
  }

  public update(detections: DetectionResult[]): TrackedObject[] {
    const highDets = detections.filter((d) => d.confidence >= this.highThresh);
    const lowDets = detections.filter((d) => d.confidence < this.highThresh);

    const matchedTrackIndices = new Set<number>();
    const matchedHighDetIndices = new Set<number>();

    // Stage 1: Match active tracks against high-confidence detections
    for (let t = 0; t < this.tracks.length; t++) {
      const track = this.tracks[t];
      let bestIou = this.iouMatchThresh;
      let bestDetIdx = -1;

      // Predict shifted box using smoothed velocity
      const predictedBox: ScreenBox = {
        x: track.screenBox.x + track.vx * 0.4,
        y: track.screenBox.y + track.vy * 0.4,
        width: track.screenBox.width,
        height: track.screenBox.height,
      };

      for (let d = 0; d < highDets.length; d++) {
        if (matchedHighDetIndices.has(d)) continue;
        if (highDets[d].className !== track.className) continue;

        const iou = computeIoU(predictedBox, highDets[d].screenBox);
        if (iou > bestIou) {
          bestIou = iou;
          bestDetIdx = d;
        }
      }

      if (bestDetIdx !== -1) {
        const det = highDets[bestDetIdx];
        const newCx = det.screenBox.x + det.screenBox.width / 2;
        const newCy = det.screenBox.y + det.screenBox.height / 2;
        const dx = newCx - track.centerX;
        const dy = newCy - track.centerY;

        this.tracks[t] = {
          ...det,
          trackId: track.trackId,
          age: track.age + 1,
          lostFrames: 0,
          vx: track.vx * 0.65 + dx * 0.35,
          vy: track.vy * 0.65 + dy * 0.35,
          centerX: newCx,
          centerY: newCy,
        };
        matchedTrackIndices.add(t);
        matchedHighDetIndices.add(bestDetIdx);
      }
    }

    // Stage 2: ByteTrack recovery — match remaining unmatched tracks with low-confidence detections
    const matchedLowDetIndices = new Set<number>();
    for (let t = 0; t < this.tracks.length; t++) {
      if (matchedTrackIndices.has(t)) continue;
      const track = this.tracks[t];
      let bestIou = this.iouMatchThresh;
      let bestDetIdx = -1;

      for (let d = 0; d < lowDets.length; d++) {
        if (matchedLowDetIndices.has(d)) continue;
        if (lowDets[d].className !== track.className) continue;

        const iou = computeIoU(track.screenBox, lowDets[d].screenBox);
        if (iou > bestIou) {
          bestIou = iou;
          bestDetIdx = d;
        }
      }

      if (bestDetIdx !== -1) {
        const det = lowDets[bestDetIdx];
        const newCx = det.screenBox.x + det.screenBox.width / 2;
        const newCy = det.screenBox.y + det.screenBox.height / 2;
        this.tracks[t] = {
          ...det,
          trackId: track.trackId,
          age: track.age + 1,
          lostFrames: 0,
          vx: track.vx * 0.7 + (newCx - track.centerX) * 0.3,
          vy: track.vy * 0.7 + (newCy - track.centerY) * 0.3,
          centerX: newCx,
          centerY: newCy,
        };
        matchedTrackIndices.add(t);
        matchedLowDetIndices.add(bestDetIdx);
      }
    }

    // Age unmatched tracks & prune expired tracks
    const nextTracks: TrackedObject[] = [];
    for (let t = 0; t < this.tracks.length; t++) {
      if (matchedTrackIndices.has(t)) {
        nextTracks.push(this.tracks[t]);
      } else {
        const aged = {
          ...this.tracks[t],
          lostFrames: this.tracks[t].lostFrames + 1,
        };
        if (aged.lostFrames <= this.maxLostFrames) {
          nextTracks.push(aged);
        }
      }
    }

    // Spawn new tracks for unmatched high-confidence detections
    for (let d = 0; d < highDets.length; d++) {
      if (matchedHighDetIndices.has(d)) continue;
      const det = highDets[d];
      const cx = det.screenBox.x + det.screenBox.width / 2;
      const cy = det.screenBox.y + det.screenBox.height / 2;
      nextTracks.push({
        ...det,
        trackId: this.nextId++,
        age: 1,
        lostFrames: 0,
        vx: 0,
        vy: 0,
        centerX: cx,
        centerY: cy,
      });
    }

    this.tracks = nextTracks;
    // Return currently visible detections (lostFrames === 0)
    return this.tracks.filter((tr) => tr.lostFrames === 0);
  }
}

export function classifySceneDensity(totalObjects: number): DensityLevel {
  if (totalObjects < 3) return 'Low';
  if (totalObjects <= 7) return 'Moderate';
  return 'High';
}

export function countByBucket(tracks: TrackedObject[]): {
  persons: number;
  vehicles: number;
  bicycles: number;
} {
  let persons = 0;
  let vehicles = 0;
  let bicycles = 0;
  for (const t of tracks) {
    if (t.className === 'person') persons++;
    else if (t.className === 'car' || t.className === 'bus') vehicles++;
    else if (t.className === 'bicycle') bicycles++;
  }
  return { persons, vehicles, bicycles };
}

export function classToId(cls: TargetClass): number {
  switch (cls) {
    case 'person':
      return 0;
    case 'bicycle':
      return 1;
    case 'car':
      return 2;
    case 'bus':
      return 5;
  }
}
