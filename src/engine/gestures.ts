/**
 * Pure gesture-detection utilities.
 * No React / Three.js imports – operates only on landmark data.
 */
import type { NormalizedLandmark, GestureType, Vector3Tuple } from '@/types';
import { HAND_LANDMARKS } from '@/types';

// ── Distances ────────────────────────────────────────────────
export function distance3D(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

export function distance2D(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// ── Pinch detection with hysteresis, smoothing & debounce ────
//
// PINCH_ON  : distance must drop below this to START a pinch
// PINCH_OFF : distance must exceed this to STOP a pinch
// The gap between the two prevents rapid toggling.
const PINCH_ON = 0.10;
const PINCH_OFF = 0.14;

// Exponential-moving-average factor (0-1). Lower = smoother but laggier.
const SMOOTH_FACTOR = 0.35;

// How many consecutive "not pinching" frames before we actually release.
const RELEASE_GRACE_FRAMES = 4;

// ── Module-level pinch state (persists across calls) ─────────
let smoothedDistance = 1;
let pinchActive = false;
let releaseCounter = 0;

export function getPinchDistance(landmarks: NormalizedLandmark[]): number {
  const thumb = landmarks[HAND_LANDMARKS.THUMB_TIP];
  const index = landmarks[HAND_LANDMARKS.INDEX_TIP];
  return distance3D(thumb, index);
}

/** Returns the smoothed pinch state. Call once per frame. */
export function isPinching(landmarks: NormalizedLandmark[]): boolean {
  const raw = getPinchDistance(landmarks);

  // Exponential moving average
  smoothedDistance = smoothedDistance * (1 - SMOOTH_FACTOR) + raw * SMOOTH_FACTOR;

  if (!pinchActive) {
    // Need to cross the tighter threshold to begin
    if (smoothedDistance < PINCH_ON) {
      pinchActive = true;
      releaseCounter = 0;
    }
  } else {
    // Already pinching – only release after exceeding the wider threshold
    // for several consecutive frames (grace period).
    if (smoothedDistance > PINCH_OFF) {
      releaseCounter++;
      if (releaseCounter >= RELEASE_GRACE_FRAMES) {
        pinchActive = false;
        releaseCounter = 0;
      }
    } else {
      // Still within range – reset the grace counter
      releaseCounter = 0;
    }
  }

  return pinchActive;
}

/** Reset pinch state (e.g. when hand disappears). */
export function resetPinchState(): void {
  smoothedDistance = 1;
  pinchActive = false;
  releaseCounter = 0;
}

// ── Finger extension checks ──────────────────────────────────
function isFingerExtended(
  landmarks: NormalizedLandmark[],
  _mcp: number,
  pip: number,
  tip: number,
): boolean {
  const wrist = landmarks[HAND_LANDMARKS.WRIST];
  const tipDist = distance2D(landmarks[tip], wrist);
  const pipDist = distance2D(landmarks[pip], wrist);
  return tipDist > pipDist;
}

export function isIndexExtended(landmarks: NormalizedLandmark[]): boolean {
  return isFingerExtended(
    landmarks,
    HAND_LANDMARKS.INDEX_MCP,
    HAND_LANDMARKS.INDEX_PIP,
    HAND_LANDMARKS.INDEX_TIP,
  );
}

export function isMiddleExtended(landmarks: NormalizedLandmark[]): boolean {
  return isFingerExtended(
    landmarks,
    HAND_LANDMARKS.MIDDLE_MCP,
    HAND_LANDMARKS.MIDDLE_PIP,
    HAND_LANDMARKS.MIDDLE_TIP,
  );
}

export function isRingExtended(landmarks: NormalizedLandmark[]): boolean {
  return isFingerExtended(
    landmarks,
    HAND_LANDMARKS.RING_MCP,
    HAND_LANDMARKS.RING_PIP,
    HAND_LANDMARKS.RING_TIP,
  );
}

export function isPinkyExtended(landmarks: NormalizedLandmark[]): boolean {
  return isFingerExtended(
    landmarks,
    HAND_LANDMARKS.PINKY_MCP,
    HAND_LANDMARKS.PINKY_PIP,
    HAND_LANDMARKS.PINKY_TIP,
  );
}

// ── Gesture detection ────────────────────────────────────────
// Accepts the *already-computed* pinch boolean so that isPinching()
// is only called once per frame (it has module-level state).
export function detectGesture(
  landmarks: NormalizedLandmark[],
  pinchResult: boolean,
): GestureType {
  if (pinchResult) return 'pinch';

  // Point = index extended, others curled
  const indexUp = isIndexExtended(landmarks);
  const middleUp = isMiddleExtended(landmarks);
  const ringUp = isRingExtended(landmarks);
  const pinkyUp = isPinkyExtended(landmarks);

  if (indexUp && !middleUp && !ringUp && !pinkyUp) return 'point';

  return 'none';
}

// ── Pointer position (normalised 0-1, mirrored for webcam) ──
export function getPointerScreenPosition(
  landmarks: NormalizedLandmark[],
): { x: number; y: number } {
  const index = landmarks[HAND_LANDMARKS.INDEX_TIP];
  // Mirror x because webcam is flipped
  return { x: 1 - index.x, y: index.y };
}

// ── Pinch midpoint (between thumb tip & index tip) ──────────
export function getPinchMidpoint(
  landmarks: NormalizedLandmark[],
): { x: number; y: number } {
  const thumb = landmarks[HAND_LANDMARKS.THUMB_TIP];
  const index = landmarks[HAND_LANDMARKS.INDEX_TIP];
  return {
    x: 1 - (thumb.x + index.x) / 2,
    y: (thumb.y + index.y) / 2,
  };
}

// ── World-space projection helpers ───────────────────────────
// These utilities convert normalised screen coords to a 3D
// world position on a horizontal plane at y = groundY.
import * as THREE from 'three';

export function screenToWorld(
  screenPos: { x: number; y: number },
  camera: THREE.Camera,
  groundY = 0,
): Vector3Tuple {
  // Convert 0-1 screen coords to NDC (-1 to 1)
  const ndc = new THREE.Vector2(screenPos.x * 2 - 1, -(screenPos.y * 2 - 1));

  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(ndc, camera);

  // Intersect with horizontal plane at groundY
  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -groundY);
  const target = new THREE.Vector3();
  raycaster.ray.intersectPlane(plane, target);

  if (!target) return [0, groundY, 0];
  return [target.x, target.y, target.z];
}
