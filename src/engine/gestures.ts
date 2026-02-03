import type { NormalizedLandmark } from '@mediapipe/hands';
import type { GestureType, HandLandmarks } from '@/types';
import { HAND_LANDMARKS } from '@/types';

// Calculate distance between two landmarks
export function distance(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) +
    Math.pow(a.y - b.y, 2) +
    Math.pow(a.z - b.z, 2)
  );
}

// Calculate 2D distance (ignoring z)
export function distance2D(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) +
    Math.pow(a.y - b.y, 2)
  );
}

// Check if a finger is extended by comparing joint positions
export function isFingerExtended(
  landmarks: NormalizedLandmark[],
  fingerTip: number,
  fingerPip: number,
  _fingerMcp: number
): boolean {
  const tip = landmarks[fingerTip];
  const pip = landmarks[fingerPip];
  const wrist = landmarks[HAND_LANDMARKS.WRIST];

  // Check if fingertip is further from wrist than PIP joint
  const tipDist = distance2D(tip, wrist);
  const pipDist = distance2D(pip, wrist);
  
  // Also check y-coordinate (up is negative in normalized coordinates)
  // For fingers, tip should be above (less y) than pip when extended
  const isAbove = tip.y < pip.y;
  
  return tipDist > pipDist && isAbove;
}

// Check if thumb is extended
export function isThumbExtended(landmarks: NormalizedLandmark[]): boolean {
  const thumbTip = landmarks[HAND_LANDMARKS.THUMB_TIP];
  const thumbIp = landmarks[HAND_LANDMARKS.THUMB_IP];
  const indexMcp = landmarks[HAND_LANDMARKS.INDEX_MCP];

  // Thumb is extended if tip is far from index MCP
  const distFromIndex = distance2D(thumbTip, indexMcp);
  const ipDistFromIndex = distance2D(thumbIp, indexMcp);
  
  return distFromIndex > ipDistFromIndex;
}

// Get finger states (extended/curled)
export interface FingerStates {
  thumb: boolean;
  index: boolean;
  middle: boolean;
  ring: boolean;
  pinky: boolean;
}

export function getFingerStates(landmarks: NormalizedLandmark[]): FingerStates {
  return {
    thumb: isThumbExtended(landmarks),
    index: isFingerExtended(
      landmarks,
      HAND_LANDMARKS.INDEX_TIP,
      HAND_LANDMARKS.INDEX_PIP,
      HAND_LANDMARKS.INDEX_MCP
    ),
    middle: isFingerExtended(
      landmarks,
      HAND_LANDMARKS.MIDDLE_TIP,
      HAND_LANDMARKS.MIDDLE_PIP,
      HAND_LANDMARKS.MIDDLE_MCP
    ),
    ring: isFingerExtended(
      landmarks,
      HAND_LANDMARKS.RING_TIP,
      HAND_LANDMARKS.RING_PIP,
      HAND_LANDMARKS.RING_MCP
    ),
    pinky: isFingerExtended(
      landmarks,
      HAND_LANDMARKS.PINKY_TIP,
      HAND_LANDMARKS.PINKY_PIP,
      HAND_LANDMARKS.PINKY_MCP
    ),
  };
}

// Calculate pinch distance between thumb and index finger
export function getPinchDistance(landmarks: NormalizedLandmark[]): number {
  const thumbTip = landmarks[HAND_LANDMARKS.THUMB_TIP];
  const indexTip = landmarks[HAND_LANDMARKS.INDEX_TIP];
  return distance(thumbTip, indexTip);
}

// Check if pinching (thumb and index close together)
export function isPinching(landmarks: NormalizedLandmark[], threshold = 0.05): boolean {
  return getPinchDistance(landmarks) < threshold;
}

// Get the pointing direction from index finger
export function getPointingDirection(landmarks: NormalizedLandmark[]): { x: number; y: number } {
  const indexTip = landmarks[HAND_LANDMARKS.INDEX_TIP];
  const indexMcp = landmarks[HAND_LANDMARKS.INDEX_MCP];
  
  return {
    x: indexTip.x - indexMcp.x,
    y: indexTip.y - indexMcp.y,
  };
}

// Get the center of the palm
export function getPalmCenter(landmarks: NormalizedLandmark[]): NormalizedLandmark {
  const palmLandmarks = [
    landmarks[HAND_LANDMARKS.WRIST],
    landmarks[HAND_LANDMARKS.INDEX_MCP],
    landmarks[HAND_LANDMARKS.MIDDLE_MCP],
    landmarks[HAND_LANDMARKS.RING_MCP],
    landmarks[HAND_LANDMARKS.PINKY_MCP],
  ];

  const center = palmLandmarks.reduce(
    (acc, l) => ({
      x: acc.x + l.x / palmLandmarks.length,
      y: acc.y + l.y / palmLandmarks.length,
      z: acc.z + l.z / palmLandmarks.length,
    }),
    { x: 0, y: 0, z: 0 }
  );

  return center as NormalizedLandmark;
}

// Detect gesture from landmarks
export function detectGesture(landmarks: NormalizedLandmark[]): GestureType {
  const fingers = getFingerStates(landmarks);
  const pinching = isPinching(landmarks);
  
  // Count extended fingers
  const extendedCount = [
    fingers.thumb,
    fingers.index,
    fingers.middle,
    fingers.ring,
    fingers.pinky,
  ].filter(Boolean).length;

  // Pinch gesture (thumb and index together, other fingers can vary)
  if (pinching) {
    // Two-finger pinch: thumb and index pinching, middle also close
    const middleTip = landmarks[HAND_LANDMARKS.MIDDLE_TIP];
    const thumbTip = landmarks[HAND_LANDMARKS.THUMB_TIP];
    const middlePinchDist = distance(middleTip, thumbTip);
    
    if (middlePinchDist < 0.08 && !fingers.ring && !fingers.pinky) {
      return 'two_finger_pinch';
    }
    
    return 'pinch';
  }

  // Pointing gesture (only index extended)
  if (fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky) {
    return 'point';
  }

  // Open palm (all fingers extended)
  if (extendedCount >= 4) {
    return 'open_palm';
  }

  // Fist (no fingers extended)
  if (extendedCount === 0) {
    return 'fist';
  }

  return 'none';
}

// Swipe detection state
interface SwipeState {
  startX: number;
  startTime: number;
  isTracking: boolean;
}

const swipeState: SwipeState = {
  startX: 0,
  startTime: 0,
  isTracking: false,
};

// Detect swipe gesture
export function detectSwipe(
  landmarks: NormalizedLandmark[],
  gesture: GestureType
): GestureType | null {
  const palmCenter = getPalmCenter(landmarks);
  const currentTime = Date.now();

  if (gesture === 'open_palm') {
    if (!swipeState.isTracking) {
      swipeState.startX = palmCenter.x;
      swipeState.startTime = currentTime;
      swipeState.isTracking = true;
      return null;
    }

    const deltaX = palmCenter.x - swipeState.startX;
    const deltaTime = currentTime - swipeState.startTime;

    // Check for swipe (movement threshold and time limit)
    if (deltaTime < 500 && Math.abs(deltaX) > 0.15) {
      swipeState.isTracking = false;
      return deltaX > 0 ? 'swipe_right' : 'swipe_left';
    }

    // Reset if too slow
    if (deltaTime > 500) {
      swipeState.startX = palmCenter.x;
      swipeState.startTime = currentTime;
    }
  } else {
    swipeState.isTracking = false;
  }

  return null;
}

// Get pointer position from index fingertip (normalized 0-1)
export function getPointerPosition(landmarks: NormalizedLandmark[]): { x: number; y: number } {
  const indexTip = landmarks[HAND_LANDMARKS.INDEX_TIP];
  return {
    x: indexTip.x,
    y: indexTip.y,
  };
}

// Get grab position from pinch center
export function getGrabPosition(landmarks: NormalizedLandmark[]): { x: number; y: number; z: number } {
  const thumbTip = landmarks[HAND_LANDMARKS.THUMB_TIP];
  const indexTip = landmarks[HAND_LANDMARKS.INDEX_TIP];
  
  return {
    x: (thumbTip.x + indexTip.x) / 2,
    y: (thumbTip.y + indexTip.y) / 2,
    z: (thumbTip.z + indexTip.z) / 2,
  };
}

// Process hand results and return gesture state
export interface GestureResult {
  gesture: GestureType;
  pinchDistance: number;
  isPinching: boolean;
  pointerPosition: { x: number; y: number } | null;
  grabPosition: { x: number; y: number; z: number } | null;
  fingerStates: FingerStates;
}

export function processHandGesture(hand: HandLandmarks): GestureResult {
  const { landmarks } = hand;
  const gesture = detectGesture(landmarks);
  const swipe = detectSwipe(landmarks, gesture);
  const pinchDist = getPinchDistance(landmarks);
  const pinching = isPinching(landmarks);
  const fingerStates = getFingerStates(landmarks);

  // Get pointer position for pointing gestures
  let pointerPosition: { x: number; y: number } | null = null;
  if (gesture === 'point' || gesture === 'pinch') {
    pointerPosition = getPointerPosition(landmarks);
  }

  // Get grab position for pinch gestures
  let grabPosition: { x: number; y: number; z: number } | null = null;
  if (pinching) {
    grabPosition = getGrabPosition(landmarks);
  }

  return {
    gesture: swipe || gesture,
    pinchDistance: pinchDist,
    isPinching: pinching,
    pointerPosition,
    grabPosition,
    fingerStates,
  };
}
