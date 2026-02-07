/**
 * Module-level store for grab state.
 *
 * The gesture hook (running in setInterval) WRITES here.
 * PhysicsObject / CameraController (running in useFrame) READ from here.
 */
import type { Vector3Tuple } from '@/types';

export interface GrabState {
  /** ID of the currently-grabbed object, or null */
  objectId: string | null;
  /** Target world position the grabbed object should move to */
  targetPosition: Vector3Tuple;
  /** Twist angle (radians) applied around the view axis */
  twistAngle: number;
  /** Camera forward direction captured at grab start (rotation axis) */
  twistAxis: Vector3Tuple;
  /** Uniform scale factor driven by the second hand's finger spread (1 = unchanged) */
  scaleFactor: number;
  /** Throw velocity to apply on release */
  releaseVelocity: Vector3Tuple;
  /** Set to true for one frame when the object should be released */
  pendingRelease: boolean;
}

export const grabState: GrabState = {
  objectId: null,
  targetPosition: [0, 0, 0],
  twistAngle: 0,
  twistAxis: [0, 0, -1],
  scaleFactor: 1,
  releaseVelocity: [0, 0, 0],
  pendingRelease: false,
};

// ── Camera zoom driven by pinch-on-empty-space ───────────────
export const cameraZoomState = {
  /** Current orbit radius (modified by pinch zoom) */
  radius: 14,
};
