/**
 * Module-level store for grab state.
 *
 * The gesture hook (running in setInterval) WRITES here.
 * PhysicsObject components (running in useFrame) READ from here
 * to control their body type and position within the R3F render loop.
 */
import type { Vector3Tuple } from '@/types';

export interface GrabState {
  /** ID of the currently-grabbed object, or null */
  objectId: string | null;
  /** Target world position the grabbed object should move to */
  targetPosition: Vector3Tuple;
  /** Throw velocity to apply on release */
  releaseVelocity: Vector3Tuple;
  /** Set to true for one frame when the object should be released */
  pendingRelease: boolean;
}

export const grabState: GrabState = {
  objectId: null,
  targetPosition: [0, 0, 0],
  releaseVelocity: [0, 0, 0],
  pendingRelease: false,
};
