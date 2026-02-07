/**
 * Velocity tracker for throw mechanics.
 * Records recent positions and computes a release velocity vector.
 */
import type { Vector3Tuple } from '@/types';

interface PositionSample {
  position: Vector3Tuple;
  time: number; // ms timestamp
}

const MAX_SAMPLES = 8;
const VELOCITY_SCALE = 2.5; // amplification factor for thrown objects

export class VelocityTracker {
  private samples: PositionSample[] = [];

  /** Record a position sample at the current time. */
  record(position: Vector3Tuple): void {
    this.samples.push({ position, time: performance.now() });
    if (this.samples.length > MAX_SAMPLES) {
      this.samples.shift();
    }
  }

  /** Compute average velocity (units/sec) from recent samples. */
  getVelocity(): Vector3Tuple {
    if (this.samples.length < 2) return [0, 0, 0];

    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const dt = (last.time - first.time) / 1000; // seconds

    if (dt < 0.001) return [0, 0, 0];

    return [
      ((last.position[0] - first.position[0]) / dt) * VELOCITY_SCALE,
      ((last.position[1] - first.position[1]) / dt) * VELOCITY_SCALE,
      ((last.position[2] - first.position[2]) / dt) * VELOCITY_SCALE,
    ];
  }

  /** Reset the sample buffer. */
  reset(): void {
    this.samples = [];
  }
}

// ── Ground constraint ────────────────────────────────────────
const GROUND_HALF = 14; // objects stay within ±GROUND_HALF on x/z

export function constrainToGround(pos: Vector3Tuple, minY = 0.25): Vector3Tuple {
  return [
    Math.max(-GROUND_HALF, Math.min(GROUND_HALF, pos[0])),
    Math.max(minY, pos[1]),
    Math.max(-GROUND_HALF, Math.min(GROUND_HALF, pos[2])),
  ];
}
