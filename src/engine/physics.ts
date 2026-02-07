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
const MIN_THROW_SPEED = 1.5; // units/sec – below this the object just drops
const MAX_THROW_SPEED = 30; // cap so objects don't fly off the scene

export class VelocityTracker {
  private samples: PositionSample[] = [];

  /** Record a position sample at the current time. */
  record(position: Vector3Tuple): void {
    this.samples.push({ position, time: performance.now() });
    if (this.samples.length > MAX_SAMPLES) {
      this.samples.shift();
    }
  }

  /** Compute average velocity (units/sec) from recent samples.
   *  Returns [0,0,0] if the speed is below MIN_THROW_SPEED so that
   *  releasing a stationary object simply drops it under gravity. */
  getVelocity(): Vector3Tuple {
    if (this.samples.length < 2) return [0, 0, 0];

    // Use only the last ~4 samples for responsiveness
    const recent = this.samples.slice(-4);
    const first = recent[0];
    const last = recent[recent.length - 1];
    const dt = (last.time - first.time) / 1000; // seconds

    if (dt < 0.001) return [0, 0, 0];

    const vx = ((last.position[0] - first.position[0]) / dt) * VELOCITY_SCALE;
    const vy = ((last.position[1] - first.position[1]) / dt) * VELOCITY_SCALE;
    const vz = ((last.position[2] - first.position[2]) / dt) * VELOCITY_SCALE;

    const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

    // Below the threshold → just drop (no impulse)
    if (speed < MIN_THROW_SPEED) return [0, 0, 0];

    // Clamp to max speed
    if (speed > MAX_THROW_SPEED) {
      const scale = MAX_THROW_SPEED / speed;
      return [vx * scale, vy * scale, vz * scale];
    }

    return [vx, vy, vz];
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
