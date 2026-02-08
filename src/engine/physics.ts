/**
 * Velocity tracker for throw mechanics.
 * Records recent positions and computes a release velocity vector.
 */
import type { Vector3Tuple } from '@/types';

interface PositionSample {
  position: Vector3Tuple;
  time: number; 
}

const MAX_SAMPLES = 8;
const VELOCITY_SCALE = 2.5; 
const MIN_THROW_SPEED = 1.5; 
const MAX_THROW_SPEED = 30; 

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

    
    const recent = this.samples.slice(-4);
    const first = recent[0];
    const last = recent[recent.length - 1];
    const dt = (last.time - first.time) / 1000; 

    if (dt < 0.001) return [0, 0, 0];

    const vx = ((last.position[0] - first.position[0]) / dt) * VELOCITY_SCALE;
    const vy = ((last.position[1] - first.position[1]) / dt) * VELOCITY_SCALE;
    const vz = ((last.position[2] - first.position[2]) / dt) * VELOCITY_SCALE;

    const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

    
    if (speed < MIN_THROW_SPEED) return [0, 0, 0];

    
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


const GROUND_HALF = 14; 

export function constrainToGround(pos: Vector3Tuple, minY = 0.25): Vector3Tuple {
  return [
    Math.max(-GROUND_HALF, Math.min(GROUND_HALF, pos[0])),
    Math.max(minY, pos[1]),
    Math.max(-GROUND_HALF, Math.min(GROUND_HALF, pos[2])),
  ];
}
