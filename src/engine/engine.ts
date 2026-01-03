import { Hands, HAND_CONNECTIONS, type Results } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import type { Point2, PhysicsBody } from './types';

export type EngineMountArgs = {
  overlay: HTMLCanvasElement;
  video: HTMLVideoElement;
  onFirstResults?: () => void;
};

export type EngineSettings = {
  zoom: number;
  gravity: number;
  floorHeight: number;
  bounciness: number;
  friction: number;
};

export type Engine = {
  mount(args: EngineMountArgs): void;
  unmount(): void;
  applySettings(settings: EngineSettings): void;
  addBody(shape: 'mixed' | 'circle' | 'box'): void;
  addRandomBody(): void;
  resetScene(): void;
};

const CONFIG = {
  maxNumHands: 2,

  // pinch in normalized landmark units
  pinchStart: 0.055,
  pinchEnd: 0.075,

  // grasp (closed hand) in normalized landmark units (avg fingertip->MCP distance)
  graspStart: 0.085,
  graspEnd: 0.105,

  // visual style
  jointRadius: 6,
  boneWidth: 3,
  boneAlpha: 0.8,
  palmAlpha: 0.18,
  glowAlpha: 0.35,
  smoothing: 0.5, // 0..1 (higher = smoother)

  // 2D physics
  physics: {
    gravity: 2200, // px/s^2
    airDrag: 0.02, // per second
    floorHeight: 140, // px from bottom
    substeps: 2,
    maxDt: 1 / 30,

    // grabbing
    grabRadius: 70,
    grabStiffness: 55, // spring strength
    grabDamping: 12, // velocity damping
    throwScale: 1.25,

    // grab responsiveness
    grabFollow: 0.9, // 0..1 velocity/position follow strength while grabbed
    grabPosGain: 18, // 1/s extra velocity from position error
    maxGrabSpeed: 5200, // world px/s

    // throw stability
    throwSmoothing: 0.35, // 0..1 (higher = more responsive)
    maxThrowSpeed: 3600, // world px/s
    throwBlend: 0.9, // 0..1 blend from current vel -> hand vel

    // throw sampling window (more stable than single-frame velocity)
    throwWindowMs: 140,
    throwMinSamples: 2,

    // playground
    hoopEnabled: true,
    hoopRestitution: 0.75,
    hoopFriction: 0.25,

    // hand/body interaction
    handCollisions: true,
    handPalmRadius: 58, // screen px (converted to world)
    handPointRadius: 18, // screen px (converted to world)
    handRestitution: 0.65,
    throwHandIgnoreMs: 140, // ignore collisions with releasing hand briefly
  },
} as const;

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function clampVecMag(v: Point2, maxMag: number): Point2 {
  const m2 = v.x * v.x + v.y * v.y;
  if (m2 <= maxMag * maxMag) return v;
  const m = Math.sqrt(Math.max(m2, 1e-12));
  const s = maxMag / m;
  return { x: v.x * s, y: v.y * s };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function robustAverageVelocity(samples: Array<{ v: Point2; t: number }>, maxSpeed: number): Point2 {
  if (samples.length === 0) return { x: 0, y: 0 };
  if (samples.length === 1) return clampVecMag(samples[0].v, maxSpeed);

  // Component-wise median to filter spikes.
  const mx = median(samples.map((s) => s.v.x));
  const my = median(samples.map((s) => s.v.y));

  // Keep samples reasonably close to the median (robust trimming).
  const filtered = samples.filter((s) => {
    const dx = s.v.x - mx;
    const dy = s.v.y - my;
    return dx * dx + dy * dy <= (maxSpeed * 0.65) * (maxSpeed * 0.65);
  });

  const use = filtered.length > 0 ? filtered : samples;

  // Weighted average favoring newer samples.
  const tMax = use[use.length - 1].t;
  let sumW = 0;
  let sumX = 0;
  let sumY = 0;
  for (const s of use) {
    const age = Math.max(0, tMax - s.t);
    const w = Math.exp(-age / 70); // ~70ms half-life
    sumW += w;
    sumX += s.v.x * w;
    sumY += s.v.y * w;
  }

  const out = {
    x: sumW > 0 ? sumX / sumW : mx,
    y: sumW > 0 ? sumY / sumW : my,
  };
  return clampVecMag(out, maxSpeed);
}

function smoothPoints(prev: Point2[] | null, next: Point2[], smoothing: number): Point2[] {
  if (!prev || prev.length !== next.length) return next;
  const s = Math.min(0.98, Math.max(0, smoothing));
  const a = 1 - s;
  return next.map((p, i) => ({
    x: prev[i].x + (p.x - prev[i].x) * a,
    y: prev[i].y + (p.y - prev[i].y) * a,
  }));
}

function dist(a: Point2, b: Point2) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function randColor() {
  const palette = ['#00ffdf', '#00ff88', '#66ffff', '#ff3355', '#ffd166', '#a78bfa'];
  return palette[Math.floor(Math.random() * palette.length)];
}

export function createEngine(): Engine {
  // DOM refs
  let overlay: HTMLCanvasElement | null = null;
  let ctx: CanvasRenderingContext2D | null = null;
  let video: HTMLVideoElement | null = null;

  // MediaPipe
  let hands: Hands | null = null;
  let camera: Camera | null = null;

  // animation
  let rafId: number | null = null;
  let running = false;
  let lastT = 0;

  let onFirstResultsCb: (() => void) | undefined;
  let firstResultsEmitted = false;

  // 2D camera (world -> screen). Zoom < 1 = zoom out
  const camera2d = {
    zoom: 1,
    centerWorld: { x: 0, y: 0 },
  };

  // ---------- Physics World ----------
  let nextId = 1;
  const bodies: PhysicsBody[] = [];

  type StaticCollider =
    | { kind: 'circle'; pos: Point2; r: number; restitution: number; friction: number }
    | {
        kind: 'box';
        pos: Point2;
        size: { w: number; h: number };
        restitution: number;
        friction: number;
      };

  const playground = {
    score: 0,
    lastBodyY: new Map<number, number>(),
  };

  type PointerState = {
    pinch: Point2 | null; // in SCREEN px
    prevPinch: Point2 | null;
    pinchVel: Point2; // in WORLD px/s
    lastPinchT: number;
  };

  type HandState = {
    pointer: PointerState;
    grabActiveId: number | null;
    lastLandmarksPx: Point2[] | null;
    isPinching: boolean;
    pinchPosPx: Point2 | null;

    isGrasping: boolean;
    palmPosPx: Point2 | null;
    grabPosPx: Point2 | null;

    // Recent grab-point velocities for robust throws
    throwSamples: Array<{ v: Point2; t: number }>;

    // Kinematic colliders in WORLD space (for bouncing bodies off the hand)
    colliders: Array<{ handIndex: number; pos: Point2; vel: Point2; r: number }>;
    prevColliderPos: Array<Point2>;
  };

  const handsState: HandState[] = Array.from({ length: CONFIG.maxNumHands }, () => ({
    pointer: {
      pinch: null,
      prevPinch: null,
      pinchVel: { x: 0, y: 0 },
      lastPinchT: 0,
    },
    grabActiveId: null,
    lastLandmarksPx: null,
    isPinching: false,
    pinchPosPx: null,

    isGrasping: false,
    palmPosPx: null,
    grabPosPx: null,

    throwSamples: [],

    colliders: [],
    prevColliderPos: [],
  }));

  // bodyId -> (handIndex -> ignoreUntilMs)
  const handCollisionIgnoreUntil = new Map<number, Map<number, number>>();

  function setHandCollisionIgnore(bodyId: number, handIndex: number, untilMs: number) {
    let byHand = handCollisionIgnoreUntil.get(bodyId);
    if (!byHand) {
      byHand = new Map<number, number>();
      handCollisionIgnoreUntil.set(bodyId, byHand);
    }
    byHand.set(handIndex, untilMs);
  }

  function shouldIgnoreHandCollision(bodyId: number, handIndex: number, nowMs: number): boolean {
    const byHand = handCollisionIgnoreUntil.get(bodyId);
    if (!byHand) return false;
    const until = byHand.get(handIndex);
    return until != null && nowMs < until;
  }

  function grabbedIds(): Set<number> {
    const s = new Set<number>();
    for (const hs of handsState) if (hs.grabActiveId != null) s.add(hs.grabActiveId);
    return s;
  }

  // Voice
  let recognition: SpeechRecognition | null = null;

  function screenToWorld(p: Point2): Point2 {
    return {
      x: (p.x - window.innerWidth / 2) / camera2d.zoom + camera2d.centerWorld.x,
      y: (p.y - window.innerHeight / 2) / camera2d.zoom + camera2d.centerWorld.y,
    };
  }

  function applyCameraTransform() {
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const z = camera2d.zoom;

    ctx.setTransform(
      dpr * z,
      0,
      0,
      dpr * z,
      dpr * (window.innerWidth / 2 - camera2d.centerWorld.x * z),
      dpr * (window.innerHeight / 2 - camera2d.centerWorld.y * z)
    );
  }

  function resetCanvasTransform() {
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function resizeOverlay() {
    if (!overlay || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    overlay.width = Math.floor(window.innerWidth * dpr);
    overlay.height = Math.floor(window.innerHeight * dpr);
    overlay.style.width = `${window.innerWidth}px`;
    overlay.style.height = `${window.innerHeight}px`;

    // First match device pixels; draw loop will apply camera transform afterwards
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function toScreenPoint(lm: any): Point2 {
    // Keep hand in SCREEN coordinates (match the mirrored webcam feel)
    return {
      x: (1 - lm.x) * window.innerWidth,
      y: lm.y * window.innerHeight,
    };
  }

  function getBodyById(id: number | null) {
    if (id == null) return null;
    return bodies.find((b) => b.id === id) ?? null;
  }

  function isBodyGrabbedByOtherHand(bodyId: number, handIndex: number): boolean {
    for (let i = 0; i < handsState.length; i++) {
      if (i === handIndex) continue;
      if (handsState[i].grabActiveId === bodyId) return true;
    }
    return false;
  }

  function bodyPickDistance(b: PhysicsBody, p: Point2): number {
    if (b.kind === 'circle') {
      return dist(b.pos, p) - (b.radius ?? 0);
    }

    const w = b.size!.w;
    const h = b.size!.h;
    const dx = Math.max(Math.abs(p.x - b.pos.x) - w / 2, 0);
    const dy = Math.max(Math.abs(p.y - b.pos.y) - h / 2, 0);
    return Math.hypot(dx, dy);
  }

  function addBody(shape: 'mixed' | 'circle' | 'box' = 'mixed') {
    const isCircle =
      shape === 'circle' ? true : shape === 'box' ? false : Math.random() < 0.6;

    // spawn near top-center in WORLD space
    const spawnW = screenToWorld({ x: window.innerWidth / 2, y: 80 });

    if (isCircle) {
      const r = 18 + Math.random() * 18;
      bodies.push({
        id: nextId++,
        kind: 'circle',
        pos: { x: spawnW.x + (Math.random() - 0.5) * 220, y: spawnW.y + Math.random() * 80 },
        vel: { x: (Math.random() - 0.5) * 220, y: 0 },
        radius: r,
        color: randColor(),
        restitution: settings.restitution,
        friction: settings.friction,
        mass: 1,
      });
    } else {
      const w = 34 + Math.random() * 40;
      const h = 34 + Math.random() * 40;
      bodies.push({
        id: nextId++,
        kind: 'box',
        pos: { x: spawnW.x + (Math.random() - 0.5) * 220, y: spawnW.y + Math.random() * 80 },
        vel: { x: (Math.random() - 0.5) * 220, y: 0 },
        size: { w, h },
        color: randColor(),
        restitution: settings.restitution * 0.8,
        friction: settings.friction,
        mass: 1.2,
      });
    }
  }

  function addRandomBody() {
    addBody('mixed');
  }

  function resetScene() {
    bodies.length = 0;
    nextId = 1;
    for (const hs of handsState) hs.grabActiveId = null;
    handCollisionIgnoreUntil.clear();
    playground.score = 0;
    playground.lastBodyY.clear();
  }

  function floorY() {
    // floor in WORLD coordinates: bottom of visible area minus floor height in *screen pixels* converted to world
    const bottomWorldY = screenToWorld({ x: 0, y: window.innerHeight }).y;
    const floorHWorld = settings.floorHeight / camera2d.zoom;
    return bottomWorldY - floorHWorld;
  }

  function getHoopRig(): {
    colliders: StaticCollider[];
    rimCenter: Point2;
    rimR: number;
    sensor: { x1: number; x2: number; yTop: number; yBottom: number };
  } {
    // Anchor hoop to the screen so it feels like a "playground".
    // Put it on the right, but shifted left so it doesn't hide under the Settings panel.
    const rimCenterS = {
      x: clamp(window.innerWidth - 420, 160, window.innerWidth - 160),
      y: clamp(220, 120, window.innerHeight - 220),
    };
    const rimCenter = screenToWorld(rimCenterS);
    const rimR = 42 / camera2d.zoom;
    const rimThickness = 10 / camera2d.zoom;

    const backboardW = 18 / camera2d.zoom;
    const backboardH = 140 / camera2d.zoom;
    const backboard: StaticCollider = {
      kind: 'box',
      pos: { x: rimCenter.x + rimR + backboardW * 0.6, y: rimCenter.y - 40 / camera2d.zoom },
      size: { w: backboardW, h: backboardH },
      restitution: CONFIG.physics.hoopRestitution,
      friction: CONFIG.physics.hoopFriction,
    };

    const rimLeft: StaticCollider = {
      kind: 'circle',
      pos: { x: rimCenter.x - rimR, y: rimCenter.y },
      r: rimThickness * 0.9,
      restitution: CONFIG.physics.hoopRestitution,
      friction: CONFIG.physics.hoopFriction,
    };
    const rimRight: StaticCollider = {
      kind: 'circle',
      pos: { x: rimCenter.x + rimR, y: rimCenter.y },
      r: rimThickness * 0.9,
      restitution: CONFIG.physics.hoopRestitution,
      friction: CONFIG.physics.hoopFriction,
    };
    const rimFront: StaticCollider = {
      kind: 'circle',
      pos: { x: rimCenter.x, y: rimCenter.y },
      r: rimThickness * 0.85,
      restitution: CONFIG.physics.hoopRestitution,
      friction: CONFIG.physics.hoopFriction,
    };
    const rimBar: StaticCollider = {
      kind: 'box',
      pos: { x: rimCenter.x, y: rimCenter.y - rimThickness * 0.35 },
      size: { w: rimR * 2.05, h: rimThickness * 0.6 },
      restitution: CONFIG.physics.hoopRestitution,
      friction: CONFIG.physics.hoopFriction,
    };

    // Sensor zone: score if a circle passes downward through the opening.
    const sensorW = rimR * 1.6;
    const sensorH = 44 / camera2d.zoom;
    const sensor = {
      x1: rimCenter.x - sensorW / 2,
      x2: rimCenter.x + sensorW / 2,
      yTop: rimCenter.y + rimThickness * 0.6,
      yBottom: rimCenter.y + sensorH,
    };

    return {
      colliders: [backboard, rimLeft, rimRight, rimFront, rimBar],
      rimCenter,
      rimR,
      sensor,
    };
  }

  function resolveBodyVsStatic(b: PhysicsBody, s: StaticCollider) {
    if (s.kind === 'circle') {
      const sx = s.pos.x;
      const sy = s.pos.y;
      const sr = s.r;

      if (b.kind === 'circle') {
        const br = b.radius ?? 0;
        const dx = b.pos.x - sx;
        const dy = b.pos.y - sy;
        const d2 = dx * dx + dy * dy;
        const minD = br + sr;
        if (d2 >= minD * minD) return;

        const d = Math.sqrt(Math.max(d2, 1e-9));
        const nx = dx / d;
        const ny = dy / d;
        const penetration = minD - d;
        b.pos.x += nx * penetration;
        b.pos.y += ny * penetration;

        const velAlongNormal = b.vel.x * nx + b.vel.y * ny;
        if (velAlongNormal < 0) {
          const e = Math.min(b.restitution, s.restitution);
          const j = -(1 + e) * velAlongNormal;
          b.vel.x += j * nx;
          b.vel.y += j * ny;

          const tx = -ny;
          const ty = nx;
          const vt = b.vel.x * tx + b.vel.y * ty;
          const mu = (b.friction + s.friction) * 0.5;
          b.vel.x -= vt * tx * mu * 0.12;
          b.vel.y -= vt * ty * mu * 0.12;
        }
        return;
      }

      // box vs static circle
      const hw = b.size!.w / 2;
      const hh = b.size!.h / 2;
      const closestX = Math.max(b.pos.x - hw, Math.min(sx, b.pos.x + hw));
      const closestY = Math.max(b.pos.y - hh, Math.min(sy, b.pos.y + hh));
      const dx = sx - closestX;
      const dy = sy - closestY;
      const d2 = dx * dx + dy * dy;
      if (d2 >= sr * sr) return;

      const d = Math.sqrt(Math.max(d2, 1e-9));
      let nx = -dx / d;
      let ny = -dy / d;
      if (!isFinite(nx) || !isFinite(ny)) {
        const px = b.pos.x - sx;
        const py = b.pos.y - sy;
        if (Math.abs(px / hw) > Math.abs(py / hh)) {
          nx = px > 0 ? 1 : -1;
          ny = 0;
        } else {
          nx = 0;
          ny = py > 0 ? 1 : -1;
        }
      }

      const penetration = sr - d;
      b.pos.x += nx * penetration;
      b.pos.y += ny * penetration;

      const velAlongNormal = b.vel.x * nx + b.vel.y * ny;
      if (velAlongNormal < 0) {
        const e = Math.min(b.restitution, s.restitution);
        const j = -(1 + e) * velAlongNormal;
        b.vel.x += j * nx;
        b.vel.y += j * ny;

        const tx = -ny;
        const ty = nx;
        const vt = b.vel.x * tx + b.vel.y * ty;
        const mu = (b.friction + s.friction) * 0.5;
        b.vel.x -= vt * tx * mu * 0.12;
        b.vel.y -= vt * ty * mu * 0.12;
      }
      return;
    }

    // static box
    const hw = s.size.w / 2;
    const hh = s.size.h / 2;
    const x1 = s.pos.x - hw;
    const x2 = s.pos.x + hw;
    const y1 = s.pos.y - hh;
    const y2 = s.pos.y + hh;

    if (b.kind === 'circle') {
      const r = b.radius ?? 0;
      const closestX = Math.max(x1, Math.min(b.pos.x, x2));
      const closestY = Math.max(y1, Math.min(b.pos.y, y2));
      const dx = b.pos.x - closestX;
      const dy = b.pos.y - closestY;
      const d2 = dx * dx + dy * dy;
      if (d2 >= r * r) return;

      const d = Math.sqrt(Math.max(d2, 1e-9));
      let nx = dx / d;
      let ny = dy / d;
      if (!isFinite(nx) || !isFinite(ny)) {
        // Push out along shallowest axis
        const left = Math.abs(b.pos.x - x1);
        const right = Math.abs(x2 - b.pos.x);
        const top = Math.abs(b.pos.y - y1);
        const bottom = Math.abs(y2 - b.pos.y);
        const m = Math.min(left, right, top, bottom);
        if (m === left) {
          nx = -1;
          ny = 0;
        } else if (m === right) {
          nx = 1;
          ny = 0;
        } else if (m === top) {
          nx = 0;
          ny = -1;
        } else {
          nx = 0;
          ny = 1;
        }
      }

      const penetration = r - d;
      b.pos.x += nx * penetration;
      b.pos.y += ny * penetration;

      const velAlongNormal = b.vel.x * nx + b.vel.y * ny;
      if (velAlongNormal < 0) {
        const e = Math.min(b.restitution, s.restitution);
        const j = -(1 + e) * velAlongNormal;
        b.vel.x += j * nx;
        b.vel.y += j * ny;

        const tx = -ny;
        const ty = nx;
        const vt = b.vel.x * tx + b.vel.y * ty;
        const mu = (b.friction + s.friction) * 0.5;
        b.vel.x -= vt * tx * mu * 0.12;
        b.vel.y -= vt * ty * mu * 0.12;
      }
      return;
    }

    // box vs static box (AABB)
    const bw = b.size!.w;
    const bh = b.size!.h;
    const bx1 = b.pos.x - bw / 2;
    const bx2 = b.pos.x + bw / 2;
    const by1 = b.pos.y - bh / 2;
    const by2 = b.pos.y + bh / 2;
    if (!(bx1 < x2 && bx2 > x1 && by1 < y2 && by2 > y1)) return;

    const overlapX = Math.min(bx2 - x1, x2 - bx1);
    const overlapY = Math.min(by2 - y1, y2 - by1);
    if (overlapX < overlapY) {
      const nx = b.pos.x < s.pos.x ? -1 : 1;
      b.pos.x += nx * overlapX;
      if (b.vel.x * nx < 0) b.vel.x = -b.vel.x * Math.min(b.restitution, s.restitution);
      b.vel.y *= 1 - ((b.friction + s.friction) * 0.5) * 0.04;
    } else {
      const ny = b.pos.y < s.pos.y ? -1 : 1;
      b.pos.y += ny * overlapY;
      if (b.vel.y * ny < 0) b.vel.y = -b.vel.y * Math.min(b.restitution, s.restitution);
      b.vel.x *= 1 - ((b.friction + s.friction) * 0.5) * 0.04;
    }
  }

  function resolveHoopCollisions() {
    if (!CONFIG.physics.hoopEnabled) return;
    const rig = getHoopRig();

    for (let pass = 0; pass < 2; pass++) {
      for (const b of bodies) {
        for (const s of rig.colliders) resolveBodyVsStatic(b, s);
      }
    }

    // Scoring: count circles that pass downward through the sensor window
    for (const b of bodies) {
      if (b.kind !== 'circle') continue;
      const prevY = playground.lastBodyY.get(b.id);
      playground.lastBodyY.set(b.id, b.pos.y);
      if (prevY == null) continue;

      const x = b.pos.x;
      const y = b.pos.y;
      const wasAbove = prevY < rig.sensor.yTop;
      const nowBelow = y > rig.sensor.yBottom;
      const within = x >= rig.sensor.x1 && x <= rig.sensor.x2;
      if (wasAbove && nowBelow && within) {
        playground.score += 1;
      }
    }
  }

  function drawHoop() {
    if (!ctx) return;
    if (!CONFIG.physics.hoopEnabled) return;
    const rig = getHoopRig();

    ctx.save();

    // Backboard
    const backboard = rig.colliders.find((c) => c.kind === 'box') as
      | Extract<StaticCollider, { kind: 'box' }>
      | undefined;
    if (backboard) {
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 3 / camera2d.zoom;
      ctx.beginPath();
      ctx.rect(
        backboard.pos.x - backboard.size.w / 2,
        backboard.pos.y - backboard.size.h / 2,
        backboard.size.w,
        backboard.size.h
      );
      ctx.fill();
      ctx.stroke();
    }

    // Rim
    ctx.strokeStyle = 'rgba(255, 140, 40, 0.95)';
    ctx.lineWidth = 6 / camera2d.zoom;
    ctx.beginPath();
    ctx.arc(rig.rimCenter.x, rig.rimCenter.y, rig.rimR, Math.PI * 0.06, Math.PI - Math.PI * 0.06);
    ctx.stroke();

    // Simple net lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = 1.5 / camera2d.zoom;
    const netTopY = rig.sensor.yTop;
    const netBottomY = rig.sensor.yBottom;
    for (let i = -3; i <= 3; i++) {
      const x = rig.rimCenter.x + (i / 3) * (rig.rimR * 0.65);
      ctx.beginPath();
      ctx.moveTo(x, netTopY);
      ctx.lineTo(x + (i % 2 === 0 ? 1 : -1) * (rig.rimR * 0.08), netBottomY);
      ctx.stroke();
    }

    ctx.restore();
  }

  function resolveBodyCollisions() {
    const grabbed = grabbedIds();
    const n = bodies.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = bodies[i];
        const b = bodies[j];

        const grabbedBias = grabbed.has(a.id) || grabbed.has(b.id) ? 0.6 : 1.0;

        if (a.kind === 'circle' && b.kind === 'circle') {
          const ra = a.radius ?? 0;
          const rb = b.radius ?? 0;
          const dx = b.pos.x - a.pos.x;
          const dy = b.pos.y - a.pos.y;
          const d = Math.hypot(dx, dy);
          const minD = ra + rb;
          if (d > 0.0001 && d < minD) {
            const nx = dx / d;
            const ny = dy / d;
            const penetration = minD - d;

            const totalMass = a.mass + b.mass;
            const moveA = penetration * (b.mass / totalMass) * grabbedBias;
            const moveB = penetration * (a.mass / totalMass) * grabbedBias;
            a.pos.x -= nx * moveA;
            a.pos.y -= ny * moveA;
            b.pos.x += nx * moveB;
            b.pos.y += ny * moveB;

            const rvx = b.vel.x - a.vel.x;
            const rvy = b.vel.y - a.vel.y;
            const velAlongNormal = rvx * nx + rvy * ny;
            if (velAlongNormal < 0) {
              const e = Math.min(a.restitution, b.restitution);
              const jImp = (-(1 + e) * velAlongNormal) / (1 / a.mass + 1 / b.mass);
              const impX = jImp * nx * grabbedBias;
              const impY = jImp * ny * grabbedBias;
              a.vel.x -= impX / a.mass;
              a.vel.y -= impY / a.mass;
              b.vel.x += impX / b.mass;
              b.vel.y += impY / b.mass;

              const tx = -ny;
              const ty = nx;
              const vt = rvx * tx + rvy * ty;
              const mu = (a.friction + b.friction) * 0.5;
              const jt = -vt / (1 / a.mass + 1 / b.mass);
              const jtClamped = Math.max(-jImp * mu, Math.min(jImp * mu, jt));
              const fX = jtClamped * tx * grabbedBias;
              const fY = jtClamped * ty * grabbedBias;
              a.vel.x -= fX / a.mass;
              a.vel.y -= fY / a.mass;
              b.vel.x += fX / b.mass;
              b.vel.y += fY / b.mass;
            }
          }
          continue;
        }

        if (a.kind === 'box' && b.kind === 'box') {
          const aw = a.size!.w,
            ah = a.size!.h;
          const bw = b.size!.w,
            bh = b.size!.h;
          const ax1 = a.pos.x - aw / 2,
            ax2 = a.pos.x + aw / 2;
          const ay1 = a.pos.y - ah / 2,
            ay2 = a.pos.y + ah / 2;
          const bx1 = b.pos.x - bw / 2,
            bx2 = b.pos.x + bw / 2;
          const by1 = b.pos.y - bh / 2,
            by2 = b.pos.y + bh / 2;

          if (ax1 < bx2 && ax2 > bx1 && ay1 < by2 && ay2 > by1) {
            const overlapX = Math.min(ax2 - bx1, bx2 - ax1);
            const overlapY = Math.min(ay2 - by1, by2 - ay1);

            let nx = 0,
              ny = 0,
              penetration = 0;
            if (overlapX < overlapY) {
              penetration = overlapX;
              nx = a.pos.x < b.pos.x ? -1 : 1;
            } else {
              penetration = overlapY;
              ny = a.pos.y < b.pos.y ? -1 : 1;
            }

            const totalMass = a.mass + b.mass;
            const moveA = penetration * (b.mass / totalMass) * grabbedBias;
            const moveB = penetration * (a.mass / totalMass) * grabbedBias;
            a.pos.x += nx * moveA;
            a.pos.y += ny * moveA;
            b.pos.x -= nx * moveB;
            b.pos.y -= ny * moveB;

            const rvx = b.vel.x - a.vel.x;
            const rvy = b.vel.y - a.vel.y;
            const velAlongNormal = rvx * nx + rvy * ny;
            if (velAlongNormal < 0) {
              const e = Math.min(a.restitution, b.restitution);
              const jImp = (-(1 + e) * velAlongNormal) / (1 / a.mass + 1 / b.mass);
              const impX = jImp * nx * grabbedBias;
              const impY = jImp * ny * grabbedBias;
              a.vel.x -= impX / a.mass;
              a.vel.y -= impY / a.mass;
              b.vel.x += impX / b.mass;
              b.vel.y += impY / b.mass;

              const tx = -ny;
              const ty = nx;
              const vt = rvx * tx + rvy * ty;
              const mu = (a.friction + b.friction) * 0.5;
              const jt = -vt / (1 / a.mass + 1 / b.mass);
              const jtClamped = Math.max(-jImp * mu, Math.min(jImp * mu, jt));
              const fX = jtClamped * tx * grabbedBias;
              const fY = jtClamped * ty * grabbedBias;
              a.vel.x -= fX / a.mass;
              a.vel.y -= fY / a.mass;
              b.vel.x += fX / b.mass;
              b.vel.y += fY / b.mass;
            }
          }
          continue;
        }

        const circle = a.kind === 'circle' ? a : b.kind === 'circle' ? b : null;
        const box = a.kind === 'box' ? a : b.kind === 'box' ? b : null;
        if (circle && box) {
          const r = circle.radius ?? 0;
          const hw = box.size!.w / 2;
          const hh = box.size!.h / 2;

          const closestX = Math.max(box.pos.x - hw, Math.min(circle.pos.x, box.pos.x + hw));
          const closestY = Math.max(box.pos.y - hh, Math.min(circle.pos.y, box.pos.y + hh));

          const dx = circle.pos.x - closestX;
          const dy = circle.pos.y - closestY;
          const d2 = dx * dx + dy * dy;

          if (d2 < r * r) {
            const d = Math.sqrt(Math.max(d2, 1e-9));
            let nx = dx / d;
            let ny = dy / d;

            if (!isFinite(nx) || !isFinite(ny)) {
              const px = circle.pos.x - box.pos.x;
              const py = circle.pos.y - box.pos.y;
              if (Math.abs(px / hw) > Math.abs(py / hh)) {
                nx = px > 0 ? 1 : -1;
                ny = 0;
              } else {
                nx = 0;
                ny = py > 0 ? 1 : -1;
              }
            }

            const penetration = r - d;

            const totalMass = circle.mass + box.mass;
            const moveC = penetration * (box.mass / totalMass) * grabbedBias;
            const moveB = penetration * (circle.mass / totalMass) * grabbedBias;
            circle.pos.x += nx * moveC;
            circle.pos.y += ny * moveC;
            box.pos.x -= nx * moveB;
            box.pos.y -= ny * moveB;

            const rvx = box.vel.x - circle.vel.x;
            const rvy = box.vel.y - circle.vel.y;
            const velAlongNormal = rvx * nx + rvy * ny;
            if (velAlongNormal < 0) {
              const e = Math.min(circle.restitution, box.restitution);
              const jImp = (-(1 + e) * velAlongNormal) / (1 / circle.mass + 1 / box.mass);
              const impX = jImp * nx * grabbedBias;
              const impY = jImp * ny * grabbedBias;

              circle.vel.x -= impX / circle.mass;
              circle.vel.y -= impY / circle.mass;
              box.vel.x += impX / box.mass;
              box.vel.y += impY / box.mass;

              const tx = -ny;
              const ty = nx;
              const vt = rvx * tx + rvy * ty;
              const mu = (circle.friction + box.friction) * 0.5;
              const jt = -vt / (1 / circle.mass + 1 / box.mass);
              const jtClamped = Math.max(-jImp * mu, Math.min(jImp * mu, jt));
              const fX = jtClamped * tx * grabbedBias;
              const fY = jtClamped * ty * grabbedBias;
              circle.vel.x -= fX / circle.mass;
              circle.vel.y -= fY / circle.mass;
              box.vel.x += fX / box.mass;
              box.vel.y += fY / box.mass;
            }
          }
        }
      }
    }
  }

  function resolveHandCollisions(
    nowMs: number,
    grabController: Map<number, number>
  ) {
    if (!CONFIG.physics.handCollisions) return;

    // Flatten colliders for faster iteration.
    const colliders: Array<{ handIndex: number; pos: Point2; vel: Point2; r: number }> = [];
    for (const hs of handsState) {
      for (const c of hs.colliders) colliders.push(c);
    }
    if (colliders.length === 0) return;

    const eHand = CONFIG.physics.handRestitution;

    for (const b of bodies) {
      const controllerIdx = grabController.get(b.id);

      for (const c of colliders) {
        // Avoid the grabbed object jittering against the same hand that's controlling it.
        if (controllerIdx != null && controllerIdx === c.handIndex) continue;
        if (shouldIgnoreHandCollision(b.id, c.handIndex, nowMs)) continue;

        if (b.kind === 'circle') {
          const r = b.radius ?? 0;
          const dx = b.pos.x - c.pos.x;
          const dy = b.pos.y - c.pos.y;
          const d2 = dx * dx + dy * dy;
          const minD = r + c.r;
          if (d2 >= minD * minD) continue;

          const d = Math.sqrt(Math.max(d2, 1e-9));
          const nx = dx / d;
          const ny = dy / d;
          const penetration = minD - d;

          // Positional correction: move body out of hand
          b.pos.x += nx * penetration;
          b.pos.y += ny * penetration;

          // Velocity impulse relative to moving hand collider
          const rvx = b.vel.x - c.vel.x;
          const rvy = b.vel.y - c.vel.y;
          const velAlongNormal = rvx * nx + rvy * ny;
          if (velAlongNormal < 0) {
            const e = Math.min(eHand, b.restitution);
            // hand is infinite mass -> impulse directly reflects relative velocity
            const j = -(1 + e) * velAlongNormal;
            b.vel.x += j * nx;
            b.vel.y += j * ny;

            // Simple tangential friction
            const tx = -ny;
            const ty = nx;
            const vt = rvx * tx + rvy * ty;
            const mu = b.friction;
            b.vel.x -= vt * tx * mu * 0.15;
            b.vel.y -= vt * ty * mu * 0.15;
          }
          continue;
        }

        // box vs hand-circle
        const hw = b.size!.w / 2;
        const hh = b.size!.h / 2;
        const closestX = Math.max(b.pos.x - hw, Math.min(c.pos.x, b.pos.x + hw));
        const closestY = Math.max(b.pos.y - hh, Math.min(c.pos.y, b.pos.y + hh));
        const dx = c.pos.x - closestX;
        const dy = c.pos.y - closestY;
        const d2 = dx * dx + dy * dy;
        if (d2 >= c.r * c.r) continue;

        const d = Math.sqrt(Math.max(d2, 1e-9));
        let nx = -dx / d;
        let ny = -dy / d;

        if (!isFinite(nx) || !isFinite(ny)) {
          // Fallback normal from box center
          const px = b.pos.x - c.pos.x;
          const py = b.pos.y - c.pos.y;
          if (Math.abs(px / hw) > Math.abs(py / hh)) {
            nx = px > 0 ? 1 : -1;
            ny = 0;
          } else {
            nx = 0;
            ny = py > 0 ? 1 : -1;
          }
        }

        const penetration = c.r - d;
        b.pos.x += nx * penetration;
        b.pos.y += ny * penetration;

        const rvx = b.vel.x - c.vel.x;
        const rvy = b.vel.y - c.vel.y;
        const velAlongNormal = rvx * nx + rvy * ny;
        if (velAlongNormal < 0) {
          const e = Math.min(eHand, b.restitution);
          const j = -(1 + e) * velAlongNormal;
          b.vel.x += j * nx;
          b.vel.y += j * ny;

          const tx = -ny;
          const ty = nx;
          const vt = rvx * tx + rvy * ty;
          const mu = b.friction;
          b.vel.x -= vt * tx * mu * 0.15;
          b.vel.y -= vt * ty * mu * 0.15;
        }
      }
    }
  }

  function stepPhysics(dt: number) {
    const fy = floorY();
    const leftWorldX = screenToWorld({ x: 0, y: 0 }).x;
    const rightWorldX = screenToWorld({ x: window.innerWidth, y: 0 }).x;
    const nowMs = performance.now();

    // Map of currently grabbed body -> controlling hand index (pinch grab only)
    const grabController = new Map<number, number>();
    for (let hi = 0; hi < handsState.length; hi++) {
      const hs = handsState[hi];
      if (hs.grabActiveId != null && hs.pointer.pinch != null) {
        // If somehow multiple hands point at same id, keep the first.
        if (!grabController.has(hs.grabActiveId)) grabController.set(hs.grabActiveId, hi);
      }
    }

    // 1) Forces + integration (including grabbed body)
    for (const b of bodies) {
      const controllerIdx = grabController.get(b.id);
      const isGrabbed = controllerIdx != null;

      if (isGrabbed) {
        const hs = handsState[controllerIdx];
        const target = screenToWorld(hs.pointer.pinch!);
        const errX = target.x - b.pos.x;
        const errY = target.y - b.pos.y;

        // Desired velocity = hand velocity + correction toward target.
        const desired = clampVecMag(
          {
            x: hs.pointer.pinchVel.x + errX * CONFIG.physics.grabPosGain,
            y: hs.pointer.pinchVel.y + errY * CONFIG.physics.grabPosGain,
          },
          CONFIG.physics.maxGrabSpeed
        );

        const k = clamp(CONFIG.physics.grabFollow, 0, 1);
        b.vel.x = lerp(b.vel.x, desired.x, k);
        b.vel.y = lerp(b.vel.y, desired.y, k);
      } else {
        b.vel.y += settings.gravity * dt;
      }

      // Don't air-drag grabbed bodies; it makes them feel "behind".
      if (!isGrabbed) {
        const drag = Math.max(0, 1 - settings.airDrag * dt);
        b.vel.x *= drag;
        b.vel.y *= drag;
      }

      b.pos.x += b.vel.x * dt;
      b.pos.y += b.vel.y * dt;
    }

    // 1.5) Resolve collisions against kinematic hand colliders
    // Do a couple passes for stability; dt is used only for semantics, not integration here.
    resolveHandCollisions(nowMs, grabController);
    resolveHandCollisions(nowMs, grabController);

    // 1.75) Playground hoop collisions
    resolveHoopCollisions();

    // 2) Floor + walls constraints for all bodies (including grabbed)
    for (const b of bodies) {
      if (b.kind === 'circle') {
        const r = b.radius ?? 0;
        if (b.pos.y + r > fy) {
          b.pos.y = fy - r;
          if (b.vel.y > 0) b.vel.y = -b.vel.y * b.restitution;
          b.vel.x *= 1 - b.friction * 0.08;
        }

        if (b.pos.x - r < leftWorldX) {
          b.pos.x = leftWorldX + r;
          if (b.vel.x < 0) b.vel.x = -b.vel.x * 0.6;
        }
        if (b.pos.x + r > rightWorldX) {
          b.pos.x = rightWorldX - r;
          if (b.vel.x > 0) b.vel.x = -b.vel.x * 0.6;
        }
      } else {
        const halfH = b.size!.h / 2;
        if (b.pos.y + halfH > fy) {
          b.pos.y = fy - halfH;
          if (b.vel.y > 0) b.vel.y = -b.vel.y * b.restitution;
          b.vel.x *= 1 - b.friction * 0.08;
        }

        const halfW = b.size!.w / 2;
        if (b.pos.x - halfW < leftWorldX) {
          b.pos.x = leftWorldX + halfW;
          if (b.vel.x < 0) b.vel.x = -b.vel.x * 0.6;
        }
        if (b.pos.x + halfW > rightWorldX) {
          b.pos.x = rightWorldX - halfW;
          if (b.vel.x > 0) b.vel.x = -b.vel.x * 0.6;
        }
      }
    }

    // 3) Resolve body-body collisions (a couple passes helps stability)
    resolveBodyCollisions();
    resolveBodyCollisions();
  }

  function drawGlowCircle(p: Point2, r: number, color: string) {
    if (!ctx) return;
    ctx.save();
    ctx.globalAlpha = settings.glowAlpha;
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 2.2);
    g.addColorStop(0, color);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawHand(pts: Point2[], pinching: boolean) {
    if (!ctx) return;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = `rgba(102, 255, 255, ${settings.boneAlpha})`;
    ctx.lineWidth = settings.boneWidth;

    for (const [a, b] of Array.from(HAND_CONNECTIONS) as Array<[number, number]>) {
      const pa = pts[a];
      const pb = pts[b];
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
    ctx.restore();

    const palmIds = [0, 5, 9, 13, 17];
    ctx.save();
    ctx.fillStyle = `rgba(0, 255, 255, ${settings.palmAlpha})`;
    ctx.beginPath();
    ctx.moveTo(pts[palmIds[0]].x, pts[palmIds[0]].y);
    for (let i = 1; i < palmIds.length; i++) ctx.lineTo(pts[palmIds[i]].x, pts[palmIds[i]].y);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const isTip = i === 4 || i === 8;
      const pinchColor = 'rgba(255, 51, 85, 1)';
      const baseColor = 'rgba(0, 255, 255, 1)';

      const r = isTip && pinching ? settings.jointRadius * 1.35 : settings.jointRadius;
      const c = isTip && pinching ? pinchColor : baseColor;

      drawGlowCircle(p, r, c);

      ctx.save();
      ctx.fillStyle = c;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function drawBody(b: PhysicsBody) {
    if (!ctx) return;
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.fillStyle = b.color;
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 2 / camera2d.zoom;

    if (b.kind === 'circle') {
      const r = b.radius ?? 0;
      ctx.beginPath();
      ctx.arc(b.pos.x, b.pos.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    } else {
      const w = b.size!.w;
      const h = b.size!.h;
      ctx.beginPath();
      ctx.rect(b.pos.x - w / 2, b.pos.y - h / 2, w, h);
      ctx.fill();
      ctx.stroke();
    }

    ctx.globalAlpha = 0.22;
    ctx.strokeStyle = b.color;
    ctx.lineWidth = 10 / camera2d.zoom;
    if (b.kind === 'circle') {
      const r = (b.radius ?? 0) + 6;
      ctx.beginPath();
      ctx.arc(b.pos.x, b.pos.y, r, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      const w = b.size!.w + 10;
      const h = b.size!.h + 10;
      ctx.strokeRect(b.pos.x - w / 2, b.pos.y - h / 2, w, h);
    }

    ctx.restore();
  }

  function drawFloor() {
    if (!ctx) return;
    const fy = floorY();

    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.06)';

    const leftWorldX = screenToWorld({ x: 0, y: 0 }).x;
    const rightWorldX = screenToWorld({ x: window.innerWidth, y: 0 }).x;
    const bottomWorldY = screenToWorld({ x: 0, y: window.innerHeight }).y;

    ctx.fillRect(leftWorldX, fy, rightWorldX - leftWorldX, bottomWorldY - fy);

    ctx.strokeStyle = 'rgba(0,255,255,0.35)';
    ctx.lineWidth = 2 / camera2d.zoom;
    ctx.beginPath();
    ctx.moveTo(leftWorldX, fy);
    ctx.lineTo(rightWorldX, fy);
    ctx.stroke();

    ctx.globalAlpha = 0.15;
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1 / camera2d.zoom;
    const step = 60;
    const stepWorld = step / camera2d.zoom;
    const start = Math.floor(leftWorldX / stepWorld) * stepWorld;
    for (let x = start; x < rightWorldX; x += stepWorld) {
      ctx.beginPath();
      ctx.moveTo(x, fy);
      ctx.lineTo(x, bottomWorldY);
      ctx.stroke();
    }

    ctx.restore();
  }

  function tryStartGrab() {
    // Legacy single-hand signature kept unused.
  }

  function tryStartGrabForHand(handIndex: number) {
    const hs = handsState[handIndex];
    if (!hs.pointer.pinch || hs.grabActiveId != null) return;

    // Fresh throw history per grab attempt
    hs.throwSamples = [];

    const pinchW = screenToWorld(hs.pointer.pinch);

    let best: { b: PhysicsBody; d: number } | null = null;
    for (const b of bodies) {
      if (isBodyGrabbedByOtherHand(b.id, handIndex)) continue;
      const d = bodyPickDistance(b, pinchW);
      if (d < settings.grabRadius / camera2d.zoom && (!best || d < best.d)) {
        best = { b, d };
      }
    }

    if (best) hs.grabActiveId = best.b.id;
  }

  function endGrab() {
    // Legacy single-hand signature kept unused.
  }

  function endGrabForHand(handIndex: number) {
    const hs = handsState[handIndex];
    const b = getBodyById(hs.grabActiveId);
    hs.grabActiveId = null;

    if (b) {
      const baseVel =
        hs.throwSamples.length >= CONFIG.physics.throwMinSamples
          ? robustAverageVelocity(hs.throwSamples, CONFIG.physics.maxThrowSpeed)
          : hs.pointer.pinchVel;

      const throwVel = clampVecMag(
        { x: baseVel.x * settings.throwScale, y: baseVel.y * settings.throwScale },
        CONFIG.physics.maxThrowSpeed
      );

      const k = clamp(CONFIG.physics.throwBlend, 0, 1);
      b.vel.x = lerp(b.vel.x, throwVel.x, k);
      b.vel.y = lerp(b.vel.y, throwVel.y, k);
      const clamped = clampVecMag(b.vel, CONFIG.physics.maxThrowSpeed);
      b.vel.x = clamped.x;
      b.vel.y = clamped.y;

      // Prevent immediate post-release self-collision while the hand is still overlapping.
      setHandCollisionIgnore(b.id, handIndex, performance.now() + CONFIG.physics.throwHandIgnoreMs);
    }

    hs.throwSamples = [];
  }

  const settings: {
    zoom: number;
    gravity: number;
    airDrag: number;
    floorHeight: number;
    grabRadius: number;
    grabStiffness: number;
    grabDamping: number;
    throwScale: number;
    restitution: number;
    friction: number;
    jointRadius: number;
    boneWidth: number;
    boneAlpha: number;
    palmAlpha: number;
    glowAlpha: number;
  } = {
    zoom: 1,
    gravity: CONFIG.physics.gravity,
    airDrag: CONFIG.physics.airDrag,
    floorHeight: CONFIG.physics.floorHeight,
    grabRadius: CONFIG.physics.grabRadius,
    grabStiffness: CONFIG.physics.grabStiffness,
    grabDamping: CONFIG.physics.grabDamping,
    throwScale: CONFIG.physics.throwScale,
    restitution: 0.45,
    friction: 0.55,

    jointRadius: CONFIG.jointRadius,
    boneWidth: CONFIG.boneWidth,
    boneAlpha: CONFIG.boneAlpha,
    palmAlpha: CONFIG.palmAlpha,
    glowAlpha: CONFIG.glowAlpha,
  };

  function tick(t = performance.now()) {
    if (!running) return;

    if (!lastT) lastT = t;
    let dt = (t - lastT) / 1000;
    lastT = t;
    dt = Math.min(CONFIG.physics.maxDt, Math.max(0, dt));

    const sub = CONFIG.physics.substeps;
    const sdt = dt / sub;
    for (let i = 0; i < sub; i++) stepPhysics(sdt);

    resetCanvasTransform();
    ctx!.clearRect(0, 0, window.innerWidth, window.innerHeight);

    applyCameraTransform();
    drawFloor();
    drawHoop();
    for (const b of bodies) drawBody(b);

    resetCanvasTransform();
    // Score overlay
    if (CONFIG.physics.hoopEnabled) {
      ctx!.save();
      ctx!.fillStyle = 'rgba(255,255,255,0.85)';
      ctx!.font = '600 16px Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial';
      ctx!.textAlign = 'center';
      ctx!.textBaseline = 'top';
      ctx!.fillText(`Score: ${playground.score}`, window.innerWidth / 2, 18);
      ctx!.restore();
    }

    for (const hs of handsState) {
      if (!hs.lastLandmarksPx) continue;
      drawHand(hs.lastLandmarksPx, hs.isPinching);
    }

    rafId = requestAnimationFrame(tick);
  }

  function initMediaPipe() {
    if (!video) return;

    const MP_HANDS_VERSION = '0.4.1675469240';

    hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MP_HANDS_VERSION}/${file}`,
    });

    hands.setOptions({
      maxNumHands: CONFIG.maxNumHands,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      selfieMode: false,
      // @ts-expect-error non-standard
      delegate: 'CPU',
    });

    hands.onResults((results: Results) => {
      if (!firstResultsEmitted) {
        firstResultsEmitted = true;
        onFirstResultsCb?.();
      }

      const lmsAll = results.multiHandLandmarks ?? [];

      for (let i = 0; i < handsState.length; i++) {
        const hs = handsState[i];
        const lms = lmsAll[i];

        if (lms) {
          const pts = lms.map(toScreenPoint);
          // Use RAW points for interaction to avoid added latency from smoothing.
          const thumbRaw = pts[4];
          const indexRaw = pts[8];
          hs.pinchPosPx = { x: (thumbRaw.x + indexRaw.x) / 2, y: (thumbRaw.y + indexRaw.y) / 2 };

          // Palm center in SCREEN px
          const palmIdsPx = [0, 5, 9, 13, 17];
          let palmXpx = 0;
          let palmYpx = 0;
          for (const pid of palmIdsPx) {
            palmXpx += pts[pid].x;
            palmYpx += pts[pid].y;
          }
          palmXpx /= palmIdsPx.length;
          palmYpx /= palmIdsPx.length;
          hs.palmPosPx = { x: palmXpx, y: palmYpx };

          const dPx = Math.hypot(thumbRaw.x - indexRaw.x, thumbRaw.y - indexRaw.y);
          const prevPinching = hs.isPinching;
          // Prefer normalized landmark distance (camera-size independent).
          // Also compute a screen-normalized distance as a fallback.
          const dNorm = Math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y);
          const minDim = Math.max(1, Math.min(window.innerWidth, window.innerHeight));
          const dScreenNorm = dPx / minDim;
          const d = Math.min(dNorm, dScreenNorm);
          const pinchNow = prevPinching ? d < CONFIG.pinchEnd : d < CONFIG.pinchStart;

          // Grasp/fist gesture is disabled: pinch is thumb+index only.
          const graspNow = false;

          // Pinch grabbing only (no grasp/palm grabbing).
          const activeGrabNow = pinchNow;
          const grabPosPx = pinchNow ? hs.pinchPosPx : null;
          hs.grabPosPx = grabPosPx;

          // Smooth rendering less while pinching so visuals match interaction.
          const renderSmoothing = activeGrabNow ? 0.15 : CONFIG.smoothing;
          hs.lastLandmarksPx = smoothPoints(hs.lastLandmarksPx, pts, renderSmoothing);
          const landmarksPx = hs.lastLandmarksPx;

          const now = performance.now();
          const dt = hs.pointer.lastPinchT > 0 ? (now - hs.pointer.lastPinchT) / 1000 : 0;
          hs.pointer.lastPinchT = now;

          hs.pointer.prevPinch = hs.pointer.pinch;
          hs.pointer.pinch = grabPosPx;

          // Compute pinch velocity in WORLD units to keep throw consistent across zoom,
          // and clamp/smooth to avoid spikes from irregular MediaPipe callback timing.
          const minDt = 1 / 120;
          const maxDt = 1 / 15;

          if (hs.pointer.prevPinch && hs.pointer.pinch && dt >= minDt && dt <= maxDt) {
            const a = screenToWorld(hs.pointer.prevPinch);
            const c = screenToWorld(hs.pointer.pinch);
            const inst = {
              x: (c.x - a.x) / dt,
              y: (c.y - a.y) / dt,
            };

            const instClamped = clampVecMag(inst, CONFIG.physics.maxThrowSpeed);
            const alpha = clamp(CONFIG.physics.throwSmoothing, 0, 1);
            hs.pointer.pinchVel.x = lerp(hs.pointer.pinchVel.x, instClamped.x, alpha);
            hs.pointer.pinchVel.y = lerp(hs.pointer.pinchVel.y, instClamped.y, alpha);

            // Collect samples for release throw while pinching.
            if (grabPosPx) {
              const nowMs = performance.now();
              hs.throwSamples.push({ v: instClamped, t: nowMs });
              const windowMs = CONFIG.physics.throwWindowMs;
              const cutoff = nowMs - windowMs;
              // Keep only recent samples; preserve order.
              hs.throwSamples = hs.throwSamples.filter((s) => s.t >= cutoff);
            }
          } else {
            hs.pointer.pinchVel.x = lerp(hs.pointer.pinchVel.x, 0, 0.25);
            hs.pointer.pinchVel.y = lerp(hs.pointer.pinchVel.y, 0, 0.25);
          }

          // Build/update kinematic colliders (WORLD space).
          // Keep it simple: one big palm circle + a few knuckle points.
          const palmW = screenToWorld(hs.palmPosPx!);

          const pointIds = [5, 9, 13, 17, 0];
          const pointsW = pointIds.map((pid) => screenToWorld(pts[pid]));

          const palmR = CONFIG.physics.handPalmRadius / camera2d.zoom;
          const pointR = CONFIG.physics.handPointRadius / camera2d.zoom;

          const nextPos: Point2[] = [palmW, ...pointsW];
          const vel: Point2[] = [];

          if (dt >= minDt && dt <= maxDt && hs.prevColliderPos.length === nextPos.length) {
            for (let k = 0; k < nextPos.length; k++) {
              vel.push({
                x: (nextPos[k].x - hs.prevColliderPos[k].x) / dt,
                y: (nextPos[k].y - hs.prevColliderPos[k].y) / dt,
              });
            }
          } else {
            for (let k = 0; k < nextPos.length; k++) vel.push({ x: 0, y: 0 });
          }

          hs.prevColliderPos = nextPos;
          hs.colliders = [
            { handIndex: i, pos: nextPos[0], vel: clampVecMag(vel[0], CONFIG.physics.maxThrowSpeed), r: palmR },
            { handIndex: i, pos: nextPos[1], vel: clampVecMag(vel[1], CONFIG.physics.maxThrowSpeed), r: pointR },
            { handIndex: i, pos: nextPos[2], vel: clampVecMag(vel[2], CONFIG.physics.maxThrowSpeed), r: pointR },
            { handIndex: i, pos: nextPos[3], vel: clampVecMag(vel[3], CONFIG.physics.maxThrowSpeed), r: pointR },
            { handIndex: i, pos: nextPos[4], vel: clampVecMag(vel[4], CONFIG.physics.maxThrowSpeed), r: pointR },
            { handIndex: i, pos: nextPos[5], vel: clampVecMag(vel[5], CONFIG.physics.maxThrowSpeed), r: pointR },
          ];

          hs.isPinching = pinchNow;
          hs.isGrasping = false;

          const wasActiveGrab = prevPinching;
          if (activeGrabNow && !wasActiveGrab) {
            tryStartGrabForHand(i);
          } else if (!activeGrabNow && wasActiveGrab) {
            endGrabForHand(i);
          }
        } else {
          // This specific hand is not present in the current frame.
          hs.lastLandmarksPx = null;
          hs.pinchPosPx = null;
          hs.isPinching = false;
          hs.isGrasping = false;
          hs.palmPosPx = null;
          hs.grabPosPx = null;

          hs.pointer.pinch = null;
          hs.pointer.prevPinch = null;
          hs.pointer.pinchVel.x = 0;
          hs.pointer.pinchVel.y = 0;
          hs.pointer.lastPinchT = 0;

          hs.throwSamples = [];

          hs.colliders = [];
          hs.prevColliderPos = [];

          endGrabForHand(i);
        }
      }
    });

    camera = new Camera(video, {
      onFrame: async () => {
        if (!hands || !video) return;
        await hands.send({ image: video });
      },
      width: 640,
      height: 480,
    });

    camera.start();
  }

  function initVoiceControl() {
    const SpeechRecognitionImpl =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognitionImpl) return;

    const rec: SpeechRecognition = new SpeechRecognitionImpl();
    recognition = rec;

    rec.continuous = true;
    rec.lang = 'en-US';
    rec.interimResults = false;

    rec.onresult = (event: any) => {
      const last = event.results.length - 1;
      const command = String(event.results[last][0].transcript ?? '')
        .trim()
        .toLowerCase();

      if (command.includes('reset') || command.includes('clear')) {
        resetScene();
      } else if (command.includes('spawn') || command.includes('add')) {
        addRandomBody();
      }
    };

    try {
      rec.start();
    } catch {
      // ignore if already started
    }
  }

  function onWheel(e: WheelEvent) {
    const target = e.target as HTMLElement | null;
    if (target?.closest?.('#panel')) return;

    e.preventDefault();
    const delta = Math.sign(e.deltaY);
    camera2d.zoom = clamp(
      camera2d.zoom * (delta > 0 ? 0.95 : 1.05),
      0.5,
      1.6
    );
    settings.zoom = camera2d.zoom;
  }

  function onResize() {
    resizeOverlay();

    const fy = floorY();
    for (const b of bodies) {
      if (b.kind === 'circle') {
        const r = b.radius ?? 0;
        b.pos.y = Math.min(b.pos.y, fy - r);
      } else {
        b.pos.y = Math.min(b.pos.y, fy - b.size!.h / 2);
      }
    }
  }

  function mount(args: EngineMountArgs) {
    overlay = args.overlay;
    video = args.video;
    ctx = overlay.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');

    onFirstResultsCb = args.onFirstResults;
    firstResultsEmitted = false;

    resizeOverlay();

    // start empty
    resetScene();

    window.addEventListener('resize', onResize);
    window.addEventListener('wheel', onWheel, { passive: false });

    initVoiceControl();
    initMediaPipe();

    running = true;
    lastT = performance.now();
    rafId = requestAnimationFrame(tick);
  }

  function unmount() {
    running = false;

    window.removeEventListener('resize', onResize);
    window.removeEventListener('wheel', onWheel as any);

    if (rafId != null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }

    try {
      recognition?.stop();
    } catch {
      // ignore
    }
    recognition = null;

    if (video?.srcObject instanceof MediaStream) {
      for (const track of video.srcObject.getTracks()) track.stop();
    }

    camera = null;
    hands?.close?.();
    hands = null;

    overlay = null;
    ctx = null;
    video = null;
  }

  function applySettings(next: EngineSettings) {
    settings.zoom = next.zoom;
    camera2d.zoom = next.zoom;

    settings.gravity = next.gravity;
    settings.floorHeight = next.floorHeight;

    settings.restitution = next.bounciness;
    settings.friction = next.friction;

    for (const b of bodies) {
      b.restitution = settings.restitution;
      b.friction = settings.friction;
    }
  }

  return {
    mount,
    unmount,
    applySettings,
    addBody,
    addRandomBody,
    resetScene,
  };
}

