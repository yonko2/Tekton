import { Hands, Results, HAND_CONNECTIONS } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

type Point2 = { x: number; y: number };

type PhysicsBody = {
  id: number;
  kind: 'circle' | 'box';
  pos: Point2;
  vel: Point2;
  radius?: number;
  size?: { w: number; h: number };
  color: string;
  restitution: number;
  friction: number;
  mass: number;
};

const CONFIG = {
  maxNumHands: 1,

  // pinch in normalized landmark units
  pinchStart: 0.055,
  pinchEnd: 0.075,

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
  },
};

const overlay = document.getElementById('overlay') as HTMLCanvasElement;
const ctx = overlay.getContext('2d')!;
const video = document.getElementById('camera-feed') as HTMLVideoElement;

// UI (settings panel)
const ui = {
  panel: document.getElementById('panel') as HTMLDivElement,
  panelToggle: document.getElementById('panel-toggle') as HTMLButtonElement,
  panelContent: document.getElementById('panel-content') as HTMLDivElement,

  spawn: document.getElementById('spawn-one') as HTMLButtonElement,
  reset: document.getElementById('reset') as HTMLButtonElement,

  zoom: document.getElementById('zoom') as HTMLInputElement,
  gravity: document.getElementById('gravity') as HTMLInputElement,
  floor: document.getElementById('floor') as HTMLInputElement,
  bounciness: document.getElementById('bounciness') as HTMLInputElement,
  friction: document.getElementById('friction') as HTMLInputElement,
};

// 2D camera (world -> screen). Zoom < 1 = zoom out (bigger visible scene)
const camera2d = {
  zoom: 1,
  // keep origin at screen center in world space
  centerWorld: { x: 0, y: 0 },
};

function screenToWorld(p: Point2): Point2 {
  // Convert pixels into world units under current zoom
  return {
    x: (p.x - window.innerWidth / 2) / camera2d.zoom + camera2d.centerWorld.x,
    y: (p.y - window.innerHeight / 2) / camera2d.zoom + camera2d.centerWorld.y,
  };
}

function worldToScreen(p: Point2): Point2 {
  return {
    x: (p.x - camera2d.centerWorld.x) * camera2d.zoom + window.innerWidth / 2,
    y: (p.y - camera2d.centerWorld.y) * camera2d.zoom + window.innerHeight / 2,
  };
}

function applyCameraTransform() {
  // Draw everything (scene) in WORLD coordinates.
  // Important: include devicePixelRatio so world<->screen math stays consistent.
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
  // Back to CSS pixel coordinates
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ---------- Physics World ----------

let nextId = 1;
const bodies: PhysicsBody[] = [];

const pointer = {
  pinch: null as Point2 | null,
  prevPinch: null as Point2 | null,
  pinchVel: { x: 0, y: 0 },
  lastPinchT: 0,
};

const grab = {
  activeId: null as number | null,
};

function randColor() {
  const palette = ['#00ffdf', '#00ff88', '#66ffff', '#ff3355', '#ffd166', '#a78bfa'];
  return palette[Math.floor(Math.random() * palette.length)];
}

function addRandomBody() {
  const isCircle = Math.random() < 0.6;

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
      restitution: 0.45,
      friction: 0.5,
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
      restitution: 0.35,
      friction: 0.55,
      mass: 1.2,
    });
  }
}

function resetScene() {
  // Do not auto-spawn on reset; clear only (user can spawn via panel)
  bodies.length = 0;
  nextId = 1;
  // also drop any active grab
  grab.activeId = null;
}

// New helper: resolve collisions between bodies (position correction + impulse)
function resolveBodyCollisions() {
  // Simple, stable-ish impulse solver for small body counts.
  // Assumes axis-aligned boxes (no rotation).

  const n = bodies.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = bodies[i];
      const b = bodies[j];

      // Skip if either is being actively grabbed; still allow collisions but it can feel jittery.
      // We'll keep it enabled, but reduce response a bit.
      const grabbedBias = (a.id === grab.activeId || b.id === grab.activeId) ? 0.6 : 1.0;

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
          const penetration = (minD - d);

          // position correction
          const totalMass = a.mass + b.mass;
          const moveA = (penetration * (b.mass / totalMass)) * grabbedBias;
          const moveB = (penetration * (a.mass / totalMass)) * grabbedBias;
          a.pos.x -= nx * moveA;
          a.pos.y -= ny * moveA;
          b.pos.x += nx * moveB;
          b.pos.y += ny * moveB;

          // velocity impulse along normal
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

            // friction (tangent)
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
        const aw = a.size!.w, ah = a.size!.h;
        const bw = b.size!.w, bh = b.size!.h;
        const ax1 = a.pos.x - aw / 2, ax2 = a.pos.x + aw / 2;
        const ay1 = a.pos.y - ah / 2, ay2 = a.pos.y + ah / 2;
        const bx1 = b.pos.x - bw / 2, bx2 = b.pos.x + bw / 2;
        const by1 = b.pos.y - bh / 2, by2 = b.pos.y + bh / 2;

        if (ax1 < bx2 && ax2 > bx1 && ay1 < by2 && ay2 > by1) {
          const overlapX = Math.min(ax2 - bx1, bx2 - ax1);
          const overlapY = Math.min(ay2 - by1, by2 - ay1);

          // resolve along smallest axis
          let nx = 0, ny = 0, penetration = 0;
          if (overlapX < overlapY) {
            penetration = overlapX;
            nx = a.pos.x < b.pos.x ? -1 : 1;
          } else {
            penetration = overlapY;
            ny = a.pos.y < b.pos.y ? -1 : 1;
          }

          const totalMass = a.mass + b.mass;
          const moveA = (penetration * (b.mass / totalMass)) * grabbedBias;
          const moveB = (penetration * (a.mass / totalMass)) * grabbedBias;
          a.pos.x += nx * moveA;
          a.pos.y += ny * moveA;
          b.pos.x -= nx * moveB;
          b.pos.y -= ny * moveB;

          // normal impulse
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

            // friction
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

      // circle-box
      const circle = a.kind === 'circle' ? a : (b.kind === 'circle' ? b : null);
      const box = a.kind === 'box' ? a : (b.kind === 'box' ? b : null);
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
          // normal from box->circle
          let nx = dx / d;
          let ny = dy / d;

          // If circle center is inside box (rare due to clamp), push out on nearest axis
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

          const penetration = (r - d);

          // position correction
          const totalMass = circle.mass + box.mass;
          const moveC = (penetration * (box.mass / totalMass)) * grabbedBias;
          const moveB = (penetration * (circle.mass / totalMass)) * grabbedBias;
          circle.pos.x += nx * moveC;
          circle.pos.y += ny * moveC;
          box.pos.x -= nx * moveB;
          box.pos.y -= ny * moveB;

          // velocity impulse
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

            // friction
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

function stepPhysics(dt: number) {
  const fy = floorY();

  // basic integration
  for (const b of bodies) {
    if (b.id !== grab.activeId) {
      b.vel.y += CONFIG.physics.gravity * dt;
      const drag = Math.max(0, 1 - CONFIG.physics.airDrag * dt);
      b.vel.x *= drag;
      b.vel.y *= drag;

      b.pos.x += b.vel.x * dt;
      b.pos.y += b.vel.y * dt;
    }

    // Floor collision
    if (b.kind === 'circle') {
      const r = b.radius ?? 0;
      if (b.pos.y + r > fy) {
        b.pos.y = fy - r;
        if (b.vel.y > 0) b.vel.y = -b.vel.y * b.restitution;
        b.vel.x *= 1 - b.friction * 0.08;
      }
    } else {
      const halfH = b.size!.h / 2;
      if (b.pos.y + halfH > fy) {
        b.pos.y = fy - halfH;
        if (b.vel.y > 0) b.vel.y = -b.vel.y * b.restitution;
        b.vel.x *= 1 - b.friction * 0.08;
      }
    }

    // Screen bounds mapped to WORLD bounds
    const leftWorldX = screenToWorld({ x: 0, y: 0 }).x;
    const rightWorldX = screenToWorld({ x: window.innerWidth, y: 0 }).x;

    if (b.kind === 'circle') {
      const r = b.radius ?? 0;
      if (b.pos.x - r < leftWorldX) {
        b.pos.x = leftWorldX + r;
        if (b.vel.x < 0) b.vel.x = -b.vel.x * 0.6;
      }
      if (b.pos.x + r > rightWorldX) {
        b.pos.x = rightWorldX - r;
        if (b.vel.x > 0) b.vel.x = -b.vel.x * 0.6;
      }
    } else {
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

  // Grab spring (pinch dragging)
  const grabbed = getBodyById(grab.activeId);
  if (grabbed && pointer.pinch) {
    const pinchW = screenToWorld(pointer.pinch);
    const dx = pinchW.x - grabbed.pos.x;
    const dy = pinchW.y - grabbed.pos.y;

    const ax = (dx * CONFIG.physics.grabStiffness - grabbed.vel.x * CONFIG.physics.grabDamping) / grabbed.mass;
    const ay = (dy * CONFIG.physics.grabStiffness - grabbed.vel.y * CONFIG.physics.grabDamping) / grabbed.mass;

    grabbed.vel.x += ax * dt;
    grabbed.vel.y += ay * dt;

    grabbed.pos.x += grabbed.vel.x * dt;
    grabbed.pos.y += grabbed.vel.y * dt;
  }

  // After integration, resolve body-body collisions
  // A couple passes helps reduce overlap explosions in a simple solver.
  resolveBodyCollisions();
  resolveBodyCollisions();
}

function floorY() {
  // floor in WORLD coordinates: bottom of visible area minus floor height in *screen pixels* converted to world
  const bottomWorldY = screenToWorld({ x: 0, y: window.innerHeight }).y;
  const floorHWorld = CONFIG.physics.floorHeight / camera2d.zoom;
  return bottomWorldY - floorHWorld;
}

// ---------- Hand tracking + rendering ----------

let lastLandmarksPx: Point2[] | null = null;
let isPinching = false;
let pinchPosPx: Point2 | null = null;

function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.floor(window.innerWidth * dpr);
  overlay.height = Math.floor(window.innerHeight * dpr);
  overlay.style.width = `${window.innerWidth}px`;
  overlay.style.height = `${window.innerHeight}px`;

  // First match device pixels; draw loop will apply camera transform afterwards
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function toScreenPoint(lm: any): Point2 {
  // keep hand in SCREEN coordinates (for stable MediaPipe mapping)
  return {
    x: (1 - lm.x) * window.innerWidth,
    y: lm.y * window.innerHeight,
  };
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

function drawGlowCircle(p: Point2, r: number, color: string) {
  ctx.save();
  ctx.globalAlpha = CONFIG.glowAlpha;
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
  // Bones
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = `rgba(102, 255, 255, ${CONFIG.boneAlpha})`;
  ctx.lineWidth = CONFIG.boneWidth;

  for (const [a, b] of Array.from(HAND_CONNECTIONS) as Array<[number, number]>) {
    const pa = pts[a];
    const pb = pts[b];
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  ctx.restore();

  // Palm fill (wrist + MCPs)
  const palmIds = [0, 5, 9, 13, 17];
  ctx.save();
  ctx.fillStyle = `rgba(0, 255, 255, ${CONFIG.palmAlpha})`;
  ctx.beginPath();
  ctx.moveTo(pts[palmIds[0]].x, pts[palmIds[0]].y);
  for (let i = 1; i < palmIds.length; i++) ctx.lineTo(pts[palmIds[i]].x, pts[palmIds[i]].y);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Joints
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const isTip = i === 4 || i === 8;
    const pinchColor = 'rgba(255, 51, 85, 1)';
    const baseColor = 'rgba(0, 255, 255, 1)';

    const r = isTip && pinching ? CONFIG.jointRadius * 1.35 : CONFIG.jointRadius;
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
  // Draw in WORLD space (camera transform already applied)
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

  // subtle glow
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
  const fy = floorY();

  // floor plane
  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.06)';

  const leftWorldX = screenToWorld({ x: 0, y: 0 }).x;
  const rightWorldX = screenToWorld({ x: window.innerWidth, y: 0 }).x;
  const bottomWorldY = screenToWorld({ x: 0, y: window.innerHeight }).y;

  ctx.fillRect(leftWorldX, fy, rightWorldX - leftWorldX, bottomWorldY - fy);

  // floor line
  ctx.strokeStyle = 'rgba(0,255,255,0.35)';
  ctx.lineWidth = 2 / camera2d.zoom;
  ctx.beginPath();
  ctx.moveTo(leftWorldX, fy);
  ctx.lineTo(rightWorldX, fy);
  ctx.stroke();

  // grid hint
  ctx.globalAlpha = 0.15;
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 1 / camera2d.zoom;
  const step = 60; // in screen px
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

let lastT = performance.now();
function tick(t = performance.now()) {
  let dt = (t - lastT) / 1000;
  lastT = t;
  dt = Math.min(CONFIG.physics.maxDt, Math.max(0, dt));

  const sub = CONFIG.physics.substeps;
  const sdt = dt / sub;
  for (let i = 0; i < sub; i++) stepPhysics(sdt);

  // Clear in SCREEN space
  resetCanvasTransform();
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  // Draw scene in WORLD space
  applyCameraTransform();
  drawFloor();
  for (const b of bodies) drawBody(b);

  // Draw hand in SCREEN space (not affected by zoom)
  resetCanvasTransform();
  if (lastLandmarksPx) {
    drawHand(lastLandmarksPx, isPinching);

    if (pinchPosPx) {
      ctx.save();
      ctx.strokeStyle = isPinching ? 'rgba(255, 51, 85, 0.95)' : 'rgba(0, 255, 255, 0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pinchPosPx.x, pinchPosPx.y, 14, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  requestAnimationFrame(tick);
}

function initMediaPipe() {
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: CONFIG.maxNumHands,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results: Results) => {
    const loader = document.getElementById('loader');
    if (loader) loader.style.opacity = '0';

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const lms = results.multiHandLandmarks[0];
      const pts = lms.map(toScreenPoint);
      lastLandmarksPx = smoothPoints(lastLandmarksPx, pts, CONFIG.smoothing);

      const thumb = lastLandmarksPx[4];
      const index = lastLandmarksPx[8];
      pinchPosPx = { x: (thumb.x + index.x) / 2, y: (thumb.y + index.y) / 2 };

      // pinch detection in landmark space (normalized)
      const dNorm = Math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y);
      const prevPinching = isPinching;
      const pinchNow = prevPinching ? dNorm < CONFIG.pinchEnd : dNorm < CONFIG.pinchStart;

      // track pinch velocity in pixels using real dt between frames
      const now = performance.now();
      const dt = pointer.lastPinchT > 0 ? (now - pointer.lastPinchT) / 1000 : 0;
      pointer.lastPinchT = now;

      // Always keep pointer positions updated (even if not pinching) so release has a sane velocity.
      pointer.prevPinch = pointer.pinch;
      pointer.pinch = pinchPosPx;

      if (pointer.prevPinch && pointer.pinch && dt > 0.0001) {
        pointer.pinchVel.x = (pointer.pinch.x - pointer.prevPinch.x) / dt;
        pointer.pinchVel.y = (pointer.pinch.y - pointer.prevPinch.y) / dt;
      } else {
        pointer.pinchVel.x = 0;
        pointer.pinchVel.y = 0;
      }

      // state transitions (use prevPinching, then assign isPinching)
      if (pinchNow && !prevPinching) {
        isPinching = true;
        tryStartGrab();
      } else if (!pinchNow && prevPinching) {
        isPinching = false;
        endGrab();
      } else {
        isPinching = pinchNow;
      }

    } else {
      lastLandmarksPx = null;
      pinchPosPx = null;
      isPinching = false;

      pointer.pinch = null;
      pointer.prevPinch = null;
      pointer.pinchVel.x = 0;
      pointer.pinchVel.y = 0;
      pointer.lastPinchT = 0;

      endGrab();
    }
  });

  const cam = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480,
  });

  cam.start();
}

function initVoiceControl() {
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.lang = 'en-US';
  recognition.interimResults = false;

  recognition.onresult = (event: any) => {
    const last = event.results.length - 1;
    const command = event.results[last][0].transcript.trim().toLowerCase();

    if (command.includes('reset') || command.includes('clear')) {
      resetScene();
    } else if (command.includes('spawn') || command.includes('add')) {
      addRandomBody();
    }
  };

  recognition.start();
}

function setupPanel() {
  // Toggle
  ui.panelToggle?.addEventListener('click', () => {
    const isHidden = ui.panelContent.style.display === 'none';
    ui.panelContent.style.display = isHidden ? 'grid' : 'none';
    ui.panelToggle.textContent = isHidden ? 'Hide' : 'Show';
  });

  // Buttons
  ui.spawn?.addEventListener('click', () => addRandomBody());
  ui.reset?.addEventListener('click', () => resetScene());

  // Sliders
  const syncFromConfig = () => {
    ui.zoom.value = String(camera2d.zoom);
    ui.gravity.value = String(CONFIG.physics.gravity);
    ui.floor.value = String(CONFIG.physics.floorHeight);
    ui.bounciness.value = String(0.45);
    ui.friction.value = String(0.55);
  };

  syncFromConfig();

  ui.zoom?.addEventListener('input', () => {
    camera2d.zoom = Number(ui.zoom.value);
  });

  ui.gravity?.addEventListener('input', () => {
    CONFIG.physics.gravity = Number(ui.gravity.value);
  });

  ui.floor?.addEventListener('input', () => {
    CONFIG.physics.floorHeight = Number(ui.floor.value);
  });

  ui.bounciness?.addEventListener('input', () => {
    const v = Number(ui.bounciness.value);
    for (const b of bodies) b.restitution = v;
  });

  ui.friction?.addEventListener('input', () => {
    const v = Number(ui.friction.value);
    for (const b of bodies) b.friction = v;
  });

  // Mouse wheel zoom - overlay has pointer-events: none, so listen on window.
  window.addEventListener('wheel', (e) => {
    // If user is scrolling inside the panel, don't steal the wheel.
    const target = e.target as HTMLElement | null;
    if (target && target.closest && target.closest('#panel')) return;

    e.preventDefault();
    const delta = Math.sign(e.deltaY);
    camera2d.zoom = Math.min(1.6, Math.max(0.5, camera2d.zoom * (delta > 0 ? 0.95 : 1.05)));
    ui.zoom.value = String(camera2d.zoom);
  }, { passive: false });
}

function getBodyById(id: number | null) {
  if (id == null) return null;
  return bodies.find(b => b.id === id) ?? null;
}

function tryStartGrab() {
  // pinch is in SCREEN space; convert to WORLD
  if (!pointer.pinch || grab.activeId != null) return;

  const pinchW = screenToWorld(pointer.pinch);

  let best: { b: PhysicsBody; d: number } | null = null;
  for (const b of bodies) {
    const d = bodyPickDistance(b, pinchW);
    if (d < CONFIG.physics.grabRadius / camera2d.zoom && (!best || d < best.d)) {
      best = { b, d };
    }
  }

  if (best) grab.activeId = best.b.id;
}

function endGrab() {
  const b = getBodyById(grab.activeId);
  grab.activeId = null;

  // On release: throw with pinch velocity (pinchVel is in screen px/s). Convert to world units.
  if (b) {
    const scale = (CONFIG.physics.throwScale / camera2d.zoom);
    b.vel.x += pointer.pinchVel.x * scale;
    b.vel.y += pointer.pinchVel.y * scale;
  }
}

function init() {
  resizeOverlay();
  window.addEventListener('resize', () => {
    resizeOverlay();
    // keep objects above the new floor if the window shrinks
    const fy = floorY();
    for (const b of bodies) {
      if (b.kind === 'circle') {
        const r = b.radius ?? 0;
        b.pos.y = Math.min(b.pos.y, fy - r);
      } else {
        b.pos.y = Math.min(b.pos.y, fy - b.size!.h / 2);
      }
    }
  });

  // Hide the old 3D container if it exists
  const container = document.getElementById('canvas-container');
  if (container) container.style.display = 'none';

  // Start empty; spawning is controlled by the panel.
  resetScene();

  initVoiceControl();
  initMediaPipe();
  requestAnimationFrame(tick);
  setupPanel();

  // Initialize zoom from slider default
  camera2d.zoom = Number(ui.zoom?.value ?? 1);
}

function bodyPickDistance(b: PhysicsBody, p: Point2): number {
  if (b.kind === 'circle') {
    return dist(b.pos, p) - (b.radius ?? 0);
  }

  // distance to AABB (axis-aligned box)
  const w = b.size!.w;
  const h = b.size!.h;
  const dx = Math.max(Math.abs(p.x - b.pos.x) - w / 2, 0);
  const dy = Math.max(Math.abs(p.y - b.pos.y) - h / 2, 0);
  return Math.hypot(dx, dy);
}

init();

