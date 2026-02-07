import { useRef, useCallback } from 'react';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import {
  detectGesture,
  getPointerScreenPosition,
  getPinchMidpoint,
  getHandScale,
  getHandRollAngle,
  screenToWorld,
  screenToWorldAtDistance,
  isPinching,
  resetPinchState,
  resetPositionSmoothing,
} from '@/engine/gestures';
import { VelocityTracker, constrainToGround } from '@/engine/physics';
import { grabState, cameraZoomState } from '@/engine/grabStore';
import type { HandData, Vector3Tuple, GestureType } from '@/types';
import type { RapierRigidBody } from '@react-three/rapier';

export const rigidBodyRefs = new Map<string, React.RefObject<RapierRigidBody | null>>();

// ── Camera orbit angles ──────────────────────────────────────
let orbitTheta = 0;
let orbitPhi = Math.PI / 4;
const ORBIT_SENSITIVITY = 3;
const MIN_ZOOM = 5;
const MAX_ZOOM = 30;
const ZOOM_SENSITIVITY = 1.4; // how aggressively zoom responds to hand scale
const DEPTH_SENSITIVITY = 1.2; // how aggressively depth responds to hand scale
const MIN_GRAB_DIST = 2; // closest the object can be to the camera
const MAX_GRAB_DIST = 25; // farthest
const ORBIT_DEADZONE = 0.003; // ignore tiny screen-space deltas for camera orbit
const SCALE_DEADZONE = 0.03; // ignore tiny hand-scale ratio changes
const TWIST_SENSITIVITY = 5.0; // multiplier for twist → rotation

function sphericalToCartesian(theta: number, phi: number, r: number): Vector3Tuple {
  return [
    r * Math.sin(phi) * Math.sin(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.cos(theta),
  ];
}

// ── Proximity hit-test ───────────────────────────────────────
const GRAB_SCREEN_RADIUS = 0.12;

function findNearestObject(
  screenPos: { x: number; y: number },
  camera: THREE.Camera,
  objectIds: string[],
): { id: string; worldPos: THREE.Vector3 } | null {
  let bestId: string | null = null;
  let bestDist = GRAB_SCREEN_RADIUS;
  let bestWorld = new THREE.Vector3();
  const projected = new THREE.Vector3();

  for (const id of objectIds) {
    const rbRef = rigidBodyRefs.get(id);
    if (!rbRef?.current) continue;

    const t = rbRef.current.translation();
    projected.set(t.x, t.y, t.z).project(camera);

    const sx = (projected.x + 1) / 2;
    const sy = (1 - projected.y) / 2;
    const dist = Math.hypot(sx - screenPos.x, sy - screenPos.y);

    if (dist < bestDist) {
      bestDist = dist;
      bestId = id;
      bestWorld = new THREE.Vector3(t.x, t.y, t.z);
    }
  }

  return bestId ? { id: bestId, worldPos: bestWorld } : null;
}

// ──────────────────────────────────────────────────────────────
export function useGestureRecognition() {
  const { state, setGesture, setPointer, setCamera, selectObject } = useSandbox();

  const velocityTracker = useRef(new VelocityTracker());
  const lastScreenPos = useRef<{ x: number; y: number } | null>(null);
  const grabOffsetRef = useRef<Vector3Tuple>([0, 0, 0]);
  const wasPinching = useRef(false);
  const modeRef = useRef<'idle' | 'grab' | 'camera'>('idle');

  // Hand-scale tracking for zoom / depth
  const initialHandScaleRef = useRef(0);
  const initialZoomRadiusRef = useRef(cameraZoomState.radius);

  // Distance from camera to grabbed object (for ray-based positioning)
  const grabCameraDistRef = useRef(0);
  const initialGrabDistRef = useRef(0);

  // Twist / rotation tracking
  const initialRollAngleRef = useRef(0);

  const objectIdsRef = useRef<string[]>([]);
  objectIdsRef.current = state.objects.map((o) => o.id);

  // Keep latest objects in a ref for reading scale on grab
  const objectsRef = useRef(state.objects);
  objectsRef.current = state.objects;

  // ── Main per-frame processor ───────────────────────────────
  const processHands = useCallback(
    (hands: HandData[], camera: THREE.Camera, _scene: THREE.Scene) => {
      // ── No hands ────────────────────────────────────────────
      if (hands.length === 0) {
        resetPinchState();
        resetPositionSmoothing();
        setGesture({ currentGesture: 'none', isPinching: false, pointerPosition: null, screenPosition: null });
        setPointer({ visible: false, mode: 'idle', hoveredObjectId: null });

        if (modeRef.current === 'grab' && grabState.objectId) {
          doRelease();
        }
        modeRef.current = 'idle';
        wasPinching.current = false;
        return;
      }

      const hand = hands[0];
      const pinching = isPinching(hand.landmarks);
      const gesture: GestureType = detectGesture(hand.landmarks, pinching);
      const handScale = getHandScale(hand.landmarks);
      const rollAngle = getHandRollAngle(hand.landmarks);

      const screenPos = pinching
        ? getPinchMidpoint(hand.landmarks)
        : getPointerScreenPosition(hand.landmarks);

      const worldPos = screenToWorld(screenPos, camera, 0.5);
      const constrained = constrainToGround(worldPos);

      setGesture({
        currentGesture: gesture,
        isPinching: pinching,
        pointerPosition: constrained,
        screenPosition: screenPos,
      });

      // ── Pinch just started ──────────────────────────────────
      if (pinching && !wasPinching.current) {
        initialHandScaleRef.current = handScale;

        const hit = findNearestObject(screenPos, camera, objectIdsRef.current);
        if (hit) {
          modeRef.current = 'grab';
          velocityTracker.current.reset();

          // Store distance from camera to the object along the view ray.
          // This lets the object follow the hand in the camera's view plane.
          const camPos = new THREE.Vector3();
          camera.getWorldPosition(camPos);
          grabCameraDistRef.current = camPos.distanceTo(hit.worldPos);

          // Store initial grab distance for depth modulation
          initialGrabDistRef.current = grabCameraDistRef.current;

          // Offset between the object centre and where the ray hit,
          // so the object doesn't snap to the pinch point.
          const rayWorldPos = screenToWorldAtDistance(screenPos, camera, grabCameraDistRef.current);
          grabOffsetRef.current = [
            hit.worldPos.x - rayWorldPos[0],
            hit.worldPos.y - rayWorldPos[1],
            hit.worldPos.z - rayWorldPos[2],
          ];

          initialRollAngleRef.current = rollAngle;

          // Store camera forward direction as the twist rotation axis.
          // This makes hand-twist rotate the object in the user's view plane.
          const fwd = new THREE.Vector3();
          camera.getWorldDirection(fwd);
          fwd.normalize();

          grabState.objectId = hit.id;
          grabState.targetPosition = [hit.worldPos.x, hit.worldPos.y, hit.worldPos.z];
          grabState.twistAngle = 0;
          grabState.twistAxis = [fwd.x, fwd.y, fwd.z];
          grabState.pendingRelease = false;

          selectObject(hit.id);
          setPointer({ visible: true, mode: 'grabbing', grabbedObjectId: hit.id });
        } else {
          // Camera zoom mode
          modeRef.current = 'camera';
          lastScreenPos.current = screenPos;
          initialZoomRadiusRef.current = cameraZoomState.radius;
          setPointer({ visible: true, mode: 'camera', grabbedObjectId: null });
        }
      }

      // ── Ongoing pinch ───────────────────────────────────────
      if (pinching && wasPinching.current) {
        // Compute how much the hand moved toward / away from camera
        const scaleRatio = initialHandScaleRef.current > 0.001
          ? handScale / initialHandScaleRef.current
          : 1;

        if (modeRef.current === 'grab' && grabState.objectId) {
          // ── Adjust depth (distance from camera) based on hand scale ─
          // Hand closer → scaleRatio > 1 → smaller distance (pull toward camera)
          // Hand farther → scaleRatio < 1 → larger distance (push away)
          if (Math.abs(scaleRatio - 1) > SCALE_DEADZONE) {
            const depthRatio = Math.pow(1 / scaleRatio, DEPTH_SENSITIVITY);
            grabCameraDistRef.current = Math.max(
              MIN_GRAB_DIST,
              Math.min(MAX_GRAB_DIST, initialGrabDistRef.current * depthRatio),
            );
          }

          // ── Move the object along the camera ray ────────────
          // Project the screen position into 3D at the (possibly updated)
          // distance. Moving hand up/down/left/right moves the object in
          // the camera's view plane; hand closer/farther moves it in depth.
          const rayPos = screenToWorldAtDistance(screenPos, camera, grabCameraDistRef.current);
          const newPos: Vector3Tuple = [
            rayPos[0] + grabOffsetRef.current[0],
            rayPos[1] + grabOffsetRef.current[1],
            rayPos[2] + grabOffsetRef.current[2],
          ];

          // Keep within scene bounds; allow Y ≥ minY (floor level)
          const safePos = constrainToGround(newPos, 0.25);
          velocityTracker.current.record(safePos);
          grabState.targetPosition = safePos;

          // ── Rotate the object based on hand twist ──────────
          const twistDelta = rollAngle - initialRollAngleRef.current;
          grabState.twistAngle = twistDelta * TWIST_SENSITIVITY;

          setPointer({ position: safePos });

        } else if (modeRef.current === 'camera' && lastScreenPos.current) {
          // ── Orbit from lateral hand movement ────────────────
          const dx = screenPos.x - lastScreenPos.current.x;
          const dy = screenPos.y - lastScreenPos.current.y;

          // Dead zone: ignore tiny deltas caused by landmark noise
          if (Math.abs(dx) > ORBIT_DEADZONE || Math.abs(dy) > ORBIT_DEADZONE) {
            orbitTheta += dx * ORBIT_SENSITIVITY;
            orbitPhi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, orbitPhi + dy * ORBIT_SENSITIVITY));
            lastScreenPos.current = screenPos;
          }

          // ── Zoom from hand proximity ────────────────────────
          // Hand closer → scaleRatio > 1 → smaller radius (zoom in)
          // Hand farther → scaleRatio < 1 → larger radius (zoom out)
          if (Math.abs(scaleRatio - 1) > SCALE_DEADZONE) {
            const zoomRatio = Math.pow(1 / scaleRatio, ZOOM_SENSITIVITY);
            const newRadius = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, initialZoomRadiusRef.current * zoomRatio));
            cameraZoomState.radius = newRadius;
          }

          setCamera({ position: sphericalToCartesian(orbitTheta, orbitPhi, cameraZoomState.radius) });
        }
      }

      // ── Pinch released ──────────────────────────────────────
      if (!pinching && wasPinching.current) {
        if (modeRef.current === 'grab' && grabState.objectId) {
          doRelease();
        }
        modeRef.current = 'idle';
        setPointer({ mode: 'idle', grabbedObjectId: null });
      }

      // ── Pointing (no pinch) ─────────────────────────────────
      if (!pinching && gesture === 'point') {
        const hit = findNearestObject(screenPos, camera, objectIdsRef.current);
        setPointer({ visible: true, position: constrained, mode: 'hovering', hoveredObjectId: hit?.id ?? null });
      } else if (!pinching && gesture === 'none') {
        setPointer({ visible: false, mode: 'idle', hoveredObjectId: null });
      }

      wasPinching.current = pinching;
    },
    [setGesture, setPointer, setCamera, selectObject],
  );

  const doRelease = useCallback(() => {
    grabState.releaseVelocity = velocityTracker.current.getVelocity();
    grabState.pendingRelease = true;
    velocityTracker.current.reset();
  }, []);

  return { processHands };
}
