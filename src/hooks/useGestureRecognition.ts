import { useRef, useCallback } from 'react';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import {
  detectGesture,
  getPointerScreenPosition,
  getPinchMidpoint,
  screenToWorld,
  isPinching,
  resetPinchState,
} from '@/engine/gestures';
import { VelocityTracker, constrainToGround } from '@/engine/physics';
import { grabState } from '@/engine/grabStore';
import type { HandData, Vector3Tuple, GestureType } from '@/types';
import type { RapierRigidBody } from '@react-three/rapier';

/**
 * Refs map that PhysicsObject components register into.
 * Used for proximity hit-testing (reading live positions).
 */
export const rigidBodyRefs = new Map<string, React.RefObject<RapierRigidBody | null>>();

// ── Camera orbit state ───────────────────────────────────────
let orbitTheta = 0;
let orbitPhi = Math.PI / 4;
const ORBIT_RADIUS = 14;
const ORBIT_SENSITIVITY = 3;

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

  const objectIdsRef = useRef<string[]>([]);
  objectIdsRef.current = state.objects.map((o) => o.id);

  // ── Main per-frame processor ───────────────────────────────
  const processHands = useCallback(
    (hands: HandData[], camera: THREE.Camera, _scene: THREE.Scene) => {
      // ── No hands ────────────────────────────────────────────
      if (hands.length === 0) {
        resetPinchState();
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
      const pinching = isPinching(hand.landmarks); // exactly ONCE per frame
      const gesture: GestureType = detectGesture(hand.landmarks, pinching);

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
        const hit = findNearestObject(screenPos, camera, objectIdsRef.current);
        if (hit) {
          modeRef.current = 'grab';
          velocityTracker.current.reset();

          // Compute grab offset from live physics position
          grabOffsetRef.current = [
            hit.worldPos.x - constrained[0],
            hit.worldPos.y - constrained[1],
            hit.worldPos.z - constrained[2],
          ];

          // Write to shared grab store (PhysicsObject reads this in useFrame)
          grabState.objectId = hit.id;
          grabState.targetPosition = [hit.worldPos.x, hit.worldPos.y, hit.worldPos.z];
          grabState.pendingRelease = false;

          selectObject(hit.id);
          setPointer({ visible: true, mode: 'grabbing', grabbedObjectId: hit.id });
        } else {
          modeRef.current = 'camera';
          lastScreenPos.current = screenPos;
          setPointer({ visible: true, mode: 'camera', grabbedObjectId: null });
        }
      }

      // ── Ongoing pinch ───────────────────────────────────────
      if (pinching && wasPinching.current) {
        if (modeRef.current === 'grab' && grabState.objectId) {
          const newPos: Vector3Tuple = [
            constrained[0] + grabOffsetRef.current[0],
            constrained[1] + grabOffsetRef.current[1],
            constrained[2] + grabOffsetRef.current[2],
          ];
          const safePos = constrainToGround(newPos, 0.25);

          velocityTracker.current.record(safePos);

          // Update the grab store – PhysicsObject will apply this in useFrame
          grabState.targetPosition = safePos;

          setPointer({ position: safePos });
        } else if (modeRef.current === 'camera' && lastScreenPos.current) {
          const dx = screenPos.x - lastScreenPos.current.x;
          const dy = screenPos.y - lastScreenPos.current.y;

          orbitTheta += dx * ORBIT_SENSITIVITY;
          orbitPhi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, orbitPhi + dy * ORBIT_SENSITIVITY));

          setCamera({ position: sphericalToCartesian(orbitTheta, orbitPhi, ORBIT_RADIUS) });
          lastScreenPos.current = screenPos;
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

  // ── Release: write velocity into the store ─────────────────
  const doRelease = useCallback(() => {
    grabState.releaseVelocity = velocityTracker.current.getVelocity();
    grabState.pendingRelease = true;
    // PhysicsObject will read pendingRelease in useFrame, switch to dynamic,
    // apply velocity, then clear the grab store.
    velocityTracker.current.reset();
  }, []);

  return { processHands };
}
