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
import type { HandData, Vector3Tuple, GestureType } from '@/types';
import type { RapierRigidBody } from '@react-three/rapier';

/**
 * Refs map that PhysicsObject components register into so that the
 * gesture hook can toggle kinematic / dynamic and apply velocities.
 */
export const rigidBodyRefs = new Map<string, React.RefObject<RapierRigidBody | null>>();

// ── Camera orbit state (module-level so it persists across renders) ──
let orbitTheta = 0; // horizontal angle
let orbitPhi = Math.PI / 4; // vertical angle (0 = top, PI/2 = horizon)
const ORBIT_RADIUS = 14;
const ORBIT_SENSITIVITY = 3;

function sphericalToCartesian(theta: number, phi: number, r: number): Vector3Tuple {
  return [
    r * Math.sin(phi) * Math.sin(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.cos(theta),
  ];
}

export function useGestureRecognition() {
  const {
    state,
    setGesture,
    setPointer,
    setCamera,
    selectObject,
  } = useSandbox();

  const velocityTracker = useRef(new VelocityTracker());
  const lastScreenPos = useRef<{ x: number; y: number } | null>(null);
  const grabOffsetRef = useRef<Vector3Tuple>([0, 0, 0]);
  const grabbedIdRef = useRef<string | null>(null);
  const wasPinching = useRef(false);
  const modeRef = useRef<'idle' | 'grab' | 'camera'>('idle');

  // ── Raycast into scene to find hovered object ──────────────
  const raycaster = useRef(new THREE.Raycaster());
  const sceneRef = useRef<THREE.Scene | null>(null);

  const findObjectAtScreen = useCallback(
    (screenPos: { x: number; y: number }, camera: THREE.Camera): string | null => {
      if (!sceneRef.current) return null;
      const ndc = new THREE.Vector2(screenPos.x * 2 - 1, -(screenPos.y * 2 - 1));
      raycaster.current.setFromCamera(ndc, camera);
      const hits = raycaster.current.intersectObjects(sceneRef.current.children, true);
      for (const hit of hits) {
        // Walk up to find a group with userData.objectId
        let obj: THREE.Object3D | null = hit.object;
        while (obj) {
          if (obj.userData?.objectId) return obj.userData.objectId as string;
          obj = obj.parent;
        }
      }
      return null;
    },
    [],
  );

  // ── Main per-frame processor ───────────────────────────────
  const processHands = useCallback(
    (hands: HandData[], camera: THREE.Camera, scene: THREE.Scene) => {
      sceneRef.current = scene;

      if (hands.length === 0) {
        // No hands visible – reset all gesture state
        resetPinchState();
        setGesture({ currentGesture: 'none', isPinching: false, pointerPosition: null, screenPosition: null });
        setPointer({ visible: false, mode: 'idle', hoveredObjectId: null });

        // If we were grabbing, release
        if (modeRef.current === 'grab' && grabbedIdRef.current) {
          releaseObject(grabbedIdRef.current);
        }
        modeRef.current = 'idle';
        wasPinching.current = false;
        return;
      }

      const hand = hands[0];
      const gesture: GestureType = detectGesture(hand.landmarks);
      const pinching = isPinching(hand.landmarks);
      const screenPos = pinching
        ? getPinchMidpoint(hand.landmarks)
        : getPointerScreenPosition(hand.landmarks);

      const worldPos = screenToWorld(screenPos, camera, 0.5);
      const constrained = constrainToGround(worldPos);

      // Update gesture state
      setGesture({
        currentGesture: gesture,
        isPinching: pinching,
        pointerPosition: constrained,
        screenPosition: screenPos,
      });

      // ── Pinch just started ──────────────────────────────────
      if (pinching && !wasPinching.current) {
        const hitId = findObjectAtScreen(screenPos, camera);
        if (hitId) {
          // Grab object
          modeRef.current = 'grab';
          grabbedIdRef.current = hitId;
          selectObject(hitId);
          velocityTracker.current.reset();

          // Set rigid body to kinematic
          const rbRef = rigidBodyRefs.get(hitId);
          if (rbRef?.current) {
            rbRef.current.setBodyType(2, true); // 2 = KinematicPositionBased
          }

          // Compute grab offset
          const obj = state.objects.find((o) => o.id === hitId);
          if (obj) {
            grabOffsetRef.current = [
              obj.position[0] - constrained[0],
              obj.position[1] - constrained[1],
              obj.position[2] - constrained[2],
            ];
          }

          setPointer({ visible: true, mode: 'grabbing', grabbedObjectId: hitId });
        } else {
          // Pinch on empty space -> camera mode
          modeRef.current = 'camera';
          lastScreenPos.current = screenPos;
          setPointer({ visible: true, mode: 'camera', grabbedObjectId: null });
        }
      }

      // ── Ongoing pinch ───────────────────────────────────────
      if (pinching && wasPinching.current) {
        if (modeRef.current === 'grab' && grabbedIdRef.current) {
          // Move the grabbed object
          const newPos: Vector3Tuple = [
            constrained[0] + grabOffsetRef.current[0],
            constrained[1] + grabOffsetRef.current[1],
            constrained[2] + grabOffsetRef.current[2],
          ];
          const safePos = constrainToGround(newPos, 0.25);

          velocityTracker.current.record(safePos);

          // Move kinematic body
          const rbRef = rigidBodyRefs.get(grabbedIdRef.current);
          if (rbRef?.current) {
            rbRef.current.setNextKinematicTranslation({
              x: safePos[0],
              y: safePos[1],
              z: safePos[2],
            });
          }

          setPointer({ position: safePos });
        } else if (modeRef.current === 'camera' && lastScreenPos.current) {
          // Orbit camera
          const dx = screenPos.x - lastScreenPos.current.x;
          const dy = screenPos.y - lastScreenPos.current.y;

          orbitTheta += dx * ORBIT_SENSITIVITY;
          orbitPhi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, orbitPhi + dy * ORBIT_SENSITIVITY));

          const camPos = sphericalToCartesian(orbitTheta, orbitPhi, ORBIT_RADIUS);
          setCamera({ position: camPos });
          lastScreenPos.current = screenPos;
        }
      }

      // ── Pinch released ──────────────────────────────────────
      if (!pinching && wasPinching.current) {
        if (modeRef.current === 'grab' && grabbedIdRef.current) {
          releaseObject(grabbedIdRef.current);
        }
        modeRef.current = 'idle';
        setPointer({ mode: 'idle', grabbedObjectId: null });
      }

      // ── Pointing (no pinch) - show pointer & highlight ──────
      if (!pinching && gesture === 'point') {
        const hitId = findObjectAtScreen(screenPos, camera);
        setPointer({
          visible: true,
          position: constrained,
          mode: 'hovering',
          hoveredObjectId: hitId,
        });
      } else if (!pinching && gesture === 'none') {
        setPointer({ visible: false, mode: 'idle', hoveredObjectId: null });
      }

      wasPinching.current = pinching;
    },
    [setGesture, setPointer, setCamera, selectObject, findObjectAtScreen, state.objects],
  );

  // ── Release object & apply throw velocity ──────────────────
  const releaseObject = useCallback(
    (objectId: string) => {
      const rbRef = rigidBodyRefs.get(objectId);
      if (rbRef?.current) {
        // Switch back to dynamic
        rbRef.current.setBodyType(0, true); // 0 = Dynamic
        // Apply throw velocity
        const vel = velocityTracker.current.getVelocity();
        rbRef.current.setLinvel({ x: vel[0], y: vel[1], z: vel[2] }, true);
      }
      grabbedIdRef.current = null;
      velocityTracker.current.reset();
    },
    [],
  );

  return { processHands };
}
