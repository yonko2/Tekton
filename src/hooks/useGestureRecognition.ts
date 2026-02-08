import { useRef, useCallback } from 'react';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import {
  detectGesture,
  getPointerScreenPosition,
  getPinchMidpoint,
  getHandScale,
  getHandRollAngle,
  getFingerSpread,
  resetSpreadSmoothing,
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


let orbitTheta = 0;
let orbitPhi = Math.PI / 4;
const ORBIT_SENSITIVITY = 3;
const MIN_ZOOM = 5;
const MAX_ZOOM = 30;
const ZOOM_SENSITIVITY = 1.4; 
const DEPTH_SENSITIVITY = 1.2; 
const MIN_GRAB_DIST = 2; 
const MAX_GRAB_DIST = 25; 
const ORBIT_DEADZONE = 0.003; 
const SCALE_DEADZONE = 0.03; 
const TWIST_SENSITIVITY = 5.0; 
const MIN_SCALE = 0.3;
const MAX_SCALE = 4.0;

function sphericalToCartesian(theta: number, phi: number, r: number): Vector3Tuple {
  return [
    r * Math.sin(phi) * Math.sin(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.cos(theta),
  ];
}


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


export function useGestureRecognition() {
  const { state, setGesture, setPointer, setCamera, selectObject, updateObjectScale, updateObjectPosition } = useSandbox();

  const velocityTracker = useRef(new VelocityTracker());
  const lastScreenPos = useRef<{ x: number; y: number } | null>(null);
  const grabOffsetRef = useRef<Vector3Tuple>([0, 0, 0]);
  const wasPinching = useRef(false);
  const modeRef = useRef<'idle' | 'grab' | 'camera'>('idle');

  
  const initialHandScaleRef = useRef(0);
  const initialZoomRadiusRef = useRef(cameraZoomState.radius);

  
  const grabCameraDistRef = useRef(0);
  const initialGrabDistRef = useRef(0);

  
  const initialRollAngleRef = useRef(0);

  
  const initialSpreadRef = useRef(0);
  const secondHandActiveRef = useRef(false);
  const initialObjectScaleRef = useRef<Vector3Tuple>([1, 1, 1]);

  const objectIdsRef = useRef<string[]>([]);
  objectIdsRef.current = state.objects.map((o) => o.id);

  
  const objectsRef = useRef(state.objects);
  objectsRef.current = state.objects;

  
  const processHands = useCallback(
    (hands: HandData[], camera: THREE.Camera, _scene: THREE.Scene) => {
      
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

      
      if (pinching && !wasPinching.current) {
        initialHandScaleRef.current = handScale;

        const hit = findNearestObject(screenPos, camera, objectIdsRef.current);
        if (hit) {
          modeRef.current = 'grab';
          velocityTracker.current.reset();

          
          
          const camPos = new THREE.Vector3();
          camera.getWorldPosition(camPos);
          grabCameraDistRef.current = camPos.distanceTo(hit.worldPos);

          
          initialGrabDistRef.current = grabCameraDistRef.current;

          
          
          const rayWorldPos = screenToWorldAtDistance(screenPos, camera, grabCameraDistRef.current);
          grabOffsetRef.current = [
            hit.worldPos.x - rayWorldPos[0],
            hit.worldPos.y - rayWorldPos[1],
            hit.worldPos.z - rayWorldPos[2],
          ];

          initialRollAngleRef.current = rollAngle;

          
          const obj = objectsRef.current.find((o) => o.id === hit.id);
          initialObjectScaleRef.current = obj ? [...obj.scale] : [1, 1, 1];
          secondHandActiveRef.current = false;
          resetSpreadSmoothing();

          
          
          const fwd = new THREE.Vector3();
          camera.getWorldDirection(fwd);
          fwd.normalize();

          grabState.objectId = hit.id;
          grabState.targetPosition = [hit.worldPos.x, hit.worldPos.y, hit.worldPos.z];
          grabState.twistAngle = 0;
          grabState.twistAxis = [fwd.x, fwd.y, fwd.z];
          grabState.scaleFactor = 1;
          grabState.pendingRelease = false;

          selectObject(hit.id);
          setPointer({ visible: true, mode: 'grabbing', grabbedObjectId: hit.id });
        } else {
          
          modeRef.current = 'camera';
          lastScreenPos.current = screenPos;
          initialZoomRadiusRef.current = cameraZoomState.radius;
          setPointer({ visible: true, mode: 'camera', grabbedObjectId: null });
        }
      }

      
      if (pinching && wasPinching.current) {
        
        const scaleRatio = initialHandScaleRef.current > 0.001
          ? handScale / initialHandScaleRef.current
          : 1;

        if (modeRef.current === 'grab' && grabState.objectId) {
          
          
          
          if (Math.abs(scaleRatio - 1) > SCALE_DEADZONE) {
            const depthRatio = Math.pow(1 / scaleRatio, DEPTH_SENSITIVITY);
            grabCameraDistRef.current = Math.max(
              MIN_GRAB_DIST,
              Math.min(MAX_GRAB_DIST, initialGrabDistRef.current * depthRatio),
            );
          }

          
          
          
          
          const rayPos = screenToWorldAtDistance(screenPos, camera, grabCameraDistRef.current);
          const newPos: Vector3Tuple = [
            rayPos[0] + grabOffsetRef.current[0],
            rayPos[1] + grabOffsetRef.current[1],
            rayPos[2] + grabOffsetRef.current[2],
          ];

          
          const safePos = constrainToGround(newPos, 0.25);
          velocityTracker.current.record(safePos);
          grabState.targetPosition = safePos;

          
          const twistDelta = rollAngle - initialRollAngleRef.current;
          grabState.twistAngle = twistDelta * TWIST_SENSITIVITY;

          
          if (hands.length >= 2) {
            const secondHand = hands[1];
            const spread = getFingerSpread(secondHand.landmarks);

            if (!secondHandActiveRef.current) {
              
              initialSpreadRef.current = spread;
              secondHandActiveRef.current = true;
            } else {
              
              const ratio = initialSpreadRef.current > 0.001
                ? spread / initialSpreadRef.current
                : 1;
              grabState.scaleFactor = Math.max(MIN_SCALE, Math.min(MAX_SCALE, ratio));
            }
          } else if (secondHandActiveRef.current) {
            
            secondHandActiveRef.current = false;
            resetSpreadSmoothing();
          }

          setPointer({ position: safePos });

        } else if (modeRef.current === 'camera' && lastScreenPos.current) {
          
          const dx = screenPos.x - lastScreenPos.current.x;
          const dy = screenPos.y - lastScreenPos.current.y;

          
          if (Math.abs(dx) > ORBIT_DEADZONE || Math.abs(dy) > ORBIT_DEADZONE) {
            orbitTheta += dx * ORBIT_SENSITIVITY;
            orbitPhi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, orbitPhi + dy * ORBIT_SENSITIVITY));
            lastScreenPos.current = screenPos;
          }

          
          
          
          if (Math.abs(scaleRatio - 1) > SCALE_DEADZONE) {
            const zoomRatio = Math.pow(1 / scaleRatio, ZOOM_SENSITIVITY);
            const newRadius = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, initialZoomRadiusRef.current * zoomRatio));
            cameraZoomState.radius = newRadius;
          }

          setCamera({ position: sphericalToCartesian(orbitTheta, orbitPhi, cameraZoomState.radius) });
        }
      }

      
      if (!pinching && wasPinching.current) {
        if (modeRef.current === 'grab' && grabState.objectId) {
          const f = grabState.scaleFactor;
          if (f !== 1) {
            const [sx, sy, sz] = initialObjectScaleRef.current;
            updateObjectScale(grabState.objectId, [sx * f, sy * f, sz * f]);
            updateObjectPosition(grabState.objectId, grabState.targetPosition);
          }
          secondHandActiveRef.current = false;
          resetSpreadSmoothing();
          doRelease();
        }
        modeRef.current = 'idle';
        setPointer({ mode: 'idle', grabbedObjectId: null });
      }

      
      if (!pinching && gesture === 'point') {
        const hit = findNearestObject(screenPos, camera, objectIdsRef.current);
        setPointer({ visible: true, position: constrained, mode: 'hovering', hoveredObjectId: hit?.id ?? null });
      } else if (!pinching && gesture === 'none') {
        setPointer({ visible: false, mode: 'idle', hoveredObjectId: null });
      }

      wasPinching.current = pinching;
    },
    [setGesture, setPointer, setCamera, selectObject, updateObjectScale, updateObjectPosition],
  );

  const doRelease = useCallback(() => {
    grabState.releaseVelocity = velocityTracker.current.getVelocity();
    grabState.pendingRelease = true;
    velocityTracker.current.reset();
  }, []);

  return { processHands };
}
