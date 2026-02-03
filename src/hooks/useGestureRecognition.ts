import { useCallback, useRef } from 'react';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import type { HandLandmarks, Vector3Tuple, GestureType } from '@/types';
import { processHandGesture, getPointerPosition, getHandSize } from '@/engine/gestures';

interface GestureCallbacks {
  onGrab?: (objectId: string, position: Vector3Tuple) => void;
  onRelease?: (objectId: string, position: Vector3Tuple) => void;
  onMove?: (objectId: string, position: Vector3Tuple) => void;
  onScale?: (objectId: string, scale: number) => void;
  onSelect?: (objectId: string | null) => void;
}

export function useGestureRecognition(callbacks?: GestureCallbacks) {
  const { state, setGesture, setPointer, selectObject, setCamera } = useSandbox();
  const lastGestureRef = useRef<GestureType>('none');
  const lastPinchStateRef = useRef(false);
  const initialPinchDistRef = useRef(0);
  const initialScreenPosRef = useRef<{ x: number; y: number } | null>(null);
  const initialHandSizeRef = useRef(0);
  const initialScaleRef = useRef<Vector3Tuple>([1, 1, 1]);
  const initialCameraRadiusRef = useRef(0);
  const initialCameraAngleRef = useRef(0);
  const smoothedDeltaXRef = useRef(0);
  const smoothedHandSizeRatioRef = useRef(1);
  const raycasterRef = useRef(new THREE.Raycaster());
  const planeRef = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0));
  
  // Keep state in refs to avoid stale closures
  const stateRef = useRef(state);
  stateRef.current = state;
  
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  // Convert screen coordinates to 3D world position
  const screenToWorld = useCallback((
    screenX: number,
    screenY: number,
    camera: THREE.Camera,
    targetY: number = 0
  ): Vector3Tuple => {
    // Convert normalized coordinates (0-1) to NDC (-1 to 1)
    // Note: screenX is mirrored because webcam is mirrored
    const ndcX = (1 - screenX) * 2 - 1;
    const ndcY = -(screenY * 2 - 1);

    // Create ray from camera
    raycasterRef.current.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);

    // Set plane at target Y height
    planeRef.current.set(new THREE.Vector3(0, 1, 0), -targetY);

    // Find intersection with horizontal plane
    const intersection = new THREE.Vector3();
    raycasterRef.current.ray.intersectPlane(planeRef.current, intersection);

    if (intersection) {
      return [intersection.x, targetY, intersection.z];
    }

    // Fallback: project forward from camera
    const direction = raycasterRef.current.ray.direction.clone();
    const origin = raycasterRef.current.ray.origin.clone();
    const t = (targetY - origin.y) / direction.y;
    
    return [
      origin.x + direction.x * t,
      targetY,
      origin.z + direction.z * t,
    ];
  }, []);

  // Find object near pointer position (proximity-based detection)
  const findNearestObject = useCallback((
    pointerPos: Vector3Tuple,
    objects: typeof state.objects,
    threshold: number = 1.5
  ): string | null => {
    let nearestId: string | null = null;
    let nearestDist = threshold;

    for (const obj of objects) {
      const dx = pointerPos[0] - obj.position[0];
      const dy = pointerPos[1] - obj.position[1];
      const dz = pointerPos[2] - obj.position[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist < nearestDist) {
        nearestDist = dist;
        nearestId = obj.id;
      }
    }

    return nearestId;
  }, []);

  // Process hand tracking results
  const processHands = useCallback((
    hands: HandLandmarks[],
    camera: THREE.Camera
  ) => {
    if (hands.length === 0) {
      // No hands detected
      setGesture({
        currentGesture: 'none',
        isPinching: false,
        pointerPosition: null,
        screenPosition: null,
        confidence: 0,
      });
      setPointer({ visible: false, hoveredObjectId: null });
      lastGestureRef.current = 'none';
      lastPinchStateRef.current = false;
      return;
    }

    // Use the first (primary) hand
    const primaryHand = hands[0];
    const result = processHandGesture(primaryHand);

    // Get current state from ref (avoids stale closures)
    const currentState = stateRef.current;
    const currentCallbacks = callbacksRef.current;

    // Get screen position from index finger tip
    const screenPos = result.pointerPosition || getPointerPosition(primaryHand.landmarks);
    
    // Calculate 3D world position
    const targetY = currentState.pointer.grabbedObjectId 
      ? currentState.objects.find(o => o.id === currentState.pointer.grabbedObjectId)?.position[1] || 0.5
      : 0.5;
    
    const worldPosition = screenToWorld(screenPos.x, screenPos.y, camera, targetY);

    // Update gesture state
    setGesture({
      currentGesture: result.gesture,
      pinchDistance: result.pinchDistance,
      isPinching: result.isPinching,
      pointerPosition: worldPosition,
      screenPosition: screenPos,
      confidence: 1,
    });

    // Find nearest object to pointer (proximity-based hover detection)
    const nearestObjectId = currentState.pointer.grabbedObjectId 
      ? currentState.pointer.grabbedObjectId  // Keep grabbed object
      : findNearestObject(worldPosition, currentState.objects);

    // Update pointer visibility - always show when hand detected
    const pointerMode = result.isPinching 
      ? 'grabbing' 
      : (nearestObjectId ? 'hovering' : 'idle');
    
    setPointer({
      visible: true,
      position: worldPosition,
      mode: pointerMode,
      hoveredObjectId: nearestObjectId,
    });

    // Handle grab/release transitions
    const wasPinching = lastPinchStateRef.current;
    const isPinching = result.isPinching;

    // Get current hand size (distance from wrist to middle fingertip) as depth proxy
    const currentHandSize = getHandSize(primaryHand.landmarks);

    if (isPinching && !wasPinching) {
      // Started pinching
      initialScreenPosRef.current = { x: screenPos.x, y: screenPos.y };
      initialHandSizeRef.current = currentHandSize;
      
      // Reset smoothed values
      smoothedDeltaXRef.current = 0;
      smoothedHandSizeRatioRef.current = 1;
      
      // Store initial camera position for rotation/zoom
      const camPos = currentState.camera.position;
      initialCameraRadiusRef.current = Math.sqrt(camPos[0] * camPos[0] + camPos[2] * camPos[2]);
      initialCameraAngleRef.current = Math.atan2(camPos[0], camPos[2]);
      
      if (nearestObjectId) {
        // Grab object
        setPointer({ grabbedObjectId: nearestObjectId, mode: 'grabbing' });
        selectObject(nearestObjectId);
        currentCallbacks?.onGrab?.(nearestObjectId, worldPosition);
        // Store initial scale for scaling
        const obj = currentState.objects.find(o => o.id === nearestObjectId);
        if (obj) {
          initialScaleRef.current = [...obj.scale];
        }
        console.log('Grabbed object:', nearestObjectId);
      } else {
        // No object - will control camera
        console.log('Camera control mode');
      }
      initialPinchDistRef.current = result.pinchDistance;
    } else if (!isPinching && wasPinching) {
      // Released pinch
      if (currentState.pointer.grabbedObjectId) {
        currentCallbacks?.onRelease?.(currentState.pointer.grabbedObjectId, worldPosition);
        setPointer({ grabbedObjectId: null, mode: 'idle' });
        console.log('Released object:', currentState.pointer.grabbedObjectId);
      }
      initialScreenPosRef.current = null;
    } else if (isPinching && initialScreenPosRef.current && initialHandSizeRef.current > 0) {
      const rawDeltaX = screenPos.x - initialScreenPosRef.current.x;
      
      // Hand size ratio: > 1 means hand moved closer, < 1 means moved farther
      const rawHandSizeRatio = currentHandSize / initialHandSizeRef.current;
      
      // Apply smoothing (low-pass filter) to reduce jitter
      const smoothingFactor = 0.3; // Lower = smoother but more lag
      smoothedDeltaXRef.current = smoothedDeltaXRef.current + (rawDeltaX - smoothedDeltaXRef.current) * smoothingFactor;
      smoothedHandSizeRatioRef.current = smoothedHandSizeRatioRef.current + (rawHandSizeRatio - smoothedHandSizeRatioRef.current) * smoothingFactor;
      
      // Apply dead zone to ignore tiny movements
      const deadZone = 0.01;
      const deltaX = Math.abs(smoothedDeltaXRef.current) < deadZone ? 0 : smoothedDeltaXRef.current;
      const handSizeRatio = smoothedHandSizeRatioRef.current;
      
      if (currentState.pointer.grabbedObjectId) {
        // Move grabbed object
        currentCallbacks?.onMove?.(currentState.pointer.grabbedObjectId, worldPosition);
        
        // Scale object based on hand distance from camera (closer = bigger)
        const clampedScale = Math.max(0.3, Math.min(3, handSizeRatio));
        currentCallbacks?.onScale?.(currentState.pointer.grabbedObjectId, clampedScale);
      } else {
        // Camera control - smoothed direct mapping
        const rotationSpeed = 4;
        
        // Rotate camera around Y axis - direct from initial angle + smoothed delta
        const newAngle = initialCameraAngleRef.current + (deltaX * rotationSpeed);
        
        // Zoom camera based on hand distance (closer = zoom in) - direct from initial radius
        const zoomRatio = 1 / handSizeRatio;
        const newRadius = Math.max(5, Math.min(25, initialCameraRadiusRef.current * zoomRatio));
        
        const currentY = currentState.camera.position[1];
        
        setCamera({
          position: [
            Math.sin(newAngle) * newRadius,
            currentY,
            Math.cos(newAngle) * newRadius,
          ],
        });
      }
    }

    lastGestureRef.current = result.gesture;
    lastPinchStateRef.current = isPinching;
  }, [screenToWorld, setGesture, setPointer, selectObject, setCamera, findNearestObject]);

  return {
    processHands,
    screenToWorld,
  };
}

export default useGestureRecognition;
