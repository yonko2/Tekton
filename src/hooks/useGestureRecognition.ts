import { useCallback, useRef } from 'react';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import type { HandLandmarks, Vector3Tuple, GestureType } from '@/types';
import { processHandGesture, getPointerPosition } from '@/engine/gestures';

interface GestureCallbacks {
  onGrab?: (objectId: string, position: Vector3Tuple) => void;
  onRelease?: (objectId: string, position: Vector3Tuple) => void;
  onMove?: (objectId: string, position: Vector3Tuple) => void;
  onScale?: (objectId: string, scale: number) => void;
  onSelect?: (objectId: string | null) => void;
}

export function useGestureRecognition(callbacks?: GestureCallbacks) {
  const { state, setGesture, setPointer, selectObject } = useSandbox();
  const lastGestureRef = useRef<GestureType>('none');
  const lastPinchStateRef = useRef(false);
  const initialPinchDistRef = useRef(0);
  const raycasterRef = useRef(new THREE.Raycaster());
  const planeRef = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0));

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
      setPointer({ visible: false });
      lastGestureRef.current = 'none';
      lastPinchStateRef.current = false;
      return;
    }

    // Use the first (primary) hand
    const primaryHand = hands[0];
    const result = processHandGesture(primaryHand);

    // Get screen position from pointer
    const screenPos = result.pointerPosition || getPointerPosition(primaryHand.landmarks);
    
    // Calculate 3D world position
    const targetY = state.pointer.grabbedObjectId 
      ? state.objects.find(o => o.id === state.pointer.grabbedObjectId)?.position[1] || 0.5
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

    // Update pointer visibility and position
    const isPointing = result.gesture === 'point' || result.gesture === 'pinch';
    setPointer({
      visible: isPointing || result.isPinching,
      position: worldPosition,
      mode: result.isPinching ? 'grabbing' : (state.pointer.hoveredObjectId ? 'hovering' : 'idle'),
    });

    // Handle grab/release transitions
    const wasPinching = lastPinchStateRef.current;
    const isPinching = result.isPinching;

    if (isPinching && !wasPinching) {
      // Started pinching - grab
      if (state.pointer.hoveredObjectId) {
        setPointer({ grabbedObjectId: state.pointer.hoveredObjectId, mode: 'grabbing' });
        selectObject(state.pointer.hoveredObjectId);
        callbacks?.onGrab?.(state.pointer.hoveredObjectId, worldPosition);
      }
      initialPinchDistRef.current = result.pinchDistance;
    } else if (!isPinching && wasPinching) {
      // Released pinch - drop
      if (state.pointer.grabbedObjectId) {
        callbacks?.onRelease?.(state.pointer.grabbedObjectId, worldPosition);
        setPointer({ grabbedObjectId: null, mode: 'idle' });
      }
    } else if (isPinching && state.pointer.grabbedObjectId) {
      // Continuing to pinch - move object
      callbacks?.onMove?.(state.pointer.grabbedObjectId, worldPosition);
    }

    // Handle two-finger pinch for scaling
    if (result.gesture === 'two_finger_pinch' && state.selectedObjectId) {
      const scaleFactor = initialPinchDistRef.current / result.pinchDistance;
      callbacks?.onScale?.(state.selectedObjectId, scaleFactor);
    }

    // Handle selection on point gesture
    if (result.gesture === 'point' && lastGestureRef.current !== 'point') {
      // Just started pointing - could trigger selection
    }

    lastGestureRef.current = result.gesture;
    lastPinchStateRef.current = isPinching;
  }, [
    state.pointer.hoveredObjectId,
    state.pointer.grabbedObjectId,
    state.selectedObjectId,
    state.objects,
    screenToWorld,
    setGesture,
    setPointer,
    selectObject,
    callbacks,
  ]);

  return {
    processHands,
    screenToWorld,
  };
}

export default useGestureRecognition;
