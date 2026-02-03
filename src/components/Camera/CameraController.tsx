import { useRef, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import { SCENE_CONFIG } from '@/types';

export function CameraController() {
  const { camera } = useThree();
  const { state, setCamera } = useSandbox();
  const targetPosition = useRef(new THREE.Vector3(...SCENE_CONFIG.camera.position));
  const targetLookAt = useRef(new THREE.Vector3(0, 0, 0));
  const isAnimating = useRef(false);

  // Reset camera to default position
  const resetCamera = () => {
    targetPosition.current.set(...SCENE_CONFIG.camera.position);
    targetLookAt.current.set(0, 0, 0);
    isAnimating.current = true;
  };

  // Update camera position from state
  useEffect(() => {
    if (state.gesture.currentGesture === 'fist') {
      resetCamera();
    }
  }, [state.gesture.currentGesture]);

  // Smooth camera animation
  useFrame(() => {
    if (isAnimating.current) {
      const currentPos = new THREE.Vector3().copy(camera.position);
      const diff = targetPosition.current.clone().sub(currentPos);
      
      if (diff.length() > 0.01) {
        camera.position.lerp(targetPosition.current, 0.05);
        camera.lookAt(targetLookAt.current);
      } else {
        isAnimating.current = false;
        setCamera({
          position: [camera.position.x, camera.position.y, camera.position.z],
          target: [targetLookAt.current.x, targetLookAt.current.y, targetLookAt.current.z],
        });
      }
    }
  });

  return null;
}

export default CameraController;
