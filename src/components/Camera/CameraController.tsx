import { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useSandbox } from '@/context/SandboxContext';

export function CameraController() {
  const { state } = useSandbox();
  const { camera } = useThree();
  const controlsRef = useRef<{ enabled: boolean; update: () => void } | null>(null);

  useFrame(() => {
    const gestureControlling = state.pointer.mode === 'camera';

    if (controlsRef.current) {
      controlsRef.current.enabled = !gestureControlling;
    }

    if (gestureControlling) {
      // Snap directly -- no lerp / acceleration
      camera.position.set(
        state.camera.position[0],
        state.camera.position[1],
        state.camera.position[2],
      );
      camera.lookAt(0, 0, 0);
    }
  });

  return (
    <OrbitControls
      ref={controlsRef as React.Ref<never>}
      enableDamping={false}
      minDistance={5}
      maxDistance={30}
      maxPolarAngle={Math.PI / 2 - 0.05}
      target={[0, 0, 0]}
    />
  );
}
