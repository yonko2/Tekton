import { useThree } from '@react-three/fiber';
import { useSandbox } from '@/context/SandboxContext';

export function CameraController() {
  const { camera } = useThree();
  const { state } = useSandbox();

  // Sync camera position from state (controlled by gesture recognition)
  if (state.camera.position) {
    camera.position.set(
      state.camera.position[0],
      state.camera.position[1],
      state.camera.position[2]
    );
    camera.lookAt(0, 0, 0);
  }

  return null;
}

export default CameraController;
