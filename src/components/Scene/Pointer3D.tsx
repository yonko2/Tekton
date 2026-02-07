import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';

const COLOR_MAP: Record<string, string> = {
  idle: '#ffffff',
  hovering: '#ffeb3b',
  grabbing: '#00e5ff',
  camera: '#ff9800',
};

export function Pointer3D() {
  const { state } = useSandbox();
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (!meshRef.current || !glowRef.current) return;

    const visible = state.pointer.visible;
    meshRef.current.visible = visible;
    glowRef.current.visible = visible;

    if (!visible) return;

    const [x, y, z] = state.pointer.position;
    meshRef.current.position.set(x, y, z);
    glowRef.current.position.set(x, y, z);

    // Pulsing scale
    const pulse = 1 + Math.sin(clock.elapsedTime * 4) * 0.15;
    glowRef.current.scale.setScalar(pulse);
  });

  const color = COLOR_MAP[state.pointer.mode] ?? '#ffffff';

  return (
    <>
      {/* Core dot */}
      <mesh ref={meshRef} visible={false}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>

      {/* Glow ring */}
      <mesh ref={glowRef} visible={false}>
        <ringGeometry args={[0.12, 0.18, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
        />
      </mesh>
    </>
  );
}
