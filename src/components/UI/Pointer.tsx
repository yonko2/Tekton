import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSandbox } from '@/context/SandboxContext';
import type { PointerMode } from '@/types';

// Get color based on pointer mode
function getPointerColor(mode: PointerMode): string {
  switch (mode) {
    case 'idle':
      return '#ffffff';
    case 'hovering':
      return '#00ff00';
    case 'grabbing':
      return '#ff6600';
    case 'scaling':
      return '#ff00ff';
    default:
      return '#ffffff';
  }
}

export function Pointer3D() {
  const { state } = useSandbox();
  const groupRef = useRef<THREE.Group>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const innerRef = useRef<THREE.Mesh>(null);
  const lastLogRef = useRef(0);

  useFrame((_, delta) => {
    if (ringRef.current) {
      // Rotate the ring
      ringRef.current.rotation.z += delta * 2;
    }
    
    if (innerRef.current) {
      // Pulse the inner sphere
      const scale = 1 + Math.sin(Date.now() * 0.005) * 0.1;
      innerRef.current.scale.setScalar(scale);
    }

    // Debug log (throttled)
    const now = Date.now();
    if (state.pointer.visible && now - lastLogRef.current > 1000) {
      console.log('Pointer3D rendering at:', state.pointer.position);
      lastLogRef.current = now;
    }
  });

  if (!state.pointer.visible) {
    return null;
  }

  const color = getPointerColor(state.pointer.mode);

  return (
    <group ref={groupRef} position={state.pointer.position}>
      {/* Outer ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.15, 0.02, 8, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.8} />
      </mesh>

      {/* Inner sphere */}
      <mesh ref={innerRef}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>

      {/* Vertical line indicator */}
      <mesh position={[0, -0.5, 0]}>
        <cylinderGeometry args={[0.01, 0.01, 1, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>

      {/* Ground circle indicator */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -state.pointer.position[1] + 0.01, 0]}>
        <ringGeometry args={[0.2, 0.25, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

export default Pointer3D;
