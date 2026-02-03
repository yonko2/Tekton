import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { SceneObject as SceneObjectType, ShapeType } from '@/types';

interface SceneObjectProps {
  object: SceneObjectType;
  onPointerOver?: () => void;
  onPointerOut?: () => void;
  onClick?: () => void;
}

// Pyramid geometry helper
function createPyramidGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.ConeGeometry(0.7, 1, 4);
  geometry.rotateY(Math.PI / 4);
  return geometry;
}

// Get geometry based on shape type
function ShapeGeometry({ type }: { type: ShapeType }) {
  const pyramidGeometry = useMemo(() => {
    if (type === 'pyramid') {
      return createPyramidGeometry();
    }
    return null;
  }, [type]);

  switch (type) {
    case 'cube':
      return <boxGeometry args={[1, 1, 1]} />;
    case 'sphere':
      return <sphereGeometry args={[0.5, 32, 32]} />;
    case 'cylinder':
      return <cylinderGeometry args={[0.5, 0.5, 2, 32]} />;
    case 'cone':
      return <coneGeometry args={[0.5, 1, 32]} />;
    case 'torus':
      return <torusGeometry args={[0.4, 0.15, 16, 32]} />;
    case 'pyramid':
      return pyramidGeometry ? <primitive object={pyramidGeometry} attach="geometry" /> : <coneGeometry args={[0.7, 1, 4]} />;
    default:
      return <boxGeometry args={[1, 1, 1]} />;
  }
}

export function SceneObject({ object, onPointerOver, onPointerOut, onClick }: SceneObjectProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const outlineRef = useRef<THREE.Mesh>(null);

  // Animate selected objects
  useFrame((state) => {
    if (meshRef.current && object.isSelected) {
      // Subtle floating animation for selected objects
      meshRef.current.position.y = object.position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.05;
    }
  });

  return (
    <group position={object.position}>
      {/* Main mesh */}
      <mesh
        ref={meshRef}
        rotation={object.rotation}
        scale={object.scale}
        castShadow
        receiveShadow
        onPointerOver={(e) => {
          e.stopPropagation();
          onPointerOver?.();
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          onPointerOut?.();
        }}
        onClick={(e) => {
          e.stopPropagation();
          onClick?.();
        }}
      >
        <ShapeGeometry type={object.type} />
        <meshStandardMaterial
          color={object.color}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>

      {/* Selection outline */}
      {object.isSelected && (
        <mesh
          ref={outlineRef}
          rotation={object.rotation}
          scale={[
            object.scale[0] * 1.05,
            object.scale[1] * 1.05,
            object.scale[2] * 1.05,
          ]}
        >
          <ShapeGeometry type={object.type} />
          <meshBasicMaterial
            color="#00ffff"
            wireframe
            transparent
            opacity={0.5}
          />
        </mesh>
      )}
    </group>
  );
}

export default SceneObject;
