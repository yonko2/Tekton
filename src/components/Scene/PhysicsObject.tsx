import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { RigidBody, type RapierRigidBody } from '@react-three/rapier';
import type { SceneObject as SceneObjectType, ShapeType } from '@/types';
import { rigidBodyRefs } from '@/hooks/useGestureRecognition';

// ── Shape geometry component ─────────────────────────────────
function ShapeGeometry({ type }: { type: ShapeType }) {
  const pyramidGeo = useMemo(() => {
    if (type !== 'pyramid') return null;
    const geo = new THREE.ConeGeometry(0.7, 1, 4);
    geo.rotateY(Math.PI / 4);
    return geo;
  }, [type]);

  switch (type) {
    case 'cube':
      return <boxGeometry args={[1, 1, 1]} />;
    case 'sphere':
      return <sphereGeometry args={[0.5, 32, 32]} />;
    case 'cylinder':
      return <cylinderGeometry args={[0.5, 0.5, 1.5, 32]} />;
    case 'cone':
      return <coneGeometry args={[0.5, 1, 32]} />;
    case 'torus':
      return <torusGeometry args={[0.4, 0.15, 16, 32]} />;
    case 'pyramid':
      return pyramidGeo ? (
        <primitive object={pyramidGeo} attach="geometry" />
      ) : (
        <coneGeometry args={[0.7, 1, 4]} />
      );
    default:
      return <boxGeometry args={[1, 1, 1]} />;
  }
}

// ── Collider type selector ───────────────────────────────────
function colliderFor(type: ShapeType): 'cuboid' | 'ball' | 'hull' {
  switch (type) {
    case 'cube':
    case 'pyramid':
      return 'cuboid';
    case 'sphere':
      return 'ball';
    default:
      return 'hull';
  }
}

// ── PhysicsObject ────────────────────────────────────────────
interface PhysicsObjectProps {
  object: SceneObjectType;
  onPointerOver?: () => void;
  onPointerOut?: () => void;
  onClick?: () => void;
}

export function PhysicsObject({
  object,
  onPointerOver,
  onPointerOut,
  onClick,
}: PhysicsObjectProps) {
  const rbRef = useRef<RapierRigidBody>(null);
  const meshRef = useRef<THREE.Mesh>(null);

  // Register / unregister rigid-body ref in the global map
  const rbRefStable = useRef<React.RefObject<RapierRigidBody | null>>(rbRef);
  useEffect(() => {
    rigidBodyRefs.set(object.id, rbRefStable.current);
    return () => {
      rigidBodyRefs.delete(object.id);
    };
  }, [object.id]);

  // Subtle floating animation when selected
  useFrame(({ clock }) => {
    if (meshRef.current && object.isSelected) {
      meshRef.current.position.y = Math.sin(clock.elapsedTime * 2) * 0.04;
    } else if (meshRef.current) {
      meshRef.current.position.y = 0;
    }
  });

  return (
    <RigidBody
      ref={rbRef}
      type="dynamic"
      colliders={colliderFor(object.type)}
      position={object.position}
      rotation={object.rotation}
      friction={0.8}
      restitution={0.2}
      mass={1}
      userData={{ objectId: object.id }}
    >
      <group userData={{ objectId: object.id }}>
        {/* Main mesh */}
        <mesh
          ref={meshRef}
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
            emissive={object.isSelected ? object.color : '#000000'}
            emissiveIntensity={object.isSelected ? 0.15 : 0}
          />
        </mesh>

        {/* Selection wireframe overlay */}
        {object.isSelected && (
          <mesh
            scale={[
              object.scale[0] * 1.06,
              object.scale[1] * 1.06,
              object.scale[2] * 1.06,
            ]}
          >
            <ShapeGeometry type={object.type} />
            <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.4} />
          </mesh>
        )}
      </group>
    </RigidBody>
  );
}
