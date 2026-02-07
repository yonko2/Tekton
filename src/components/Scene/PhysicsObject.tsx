import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { RigidBody, type RapierRigidBody } from '@react-three/rapier';
import type { SceneObject as SceneObjectType, ShapeType } from '@/types';
import { rigidBodyRefs } from '@/hooks/useGestureRecognition';
import { grabState } from '@/engine/grabStore';

// ── Shape geometry ───────────────────────────────────────────
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

// ── Collider selector ────────────────────────────────────────
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

  // Track whether THIS object is currently grabbed (local to this component)
  const isGrabbed = useRef(false);

  // Register rigid-body ref in the global map
  const rbRefStable = useRef<React.RefObject<RapierRigidBody | null>>(rbRef);
  useEffect(() => {
    rigidBodyRefs.set(object.id, rbRefStable.current);
    return () => {
      rigidBodyRefs.delete(object.id);
    };
  }, [object.id]);

  // ── Per-frame: read grab store and drive the body ──────────
  useFrame(() => {
    const rb = rbRef.current;
    if (!rb) return;

    const amGrabbed = grabState.objectId === object.id;

    // ── Transition: not grabbed → grabbed ────────────────────
    if (amGrabbed && !isGrabbed.current) {
      isGrabbed.current = true;
      rb.setBodyType(2, true); // KinematicPositionBased
      rb.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }

    // ── While grabbed: follow the target position ────────────
    if (amGrabbed && isGrabbed.current) {
      const [tx, ty, tz] = grabState.targetPosition;
      rb.setNextKinematicTranslation({ x: tx, y: ty, z: tz });
    }

    // ── Pending release ──────────────────────────────────────
    if (amGrabbed && grabState.pendingRelease) {
      isGrabbed.current = false;
      rb.setBodyType(0, true); // Dynamic
      const [vx, vy, vz] = grabState.releaseVelocity;
      rb.setLinvel({ x: vx, y: vy, z: vz }, true);

      // Clear the grab store
      grabState.objectId = null;
      grabState.pendingRelease = false;
      grabState.releaseVelocity = [0, 0, 0];
    }

    // ── Transition: was grabbed but store says someone else / nobody ──
    if (!amGrabbed && isGrabbed.current) {
      isGrabbed.current = false;
      rb.setBodyType(0, true); // Dynamic
    }

    // Selection floating animation
    if (meshRef.current) {
      meshRef.current.position.y = object.isSelected && !isGrabbed.current
        ? Math.sin(performance.now() / 500) * 0.04
        : 0;
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
    >
      <group userData={{ objectId: object.id }}>
        <mesh
          ref={meshRef}
          scale={object.scale}
          castShadow
          receiveShadow
          onPointerOver={(e) => { e.stopPropagation(); onPointerOver?.(); }}
          onPointerOut={(e) => { e.stopPropagation(); onPointerOut?.(); }}
          onClick={(e) => { e.stopPropagation(); onClick?.(); }}
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
