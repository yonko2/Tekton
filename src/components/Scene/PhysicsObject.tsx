import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { RigidBody, type RapierRigidBody } from '@react-three/rapier';
import type { SceneObject as SceneObjectType, ShapeType, Vector3Tuple } from '@/types';
import { rigidBodyRefs } from '@/hooks/useGestureRecognition';
import { grabState } from '@/engine/grabStore';

const BASE_DENSITY = 5;

function computeMass(scale: Vector3Tuple): number {
  const volume = scale[0] * scale[1] * scale[2];
  return Math.max(0.5, volume * BASE_DENSITY);
}


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

  
  const isGrabbed = useRef(false);
  const wireRef = useRef<THREE.Mesh>(null);

  
  const grabStartQuat = useRef(new THREE.Quaternion());

  
  const rbRefStable = useRef<React.RefObject<RapierRigidBody | null>>(rbRef);
  useEffect(() => {
    rigidBodyRefs.set(object.id, rbRefStable.current);
    return () => {
      rigidBodyRefs.delete(object.id);
    };
  }, [object.id]);

  
  useFrame(() => {
    const rb = rbRef.current;
    if (!rb) return;

    const amGrabbed = grabState.objectId === object.id;

    
    if (amGrabbed && !isGrabbed.current) {
      isGrabbed.current = true;
      rb.setBodyType(2, true); 
      rb.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rb.setAngvel({ x: 0, y: 0, z: 0 }, true);

      
      const r = rb.rotation();
      grabStartQuat.current.set(r.x, r.y, r.z, r.w);
    }

    
    if (amGrabbed && isGrabbed.current) {
      const [tx, ty, tz] = grabState.targetPosition;
      rb.setNextKinematicTranslation({ x: tx, y: ty, z: tz });

      
      
      const [ax, ay, az] = grabState.twistAxis;
      const twistQ = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(ax, ay, az),
        grabState.twistAngle,
      );
      const finalQ = twistQ.clone().multiply(grabStartQuat.current);
      rb.setNextKinematicRotation({ x: finalQ.x, y: finalQ.y, z: finalQ.z, w: finalQ.w });

      
      const f = grabState.scaleFactor;
      const sx = object.scale[0] * f;
      const sy = object.scale[1] * f;
      const sz = object.scale[2] * f;
      if (meshRef.current) meshRef.current.scale.set(sx, sy, sz);
      if (wireRef.current) wireRef.current.scale.set(sx * 1.06, sy * 1.06, sz * 1.06);
    }

    
    if (amGrabbed && grabState.pendingRelease) {
      isGrabbed.current = false;
      rb.setBodyType(0, true); 
      const [vx, vy, vz] = grabState.releaseVelocity;
      rb.setLinvel({ x: vx, y: vy, z: vz }, true);

      
      grabState.objectId = null;
      grabState.twistAngle = 0;
      grabState.twistAxis = [0, 0, -1];
      grabState.scaleFactor = 1;
      grabState.pendingRelease = false;
      grabState.releaseVelocity = [0, 0, 0];
    }

    
    if (!amGrabbed && isGrabbed.current) {
      isGrabbed.current = false;
      rb.setBodyType(0, true); 
    }

    
    if (!isGrabbed.current && meshRef.current) {
      meshRef.current.scale.set(object.scale[0], object.scale[1], object.scale[2]);
      if (wireRef.current) {
        wireRef.current.scale.set(
          object.scale[0] * 1.06,
          object.scale[1] * 1.06,
          object.scale[2] * 1.06,
        );
      }
    }

    
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
      friction={1.2}
      restitution={0.05}
      mass={computeMass(object.scale)}
      linearDamping={1.5}
      angularDamping={2.0}
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
            ref={wireRef}
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
