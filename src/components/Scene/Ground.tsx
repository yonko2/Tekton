import { RigidBody } from '@react-three/rapier';
import { SCENE_CONFIG } from '@/types';

export function Ground() {
  const { ground } = SCENE_CONFIG;

  return (
    <RigidBody type="fixed" colliders="cuboid" friction={1} restitution={0.1}>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={ground.size} />
        <meshStandardMaterial color={ground.color} roughness={0.9} metalness={0} />
      </mesh>
    </RigidBody>
  );
}
