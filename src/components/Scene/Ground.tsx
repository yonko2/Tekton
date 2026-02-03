import { SCENE_CONFIG } from '@/types';

export function Ground() {
  const { ground } = SCENE_CONFIG;

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, 0, 0]}
      receiveShadow={ground.receiveShadow}
    >
      <planeGeometry args={ground.size} />
      <meshStandardMaterial color={ground.color} />
    </mesh>
  );
}

export default Ground;
