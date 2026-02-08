import { SCENE_CONFIG } from '@/types';

export function Lighting() {
  const { lighting } = SCENE_CONFIG;

  return (
    <>
      {/* Ambient fill */}
      <ambientLight intensity={lighting.ambient.intensity} />

      {/* Main directional light with shadows */}
      <directionalLight
        position={lighting.directional.position as [number, number, number]}
        intensity={lighting.directional.intensity}
        castShadow={lighting.directional.castShadow}
        shadow-mapSize-width={lighting.directional.shadowMapSize}
        shadow-mapSize-height={lighting.directional.shadowMapSize}
        shadow-camera-far={50}
        shadow-camera-left={-15}
        shadow-camera-right={15}
        shadow-camera-top={15}
        shadow-camera-bottom={-15}
        shadow-bias={-0.0001}
      />

      {/* Fill light from opposite side */}
      <directionalLight position={[-5, 8, -5]} intensity={0.3} />

      {/* Sky / ground colour variation */}
      <hemisphereLight args={['#87ceeb', '#3a3a4a', 0.3]} />
    </>
  );
}
