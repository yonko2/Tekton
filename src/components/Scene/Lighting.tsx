import { useRef } from 'react';
import * as THREE from 'three';
import { SCENE_CONFIG } from '@/types';

interface LightingProps {
  showHelpers?: boolean;
}

export function Lighting({ showHelpers: _showHelpers = false }: LightingProps) {
  const { lighting } = SCENE_CONFIG;
  const directionalLightRef = useRef<THREE.DirectionalLight>(null);

  return (
    <>
      {/* Ambient light for overall illumination */}
      <ambientLight intensity={lighting.ambient.intensity} />

      {/* Main directional light with shadows */}
      <directionalLight
        ref={directionalLightRef}
        position={lighting.directional.position}
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
      <directionalLight
        position={[-5, 8, -5]}
        intensity={0.3}
      />

      {/* Hemisphere light for sky/ground color variation */}
      <hemisphereLight
        args={['#87ceeb', '#505050', 0.3]}
      />
    </>
  );
}

export default Lighting;
