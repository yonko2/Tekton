import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { Suspense, useEffect } from 'react';
import * as THREE from 'three';
import { Ground } from './Ground';
import { Lighting } from './Lighting';
import { SceneObjects } from './SceneObjects';
import { Pointer3D } from '../UI/Pointer';
import { CameraController } from '../Camera/CameraController';
import { SCENE_CONFIG } from '@/types';

// Component to capture and expose camera reference
function CameraCapture({ onCameraReady }: { onCameraReady?: (camera: THREE.Camera) => void }) {
  const { camera } = useThree();
  
  useEffect(() => {
    onCameraReady?.(camera);
  }, [camera, onCameraReady]);
  
  return null;
}

function SceneContent({ onCameraReady }: { onCameraReady?: (camera: THREE.Camera) => void }) {
  return (
    <>
      <CameraCapture onCameraReady={onCameraReady} />
      <Lighting />
      <Ground />
      <SceneObjects />
      <Pointer3D />
      <CameraController />
      
      {/* Grid helper for visual reference */}
      <gridHelper args={[20, 20, '#666666', '#444444']} position={[0, 0.01, 0]} />
    </>
  );
}

interface SceneProps {
  onCameraReady?: (camera: THREE.Camera) => void;
}

export function Scene({ onCameraReady }: SceneProps) {
  const { camera } = SCENE_CONFIG;

  return (
    <div className="canvas-container">
      <Canvas
        shadows
        camera={{
          position: camera.position,
          fov: camera.fov,
          near: 0.1,
          far: 1000,
        }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1,
        }}
        onCreated={({ gl }) => {
          gl.shadowMap.enabled = true;
          gl.shadowMap.type = THREE.PCFSoftShadowMap;
        }}
      >
        <color attach="background" args={['#1a1a2e']} />
        <fog attach="fog" args={['#1a1a2e', 15, 35]} />
        
        <Suspense fallback={null}>
          <SceneContent onCameraReady={onCameraReady} />
          <Environment preset="city" />
        </Suspense>

        {/* Default orbit controls - will be overridden by gesture controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={30}
          maxPolarAngle={Math.PI / 2 - 0.1}
        />
      </Canvas>
    </div>
  );
}

export default Scene;
