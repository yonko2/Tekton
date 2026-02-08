import { Suspense, useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { Environment } from '@react-three/drei';
import { Physics } from '@react-three/rapier';
import * as THREE from 'three';
import { Ground } from './Ground';
import { Lighting } from './Lighting';
import { SceneObjects } from './SceneObjects';
import { Pointer3D } from './Pointer3D';
import { CameraController } from '@/components/Camera/CameraController';
import { SCENE_CONFIG } from '@/types';


function SceneCapture({
  onReady,
}: {
  onReady?: (camera: THREE.Camera, scene: THREE.Scene) => void;
}) {
  const { camera, scene } = useThree();
  useEffect(() => {
    onReady?.(camera, scene);
  }, [camera, scene, onReady]);
  return null;
}


function SceneContent({
  onReady,
}: {
  onReady?: (camera: THREE.Camera, scene: THREE.Scene) => void;
}) {
  return (
    <>
      <SceneCapture onReady={onReady} />
      <Lighting />

      <Physics gravity={[0, -9.81, 0]} debug={false}>
        <Ground />
        <SceneObjects />
      </Physics>

      <Pointer3D />
      <CameraController />

      {/* Grid for visual reference */}
      <gridHelper args={[30, 30, '#555555', '#333333']} position={[0, 0.01, 0]} />
    </>
  );
}


interface SceneProps {
  onReady?: (camera: THREE.Camera, scene: THREE.Scene) => void;
}

export function Scene({ onReady }: SceneProps) {
  const cam = SCENE_CONFIG.camera;

  return (
    <div className="canvas-container">
      <Canvas
        shadows
        camera={{ position: cam.position as [number, number, number], fov: cam.fov, near: 0.1, far: 1000 }}
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
        <fog attach="fog" args={['#1a1a2e', 20, 45]} />

        <Suspense fallback={null}>
          <SceneContent onReady={onReady} />
          <Environment preset="city" />
        </Suspense>
      </Canvas>
    </div>
  );
}
