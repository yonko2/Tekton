import { useState, useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { SandboxProvider, useSandbox } from '@/context/SandboxContext';
import { Scene } from '@/components/Scene/Scene';
import { Overlay } from '@/components/UI/Overlay';
import { useHandTracking } from '@/hooks/useHandTracking';
import { useGestureRecognition } from '@/hooks/useGestureRecognition';
import { useVoiceRecognition } from '@/hooks/useVoiceRecognition';
import type { HandData } from '@/types';


function SandboxApp() {
  const { setHandTracking, setLoading, setPermissions } = useSandbox();

  const [isReady, setIsReady] = useState(false);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);

  
  const handleHandResults = useCallback(
    (hands: HandData[]) => {
      setHandTracking({
        isTracking: hands.length > 0,
        hands,
        primaryHand: hands[0] ?? null,
      });
    },
    [setHandTracking],
  );

  const {
    videoRef,
    canvasRef,
    isLoading: handTrackingLoading,
    isTracking,
    error: handTrackingError,
    startTracking,
  } = useHandTracking({ onResults: handleHandResults });

  
  const { processHands } = useGestureRecognition();
  const processHandsRef = useRef(processHands);
  useEffect(() => {
    processHandsRef.current = processHands;
  }, [processHands]);

  
  const { isListening: isVoiceListening, startListening: startVoice } =
    useVoiceRecognition();

  
  const handleSceneReady = useCallback(
    (camera: THREE.Camera, scene: THREE.Scene) => {
      cameraRef.current = camera;
      sceneRef.current = scene;
    },
    [],
  );

  
  const requestPermissions = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach((t) => t.stop());
      setPermissions(true);
      setIsReady(true);
      setLoading(false);
    } catch (err) {
      console.error('Permission denied:', err);
      setPermissions(false);
      setLoading(false);
    }
  }, [setPermissions, setLoading]);

  
  const hasStartedRef = useRef(false);
  useEffect(() => {
    if (isReady && !hasStartedRef.current) {
      hasStartedRef.current = true;
      const timer = setTimeout(() => {
        startTracking();
        startVoice();
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isReady, startTracking, startVoice]);

  
  const handsRef = useRef<HandData[]>([]);
  const { state } = useSandbox();
  useEffect(() => {
    handsRef.current = state.handTracking.hands;
  }, [state.handTracking.hands]);

  
  useEffect(() => {
    if (!isTracking || !cameraRef.current || !sceneRef.current) return;

    const tick = () => {
      if (handsRef.current.length > 0 && cameraRef.current && sceneRef.current) {
        processHandsRef.current(handsRef.current, cameraRef.current, sceneRef.current);
      }
    };

    const id = setInterval(tick, 33); 
    return () => clearInterval(id);
  }, [isTracking]);

  
  if (!isReady) {
    return (
      <div className="permission-prompt">
        <h2>Camera &amp; Microphone Access</h2>
        <p>
          This app uses your camera for hand-gesture tracking and your microphone
          for voice commands. Grant permissions to get started.
        </p>
        {handTrackingError && (
          <p className="error-text">{handTrackingError}</p>
        )}
        <button onClick={requestPermissions}>Grant Permissions</button>
      </div>
    );
  }

  
  
  return (
    <div className="app-container">
      <Scene onReady={handleSceneReady} />
      <Overlay
        videoRef={videoRef}
        canvasRef={canvasRef}
        isTracking={isTracking}
        isVoiceListening={isVoiceListening}
      />

      {/* Loading overlay – never unmounts the scene/video underneath */}
      {handTrackingLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <div className="loading-text">Initialising hand tracking&hellip;</div>
        </div>
      )}
    </div>
  );
}


export default function App() {
  return (
    <SandboxProvider>
      <SandboxApp />
    </SandboxProvider>
  );
}
