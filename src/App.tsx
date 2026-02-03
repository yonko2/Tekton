import { useState, useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { SandboxProvider, useSandbox } from '@/context/SandboxContext';
import { Scene } from '@/components/Scene';
import { Overlay } from '@/components/UI';
import { useHandTracking } from '@/hooks/useHandTracking';
import { useGestureRecognition } from '@/hooks/useGestureRecognition';
import { useVoiceRecognition } from '@/hooks/useVoiceRecognition';
import { calculateRestingPosition, constrainToGround } from '@/engine/physics';
import type { HandLandmarks, Vector3Tuple } from '@/types';

// Inner component that uses the sandbox context
function SandboxApp() {
  const { state, setHandTracking, updateObjectPosition, updateObjectScale, setLoading, setPermissions } = useSandbox();
  const [isReady, setIsReady] = useState(false);
  const cameraRef = useRef<THREE.Camera | null>(null);

  // Hand tracking setup
  const handleHandResults = useCallback((hands: HandLandmarks[]) => {
    setHandTracking({
      isTracking: hands.length > 0,
      hands,
      primaryHand: hands[0] || null,
    });
  }, [setHandTracking]);

  const {
    videoRef,
    canvasRef,
    isLoading: isHandTrackingLoading,
    isTracking,
    error: handTrackingError,
    startTracking,
  } = useHandTracking({
    onResults: handleHandResults,
  });

  // Gesture recognition with callbacks
  const handleGrab = useCallback((objectId: string, _position: Vector3Tuple) => {
    console.log('Grabbed object:', objectId);
  }, []);

  const handleRelease = useCallback((objectId: string, position: Vector3Tuple) => {
    // Calculate resting position (for stacking)
    const object = state.objects.find(o => o.id === objectId);
    if (object) {
      const restingPosition = calculateRestingPosition(
        position,
        object,
        state.objects.filter(o => o.id !== objectId)
      );
      const constrainedPosition = constrainToGround(restingPosition);
      updateObjectPosition(objectId, constrainedPosition);
    }
  }, [state.objects, updateObjectPosition]);

  const handleMove = useCallback((objectId: string, position: Vector3Tuple) => {
    const constrainedPosition = constrainToGround(position);
    updateObjectPosition(objectId, constrainedPosition);
  }, [updateObjectPosition]);

  const handleScale = useCallback((objectId: string, scaleFactor: number) => {
    const object = state.objects.find(o => o.id === objectId);
    if (object) {
      const clampedScale = Math.max(0.2, Math.min(3, scaleFactor));
      const newScale: Vector3Tuple = [
        object.scale[0] * clampedScale,
        object.scale[1] * clampedScale,
        object.scale[2] * clampedScale,
      ];
      updateObjectScale(objectId, newScale);
    }
  }, [state.objects, updateObjectScale]);

  const { processHands } = useGestureRecognition({
    onGrab: handleGrab,
    onRelease: handleRelease,
    onMove: handleMove,
    onScale: handleScale,
  });

  // Voice recognition
  const { isListening: isVoiceListening, startListening: startVoice } = useVoiceRecognition();

  // Camera ready callback
  const handleCameraReady = useCallback((camera: THREE.Camera) => {
    cameraRef.current = camera;
  }, []);

  // Request permissions and start tracking
  const requestPermissions = useCallback(async () => {
    try {
      // Request camera permission
      await navigator.mediaDevices.getUserMedia({ video: true });
      setPermissions(true);
      
      // Start hand tracking
      await startTracking();
      
      // Start voice recognition
      startVoice();
      
      setIsReady(true);
      setLoading(false);
    } catch (err) {
      console.error('Failed to get permissions:', err);
      setPermissions(false);
      setLoading(false);
    }
  }, [startTracking, startVoice, setPermissions, setLoading]);

  // Process hand tracking on each frame
  useEffect(() => {
    if (!isTracking || !cameraRef.current) return;

    const processFrame = () => {
      if (state.handTracking.hands.length > 0 && cameraRef.current) {
        processHands(state.handTracking.hands, cameraRef.current);
      }
    };

    const intervalId = setInterval(processFrame, 33); // ~30fps
    return () => clearInterval(intervalId);
  }, [isTracking, state.handTracking.hands, processHands]);

  // Loading state
  if (isHandTrackingLoading || state.isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <div className="loading-text">Loading hand tracking...</div>
      </div>
    );
  }

  // Permission prompt
  if (!state.hasPermissions && !isReady) {
    return (
      <div className="permission-prompt">
        <h2>Camera & Microphone Access Required</h2>
        <p>
          This application uses your camera for hand gesture tracking and your microphone for voice commands.
          Please grant permissions to continue.
        </p>
        {handTrackingError && (
          <p style={{ color: '#f44336' }}>{handTrackingError}</p>
        )}
        <button onClick={requestPermissions}>
          Grant Permissions
        </button>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* 3D Scene with camera ref capture */}
      <Scene onCameraReady={handleCameraReady} />
      
      {/* Overlay UI */}
      <Overlay
        videoRef={videoRef}
        canvasRef={canvasRef}
        isTracking={isTracking}
        isVoiceListening={isVoiceListening}
      />
    </div>
  );
}

// Main App wrapper with provider
function App() {
  return (
    <SandboxProvider>
      <SandboxApp />
    </SandboxProvider>
  );
}

export default App;
