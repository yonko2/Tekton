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
  const [isCameraReady, setIsCameraReady] = useState(false);
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
    // Store initial scale for scaling
    const object = state.objects.find(o => o.id === objectId);
    if (object) {
      initialScaleRef.current = [...object.scale];
    }
    console.log('Grabbed object:', objectId);
  }, [state.objects]);

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

  // Store initial scale when grab starts
  const initialScaleRef = useRef<Vector3Tuple>([1, 1, 1]);
  
  const handleScale = useCallback((objectId: string, scaleFactor: number) => {
    // Scale factor is already computed relative to initial
    const newScale: Vector3Tuple = [
      initialScaleRef.current[0] * scaleFactor,
      initialScaleRef.current[1] * scaleFactor,
      initialScaleRef.current[2] * scaleFactor,
    ];
    updateObjectScale(objectId, newScale);
  }, [updateObjectScale]);

  const { processHands } = useGestureRecognition({
    onGrab: handleGrab,
    onRelease: handleRelease,
    onMove: handleMove,
    onScale: handleScale,
  });

  // Store processHands in ref to avoid effect re-runs
  const processHandsRef = useRef(processHands);
  useEffect(() => {
    processHandsRef.current = processHands;
  }, [processHands]);

  // Voice recognition
  const { isListening: isVoiceListening, startListening: startVoice } = useVoiceRecognition();

  // Camera ready callback
  const handleCameraReady = useCallback((camera: THREE.Camera) => {
    console.log('App: Camera ready callback received');
    cameraRef.current = camera;
    setIsCameraReady(true);
  }, []);

  // Request permissions - just mark as ready, tracking starts after render
  const requestPermissions = useCallback(async () => {
    try {
      // Request camera permission (pre-check)
      await navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        // Stop this test stream, we'll start the real one after render
        stream.getTracks().forEach(track => track.stop());
      });
      
      setPermissions(true);
      setIsReady(true);
      setLoading(false);
    } catch (err) {
      console.error('Failed to get permissions:', err);
      setPermissions(false);
      setLoading(false);
    }
  }, [setPermissions, setLoading]);

  // Start tracking after component renders with video element
  useEffect(() => {
    if (isReady && !isTracking && !isHandTrackingLoading) {
      // Small delay to ensure video element is in DOM
      const timer = setTimeout(() => {
        console.log('Starting hand tracking...');
        startTracking();
        startVoice();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isReady, isTracking, isHandTrackingLoading, startTracking, startVoice]);

  // Store hands in a ref to avoid re-running effect on every hand update
  const handsRef = useRef<HandLandmarks[]>([]);
  useEffect(() => {
    handsRef.current = state.handTracking.hands;
  }, [state.handTracking.hands]);

  // Process hand tracking on each frame - only depends on tracking/camera state
  useEffect(() => {
    if (!isTracking || !isCameraReady || !cameraRef.current) {
      return;
    }

    console.log('Hand tracking processing loop started');
    
    const processFrame = () => {
      if (handsRef.current.length > 0 && cameraRef.current) {
        processHandsRef.current(handsRef.current, cameraRef.current);
      }
    };

    const intervalId = setInterval(processFrame, 33); // ~30fps
    return () => {
      console.log('Hand tracking processing loop stopped');
      clearInterval(intervalId);
    };
  }, [isTracking, isCameraReady]);

  // Loading state - only wait for MediaPipe to initialize
  if (isHandTrackingLoading) {
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
