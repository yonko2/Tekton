import { useEffect, useRef, useCallback, useState } from 'react';
import { Hands, Results, NormalizedLandmark } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import type { HandLandmarks } from '@/types';

interface UseHandTrackingOptions {
  onResults: (hands: HandLandmarks[]) => void;
  maxNumHands?: number;
  minDetectionConfidence?: number;
  minTrackingConfidence?: number;
}

interface UseHandTrackingReturn {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isLoading: boolean;
  isTracking: boolean;
  error: string | null;
  startTracking: () => Promise<void>;
  stopTracking: () => void;
}

export function useHandTracking({
  onResults,
  maxNumHands = 2,
  minDetectionConfidence = 0.7,
  minTrackingConfidence = 0.5,
}: UseHandTrackingOptions): UseHandTrackingReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<Hands | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Process MediaPipe results
  const processResults = useCallback((results: Results) => {
    const hands: HandLandmarks[] = [];

    if (results.multiHandLandmarks && results.multiHandedness) {
      for (let i = 0; i < results.multiHandLandmarks.length; i++) {
        const landmarks = results.multiHandLandmarks[i];
        const handedness = results.multiHandedness[i];
        
        hands.push({
          landmarks: landmarks as NormalizedLandmark[],
          handedness: handedness.label as 'Left' | 'Right',
        });
      }
    }

    onResults(hands);

    // Draw landmarks on canvas for visualization
    if (canvasRef.current && videoRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        
        if (results.multiHandLandmarks) {
          for (const landmarks of results.multiHandLandmarks) {
            drawLandmarks(ctx, landmarks, canvasRef.current.width, canvasRef.current.height);
          }
        }
      }
    }
  }, [onResults]);

  // Initialize MediaPipe Hands
  const initializeHands = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        },
      });

      hands.setOptions({
        maxNumHands,
        modelComplexity: 1,
        minDetectionConfidence,
        minTrackingConfidence,
      });

      hands.onResults(processResults);

      await hands.initialize();
      handsRef.current = hands;
      setIsLoading(false);
    } catch (err) {
      console.error('Failed to initialize MediaPipe Hands:', err);
      setError('Failed to initialize hand tracking');
      setIsLoading(false);
    }
  }, [maxNumHands, minDetectionConfidence, minTrackingConfidence, processResults]);

  // Start tracking
  const startTracking = useCallback(async () => {
    if (!videoRef.current || !handsRef.current) {
      setError('Video element or Hands not initialized');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user',
        },
      });

      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      // Set canvas size
      if (canvasRef.current) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
      }

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (handsRef.current && videoRef.current) {
            await handsRef.current.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });

      await camera.start();
      cameraRef.current = camera;
      setIsTracking(true);
    } catch (err) {
      console.error('Failed to start camera:', err);
      setError('Failed to access camera. Please grant camera permissions.');
    }
  }, []);

  // Stop tracking
  const stopTracking = useCallback(() => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsTracking(false);
  }, []);

  // Initialize on mount
  useEffect(() => {
    initializeHands();

    return () => {
      stopTracking();
      if (handsRef.current) {
        handsRef.current.close();
        handsRef.current = null;
      }
    };
  }, [initializeHands, stopTracking]);

  return {
    videoRef,
    canvasRef,
    isLoading,
    isTracking,
    error,
    startTracking,
    stopTracking,
  };
}

// Draw hand landmarks on canvas
function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number
) {
  // Connection pairs for hand skeleton
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17], // Palm
  ];

  // Draw connections
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  
  for (const [start, end] of connections) {
    const startPoint = landmarks[start];
    const endPoint = landmarks[end];
    
    ctx.beginPath();
    ctx.moveTo(startPoint.x * width, startPoint.y * height);
    ctx.lineTo(endPoint.x * width, endPoint.y * height);
    ctx.stroke();
  }

  // Draw landmarks
  for (let i = 0; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    const x = landmark.x * width;
    const y = landmark.y * height;

    // Highlight fingertips
    const isFingertip = [4, 8, 12, 16, 20].includes(i);
    
    ctx.beginPath();
    ctx.arc(x, y, isFingertip ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = isFingertip ? '#ff6600' : '#00ff00';
    ctx.fill();
  }
}

export default useHandTracking;
