import { useEffect, useRef, useCallback, useState } from 'react';
import type { NormalizedLandmark } from '@mediapipe/hands';
import type { HandLandmarks } from '@/types';

// Dynamic import types
interface MediaPipeHands {
  setOptions: (options: {
    maxNumHands: number;
    modelComplexity: number;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
  }) => void;
  onResults: (callback: (results: MediaPipeResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
}

interface MediaPipeResults {
  multiHandLandmarks?: NormalizedLandmark[][];
  multiHandedness?: { label: string }[];
}

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

// Load MediaPipe from CDN with specific version
async function loadMediaPipeHands(): Promise<new (config: { locateFile: (file: string) => string }) => MediaPipeHands> {
  return new Promise((resolve, reject) => {
    // Check if already loaded
    if ((window as any).Hands) {
      resolve((window as any).Hands);
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.min.js';
    script.crossOrigin = 'anonymous';
    
    script.onload = () => {
      // Give it a moment to register
      setTimeout(() => {
        if ((window as any).Hands) {
          resolve((window as any).Hands);
        } else {
          reject(new Error('MediaPipe Hands not found after script load'));
        }
      }, 100);
    };
    
    script.onerror = () => reject(new Error('Failed to load MediaPipe Hands script'));
    document.head.appendChild(script);
  });
}

export function useHandTracking({
  onResults,
  maxNumHands = 2,
  minDetectionConfidence = 0.7,
  minTrackingConfidence = 0.5,
}: UseHandTrackingOptions): UseHandTrackingReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handsRef = useRef<MediaPipeHands | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onResultsRef = useRef(onResults);
  const isTrackingRef = useRef(false);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Keep onResults ref up to date
  useEffect(() => {
    onResultsRef.current = onResults;
  }, [onResults]);

  // Keep isTracking ref in sync
  useEffect(() => {
    isTrackingRef.current = isTracking;
  }, [isTracking]);

  // Initialize MediaPipe Hands on mount
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        console.log('Loading MediaPipe Hands...');
        const Hands = await loadMediaPipeHands();
        
        if (!mounted) return;
        console.log('MediaPipe Hands loaded, initializing...');
        
        const hands = new Hands({
          locateFile: (file: string) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands,
          modelComplexity: 1,
          minDetectionConfidence,
          minTrackingConfidence,
        });

        hands.onResults((results: MediaPipeResults) => {
          const handResults: HandLandmarks[] = [];

          if (results.multiHandLandmarks && results.multiHandedness) {
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
              const landmarks = results.multiHandLandmarks[i];
              const handedness = results.multiHandedness[i];
              
              handResults.push({
                landmarks: landmarks as NormalizedLandmark[],
                handedness: handedness.label as 'Left' | 'Right',
              });
            }
          }

          onResultsRef.current(handResults);

          // Draw landmarks on canvas for visualization
          if (canvasRef.current) {
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
        });

        handsRef.current = hands;
        
        if (mounted) {
          console.log('MediaPipe Hands initialized successfully');
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Failed to initialize MediaPipe Hands:', err);
        if (mounted) {
          setError('Failed to initialize hand tracking: ' + (err as Error).message);
          setIsLoading(false);
        }
      }
    }

    init();

    return () => {
      mounted = false;
      if (handsRef.current) {
        handsRef.current.close();
        handsRef.current = null;
      }
    };
  }, [maxNumHands, minDetectionConfidence, minTrackingConfidence]);

  // Detection loop
  const runDetection = useCallback(async () => {
    if (!handsRef.current || !videoRef.current || !isTrackingRef.current) return;

    if (videoRef.current.readyState >= 2) {
      try {
        await handsRef.current.send({ image: videoRef.current });
      } catch (e) {
        // Ignore send errors during cleanup
      }
    }

    if (isTrackingRef.current) {
      animationFrameRef.current = requestAnimationFrame(runDetection);
    }
  }, []);

  // Start tracking
  const startTracking = useCallback(async () => {
    console.log('startTracking called', { 
      handsReady: !!handsRef.current, 
      videoReady: !!videoRef.current 
    });

    if (!handsRef.current) {
      console.warn('Hand tracking not initialized yet, retrying in 500ms...');
      setTimeout(() => startTracking(), 500);
      return;
    }

    if (!videoRef.current) {
      console.warn('Video element not available, retrying in 500ms...');
      setTimeout(() => startTracking(), 500);
      return;
    }

    try {
      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      });

      console.log('Camera stream obtained');
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      
      await new Promise<void>((resolve) => {
        if (videoRef.current) {
          videoRef.current.onloadedmetadata = () => {
            console.log('Video metadata loaded');
            resolve();
          };
        }
      });
      
      await videoRef.current.play();
      console.log('Video playing');

      // Set canvas size
      if (canvasRef.current) {
        canvasRef.current.width = videoRef.current.videoWidth || 640;
        canvasRef.current.height = videoRef.current.videoHeight || 480;
      }

      setIsTracking(true);
      isTrackingRef.current = true;
      
      console.log('Hand tracking started successfully');
      // Start detection loop
      runDetection();
    } catch (err) {
      console.error('Failed to start camera:', err);
      setError('Failed to access camera. Please grant camera permissions.');
    }
  }, [runDetection]);

  // Stop tracking
  const stopTracking = useCallback(() => {
    isTrackingRef.current = false;
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsTracking(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTracking();
    };
  }, [stopTracking]);

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
