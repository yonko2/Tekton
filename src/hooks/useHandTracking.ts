import { useRef, useState, useCallback, useEffect } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision';
import type { HandData, NormalizedLandmark } from '@/types';
import { HAND_LANDMARKS } from '@/types';


const HAND_CONNECTIONS: [number, number][] = [
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.THUMB_CMC],
  [HAND_LANDMARKS.THUMB_CMC, HAND_LANDMARKS.THUMB_MCP],
  [HAND_LANDMARKS.THUMB_MCP, HAND_LANDMARKS.THUMB_IP],
  [HAND_LANDMARKS.THUMB_IP, HAND_LANDMARKS.THUMB_TIP],
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.INDEX_MCP],
  [HAND_LANDMARKS.INDEX_MCP, HAND_LANDMARKS.INDEX_PIP],
  [HAND_LANDMARKS.INDEX_PIP, HAND_LANDMARKS.INDEX_DIP],
  [HAND_LANDMARKS.INDEX_DIP, HAND_LANDMARKS.INDEX_TIP],
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.MIDDLE_MCP],
  [HAND_LANDMARKS.MIDDLE_MCP, HAND_LANDMARKS.MIDDLE_PIP],
  [HAND_LANDMARKS.MIDDLE_PIP, HAND_LANDMARKS.MIDDLE_DIP],
  [HAND_LANDMARKS.MIDDLE_DIP, HAND_LANDMARKS.MIDDLE_TIP],
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.RING_MCP],
  [HAND_LANDMARKS.RING_MCP, HAND_LANDMARKS.RING_PIP],
  [HAND_LANDMARKS.RING_PIP, HAND_LANDMARKS.RING_DIP],
  [HAND_LANDMARKS.RING_DIP, HAND_LANDMARKS.RING_TIP],
  [HAND_LANDMARKS.WRIST, HAND_LANDMARKS.PINKY_MCP],
  [HAND_LANDMARKS.PINKY_MCP, HAND_LANDMARKS.PINKY_PIP],
  [HAND_LANDMARKS.PINKY_PIP, HAND_LANDMARKS.PINKY_DIP],
  [HAND_LANDMARKS.PINKY_DIP, HAND_LANDMARKS.PINKY_TIP],
  [HAND_LANDMARKS.INDEX_MCP, HAND_LANDMARKS.MIDDLE_MCP],
  [HAND_LANDMARKS.MIDDLE_MCP, HAND_LANDMARKS.RING_MCP],
  [HAND_LANDMARKS.RING_MCP, HAND_LANDMARKS.PINKY_MCP],
];

interface UseHandTrackingOptions {
  onResults: (hands: HandData[]) => void;
}

export function useHandTracking({ onResults }: UseHandTrackingOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const onResultsRef = useRef(onResults);
  onResultsRef.current = onResults;

  const [isLoading, setIsLoading] = useState(false);
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  
  
  
  const isTrackingRef = useRef(false);
  const isLoadingRef = useRef(false);

  
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  
  const drawLandmarks = useCallback(
    (results: HandLandmarkerResult) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const hand of results.landmarks) {
        
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
          const pA = hand[a];
          const pB = hand[b];
          ctx.beginPath();
          ctx.moveTo(pA.x * canvas.width, pA.y * canvas.height);
          ctx.lineTo(pB.x * canvas.width, pB.y * canvas.height);
          ctx.stroke();
        }

        
        for (const lm of hand) {
          ctx.fillStyle = '#ff0066';
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    },
    [],
  );

  
  
  
  const startTracking = useCallback(async () => {
    if (isTrackingRef.current || isLoadingRef.current) return;

    isLoadingRef.current = true;
    setIsLoading(true);
    setError(null);

    try {
      
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      );

      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      landmarkerRef.current = landmarker;

      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
      });
      streamRef.current = stream;

      
      const video = videoRef.current;
      if (!video) throw new Error('Video element not mounted');
      video.srcObject = stream;
      await video.play();

      isTrackingRef.current = true;
      isLoadingRef.current = false;
      setIsTracking(true);
      setIsLoading(false);

      
      let lastTime = -1;
      const detect = () => {
        if (!video || video.readyState < 2) {
          rafRef.current = requestAnimationFrame(detect);
          return;
        }

        const now = performance.now();
        if (now === lastTime) {
          rafRef.current = requestAnimationFrame(detect);
          return;
        }
        lastTime = now;

        const results = landmarker.detectForVideo(video, now);

        
        const hands: HandData[] = (results.landmarks ?? []).map(
          (lm: NormalizedLandmark[], i: number) => ({
            landmarks: lm,
            worldLandmarks: results.worldLandmarks?.[i] ?? lm,
            handedness:
              (results.handednesses?.[i]?.[0]?.categoryName as 'Left' | 'Right') ?? 'Right',
          }),
        );

        onResultsRef.current(hands);
        drawLandmarks(results);

        rafRef.current = requestAnimationFrame(detect);
      };

      rafRef.current = requestAnimationFrame(detect);
    } catch (err) {
      console.error('Hand tracking init failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      isLoadingRef.current = false;
      setIsLoading(false);
    }
  }, [drawLandmarks]); 

  
  const stopTracking = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    setIsTracking(false);
  }, []);

  return { videoRef, canvasRef, isLoading, isTracking, error, startTracking, stopTracking };
}
