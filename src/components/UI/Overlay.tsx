import type { RefObject } from 'react';
import { useSandbox } from '@/context/SandboxContext';
import { HandVisualization } from './HandVisualization';
import { StatusPanel } from './StatusPanel';
import { VoiceIndicator } from './VoiceIndicator';
import { Instructions } from './Instructions';

interface OverlayProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  isTracking: boolean;
  isVoiceListening: boolean;
}

export function Overlay({
  videoRef,
  canvasRef,
  isTracking,
  isVoiceListening,
}: OverlayProps) {
  const { state } = useSandbox();

  const selectedObj =
    state.selectedObjectId
      ? state.objects.find((o) => o.id === state.selectedObjectId) ?? null
      : null;

  return (
    <div className="overlay-container">
      <StatusPanel
        gesture={state.gesture.currentGesture}
        objectCount={state.objects.length}
        selectedObject={selectedObj}
        isTracking={isTracking}
      />

      <Instructions />

      <HandVisualization
        videoRef={videoRef}
        canvasRef={canvasRef}
        isTracking={isTracking}
      />

      <VoiceIndicator
        isListening={isVoiceListening}
        lastCommand={state.voice.lastCommand}
      />
    </div>
  );
}
