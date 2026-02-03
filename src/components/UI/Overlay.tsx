import { useSandbox } from '@/context/SandboxContext';
import { HandVisualization } from './HandVisualization';
import { VoiceIndicator } from './VoiceIndicator';
import { StatusPanel } from './StatusPanel';
import { Instructions } from './Instructions';

interface OverlayProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isTracking: boolean;
  isVoiceListening: boolean;
}

export function Overlay({ videoRef, canvasRef, isTracking, isVoiceListening }: OverlayProps) {
  const { state } = useSandbox();

  return (
    <div className="overlay-container">
      {/* Status Panel */}
      <StatusPanel
        gesture={state.gesture.currentGesture}
        objectCount={state.objects.length}
        selectedObject={state.selectedObjectId ? state.objects.find(o => o.id === state.selectedObjectId) ?? null : null}
        isTracking={isTracking}
      />

      {/* Instructions Panel */}
      <Instructions />

      {/* Webcam with hand visualization */}
      <HandVisualization
        videoRef={videoRef}
        canvasRef={canvasRef}
        isTracking={isTracking}
      />

      {/* Voice indicator */}
      <VoiceIndicator
        isListening={isVoiceListening}
        lastCommand={state.voice.lastCommand}
      />
    </div>
  );
}

export default Overlay;
