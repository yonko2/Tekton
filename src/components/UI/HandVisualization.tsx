import type { RefObject } from 'react';

interface HandVisualizationProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  isTracking: boolean;
}

export function HandVisualization({
  videoRef,
  canvasRef,
  isTracking,
}: HandVisualizationProps) {
  return (
    <div className={`webcam-container ${isTracking ? 'active' : ''}`}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="webcam-video"
      />
      <canvas ref={canvasRef} className="hand-canvas" />
      {!isTracking && (
        <div className="webcam-placeholder">Camera off</div>
      )}
    </div>
  );
}
