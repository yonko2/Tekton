interface HandVisualizationProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isTracking: boolean;
}

export function HandVisualization({ videoRef, canvasRef, isTracking }: HandVisualizationProps) {
  return (
    <div className="webcam-container">
      <video
        ref={videoRef}
        className="webcam-video"
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className="hand-canvas"
      />
      {!isTracking && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          textAlign: 'center',
          padding: '10px',
        }}>
          Camera not active
        </div>
      )}
    </div>
  );
}

export default HandVisualization;
