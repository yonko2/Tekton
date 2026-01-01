import React, { useEffect } from 'react';
import type { Engine } from '../../engine/engine';
import type { SettingsModel } from './SettingsPanel';

type Props = {
  overlayCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  engine: Engine;
  onFirstResults: () => void;
  initialSettings: SettingsModel;
};

export default function Scene({
  overlayCanvasRef,
  videoRef,
  engine,
  onFirstResults,
  initialSettings,
}: Props) {
  useEffect(() => {
    const overlay = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    // Matches old behavior:
    // - hide old 3D container if it exists
    // - start empty; spawning is via the panel
    engine.mount({ overlay, video, onFirstResults });
    return () => {
      engine.unmount();
    };
  }, [engine, onFirstResults, overlayCanvasRef, videoRef]);

  // Apply settings without remounting the MediaPipe graph.
  useEffect(() => {
    engine.applySettings(initialSettings);
  }, [engine, initialSettings]);

  return (
    <>
      <canvas id="overlay" ref={overlayCanvasRef} />
      <video id="camera-feed" ref={videoRef} playsInline muted autoPlay />
      <div id="canvas-container" style={{ display: 'none' }} />
    </>
  );
}

