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
    // refs are stable objects; don't include them as dependencies
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine, onFirstResults]);

  // Apply settings without remounting the MediaPipe graph.
  useEffect(() => {
    engine.applySettings(initialSettings);
    // Only re-apply when values change (avoid object-identity churn).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    engine,
    initialSettings.zoom,
    initialSettings.spawnShape,
    initialSettings.gravity,
    initialSettings.floorHeight,
    initialSettings.bounciness,
    initialSettings.friction,
  ]);

  return (
    <>
      <canvas id="overlay" ref={overlayCanvasRef} />
      <video id="camera-feed" ref={videoRef} playsInline muted autoPlay />
      <div id="canvas-container" style={{ display: 'none' }} />
    </>
  );
}

