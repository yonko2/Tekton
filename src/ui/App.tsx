import React, { useEffect, useMemo, useRef, useState } from 'react';
import Loader from './components/Loader';
import OverlayHUD from './components/OverlayHUD';
import SettingsPanel, { SettingsModel } from './components/SettingsPanel';
import Scene from './components/Scene';
import { createEngine } from '../engine/engine';

export default function App() {
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const engine = useMemo(() => createEngine(), []);

  const [loaderVisible, setLoaderVisible] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);

  const [settings, setSettings] = useState<SettingsModel>(() => ({
    zoom: 0.9,
    spawnShape: 'mixed',
    gravity: 2200,
    floorHeight: 140,
    bounciness: 0.45,
    friction: 0.55,
  }));

  // IMPORTANT: keep callback identity stable so <Scene /> doesn't unmount/remount
  // the MediaPipe graph on any re-render.
  const handleFirstResults = useMemo(() => () => setLoaderVisible(false), []);

  return (
    <>
      <Loader visible={loaderVisible} />
      <OverlayHUD />

      <SettingsPanel
        open={panelOpen}
        settings={settings}
        onToggleOpen={() => setPanelOpen((v) => !v)}
        onSpawn={() => engine.addBody(settings.spawnShape)}
        onReset={() => engine.resetScene()}
        onChange={(next) => {
          setSettings(next);
          engine.applySettings(next);
        }}
      />

      <Scene
        overlayCanvasRef={overlayCanvasRef}
        videoRef={videoRef}
        engine={engine}
        onFirstResults={handleFirstResults}
        initialSettings={settings}
      />
    </>
  );
}
