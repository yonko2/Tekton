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
    zoom: 1,
    gravity: 2200,
    floorHeight: 140,
    bounciness: 0.45,
    friction: 0.55,
  }));

  useEffect(() => {
    function clamp(v: number, min: number, max: number) {
      return Math.max(min, Math.min(max, v));
    }

    function onWheel(e: WheelEvent) {
      // Don't hijack scroll when the user is interacting with the panel.
      const target = e.target as HTMLElement | null;
      if (target?.closest?.('#panel')) return;

      // Prevent page scroll (we're using wheel for zoom)
      e.preventDefault();

      const direction = Math.sign(e.deltaY);
      // deltaY > 0 means scrolling down (zoom out)
      const step = 0.05;
      const nextZoom = clamp(settings.zoom - direction * step, 0.5, 1.6);

      if (nextZoom === settings.zoom) return;

      const next: SettingsModel = { ...settings, zoom: nextZoom };
      setSettings(next);
      engine.applySettings(next);
    }

    window.addEventListener('wheel', onWheel, { passive: false });
    return () => window.removeEventListener('wheel', onWheel as any);
  }, [engine, settings]);

  return (
    <>
      <Loader visible={loaderVisible} />
      <OverlayHUD />

      <SettingsPanel
        open={panelOpen}
        settings={settings}
        onToggleOpen={() => setPanelOpen((v) => !v)}
        onSpawn={() => engine.addRandomBody()}
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
        onFirstResults={() => setLoaderVisible(false)}
        initialSettings={settings}
      />
    </>
  );
}
