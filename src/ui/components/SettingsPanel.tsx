import React, { useEffect, useMemo, useState } from 'react';

export type SettingsModel = {
  zoom: number;
  gravity: number;
  floorHeight: number;
  bounciness: number;
  friction: number;
};

type Props = {
  open: boolean;
  settings: SettingsModel;
  onToggleOpen: () => void;
  onSpawn: () => void;
  onReset: () => void;
  onChange: (next: SettingsModel) => void;
};

export default function SettingsPanel({
  open,
  settings,
  onToggleOpen,
  onSpawn,
  onReset,
  onChange,
}: Props) {
  // Keep internal draft so sliders feel immediate and we preserve the existing behavior.
  const [draft, setDraft] = useState(settings);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  const contentStyle = useMemo<React.CSSProperties>(
    () => ({ display: open ? 'grid' : 'none' }),
    [open]
  );

  function set<K extends keyof SettingsModel>(key: K, value: number) {
    const next = { ...draft, [key]: value };
    setDraft(next);
    onChange(next);
  }

  return (
    <div id="panel" aria-label="Settings panel">
      <div className="header">
        <div style={{ display: 'grid', gap: 2 }}>
          <strong>Settings</strong>
          <span className="hint">Scroll anywhere to zoom</span>
        </div>
        <button id="panel-toggle" type="button" onClick={onToggleOpen} aria-expanded={open}>
          {open ? 'Hide' : 'Show'}
        </button>
      </div>

      <div className="content" id="panel-content" style={contentStyle}>
        <div className="buttons">
          <button id="spawn-one" type="button" onClick={onSpawn}>
            Spawn
          </button>
          <button id="reset" type="button" onClick={onReset}>
            Reset
          </button>
        </div>

        <div className="row">
          <label htmlFor="zoom">Zoom</label>
          <div className="control">
            <input
              id="zoom"
              type="range"
              min={0.5}
              max={1.6}
              step={0.05}
              value={draft.zoom}
              onChange={(e) => set('zoom', Number(e.target.value))}
            />
            <span className="value">{draft.zoom.toFixed(2)}</span>
          </div>
        </div>

        <div className="row">
          <label htmlFor="gravity">Gravity</label>
          <div className="control">
            <input
              id="gravity"
              type="range"
              min={200}
              max={4200}
              step={50}
              value={draft.gravity}
              onChange={(e) => set('gravity', Number(e.target.value))}
            />
            <span className="value">{Math.round(draft.gravity)}</span>
          </div>
        </div>

        <div className="row">
          <label htmlFor="floor">Floor height</label>
          <div className="control">
            <input
              id="floor"
              type="range"
              min={60}
              max={320}
              step={5}
              value={draft.floorHeight}
              onChange={(e) => set('floorHeight', Number(e.target.value))}
            />
            <span className="value">{Math.round(draft.floorHeight)}</span>
          </div>
        </div>

        <div className="row">
          <label htmlFor="bounciness">Bounciness</label>
          <div className="control">
            <input
              id="bounciness"
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={draft.bounciness}
              onChange={(e) => set('bounciness', Number(e.target.value))}
            />
            <span className="value">{draft.bounciness.toFixed(2)}</span>
          </div>
        </div>

        <div className="row">
          <label htmlFor="friction">Friction</label>
          <div className="control">
            <input
              id="friction"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={draft.friction}
              onChange={(e) => set('friction', Number(e.target.value))}
            />
            <span className="value">{draft.friction.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

