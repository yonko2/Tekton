import { useRef, useEffect } from 'react';
import { useSandbox } from '@/context/SandboxContext';
import type { ShapeType, Vector3Tuple } from '@/types';

interface TileDef {
  shape: ShapeType;
  color: string;
  label: string;
}

const TILES: TileDef[] = [
  { shape: 'cube', color: '#e53935', label: 'Cube' },
  { shape: 'sphere', color: '#1e88e5', label: 'Sphere' },
  { shape: 'cylinder', color: '#43a047', label: 'Cylinder' },
];

export function SpawnBar() {
  const { state, addObject } = useSandbox();
  const tileRefs = useRef<(HTMLDivElement | null)[]>([]);
  const wasPinchingRef = useRef(false);

  const { gesture } = state;

  // Detect pinch start over a tile
  useEffect(() => {
    const pinching = gesture.isPinching;
    const screen = gesture.screenPosition;

    if (pinching && !wasPinchingRef.current && screen) {
      // Convert normalised screen position (0-1) to pixel coordinates
      const px = screen.x * window.innerWidth;
      const py = screen.y * window.innerHeight;

      for (let i = 0; i < TILES.length; i++) {
        const el = tileRefs.current[i];
        if (!el) continue;
        const rect = el.getBoundingClientRect();

        // Add generous padding around the tile for hand-gesture tolerance
        const pad = 20;
        if (
          px >= rect.left - pad &&
          px <= rect.right + pad &&
          py >= rect.top - pad &&
          py <= rect.bottom + pad
        ) {
          const pos: Vector3Tuple = gesture.pointerPosition
            ? [...gesture.pointerPosition]
            : [0, 0.5, 0];
          if (pos[1] < 0.5) pos[1] = 0.5;
          addObject(TILES[i].shape, pos, TILES[i].color);
          break;
        }
      }
    }

    wasPinchingRef.current = pinching;
  }, [gesture.isPinching, gesture.screenPosition, gesture.pointerPosition, addObject]);

  // Compute which tile the pointer is currently hovering (for visual highlight)
  let hoveredIdx = -1;
  const screen = gesture.screenPosition;
  if (screen && !gesture.isPinching) {
    const px = screen.x * window.innerWidth;
    const py = screen.y * window.innerHeight;

    for (let i = 0; i < TILES.length; i++) {
      const el = tileRefs.current[i];
      if (!el) continue;
      const rect = el.getBoundingClientRect();
      const pad = 20;
      if (
        px >= rect.left - pad &&
        px <= rect.right + pad &&
        py >= rect.top - pad &&
        py <= rect.bottom + pad
      ) {
        hoveredIdx = i;
        break;
      }
    }
  }

  return (
    <div className="spawn-bar">
      {TILES.map((tile, i) => (
        <div
          key={tile.shape}
          ref={(el) => {
            tileRefs.current[i] = el;
          }}
          className={`spawn-tile glass${hoveredIdx === i ? ' spawn-tile-hover' : ''}`}
          onClick={() => {
            const pos: Vector3Tuple = gesture.pointerPosition
              ? [...gesture.pointerPosition]
              : [0, 0.5, 0];
            if (pos[1] < 0.5) pos[1] = 0.5;
            addObject(tile.shape, pos, tile.color);
          }}
        >
          <div
            className={`spawn-shape spawn-shape-${tile.shape}`}
            style={{ backgroundColor: tile.color }}
          />
          <span className="spawn-tile-label">{tile.label}</span>
        </div>
      ))}
    </div>
  );
}
