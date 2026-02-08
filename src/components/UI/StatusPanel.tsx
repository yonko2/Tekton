import type { GestureType, SceneObject } from '@/types';

interface StatusPanelProps {
  gesture: GestureType;
  objectCount: number;
  selectedObject: SceneObject | null;
  isTracking: boolean;
}

export function StatusPanel({
  gesture,
  objectCount,
  selectedObject,
  isTracking,
}: StatusPanelProps) {
  return (
    <div className="status-panel glass">
      <div className="status-row">
        <span className="status-label">Tracking</span>
        <span className={`status-dot ${isTracking ? 'on' : 'off'}`} />
      </div>
      <div className="status-row">
        <span className="status-label">Gesture</span>
        <span className="status-value">{gesture}</span>
      </div>
      <div className="status-row">
        <span className="status-label">Objects</span>
        <span className="status-value">{objectCount}</span>
      </div>
      {selectedObject && (
        <div className="status-row">
          <span className="status-label">Selected</span>
          <span className="status-value" style={{ color: selectedObject.color }}>
            {selectedObject.type}
          </span>
        </div>
      )}
    </div>
  );
}
