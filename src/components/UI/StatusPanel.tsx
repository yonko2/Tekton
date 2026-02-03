import type { GestureType, SceneObject } from '@/types';

interface StatusPanelProps {
  gesture: GestureType;
  objectCount: number;
  selectedObject: SceneObject | null;
  isTracking: boolean;
}

const gestureLabels: Record<GestureType, string> = {
  none: 'None',
  point: 'Pointing',
  pinch: 'Pinch',
};

export function StatusPanel({ gesture, objectCount, selectedObject, isTracking }: StatusPanelProps) {
  return (
    <div className="status-panel">
      <h3>Status</h3>
      
      <div className="status-item">
        <span className="label">Hand Tracking:</span>
        <span className="value" style={{ color: isTracking ? '#4caf50' : '#f44336' }}>
          {isTracking ? 'Active' : 'Inactive'}
        </span>
      </div>

      <div className="status-item">
        <span className="label">Gesture:</span>
        <span className="value">
          <span className="gesture-badge">{gestureLabels[gesture]}</span>
        </span>
      </div>

      <div className="status-item">
        <span className="label">Objects:</span>
        <span className="value">{objectCount}</span>
      </div>

      {selectedObject && (
        <>
          <div className="status-item">
            <span className="label">Selected:</span>
            <span className="value" style={{ textTransform: 'capitalize' }}>
              {selectedObject.type}
            </span>
          </div>
          <div className="status-item">
            <span className="label">Color:</span>
            <span 
              className="value"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <span
                style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '2px',
                  backgroundColor: selectedObject.color,
                  display: 'inline-block',
                }}
              />
              {selectedObject.color}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

export default StatusPanel;
