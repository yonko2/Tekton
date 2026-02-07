import { useState } from 'react';

export function Instructions() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`instructions-panel glass ${collapsed ? 'collapsed' : ''}`}>
      <button
        className="instructions-toggle"
        onClick={() => setCollapsed((c) => !c)}
      >
        {collapsed ? 'Show Help' : 'Hide Help'}
      </button>

      {!collapsed && (
        <div className="instructions-content">
          <h3>Controls</h3>
          <ul>
            <li>
              <strong>Point</strong> &mdash; extend index finger to aim the pointer
            </li>
            <li>
              <strong>Pinch object</strong> &mdash; thumb + index to grab &amp; move
            </li>
            <li>
              <strong>Pinch empty space</strong> &mdash; orbit the camera
            </li>
            <li>
              <strong>Release quickly</strong> &mdash; throw the object
            </li>
            <li>
              <strong>Mouse</strong> &mdash; left-drag to orbit, scroll to zoom
            </li>
          </ul>

          <h3>Voice Commands</h3>
          <ul>
            <li>&ldquo;Create red cube&rdquo;</li>
            <li>&ldquo;Make blue sphere&rdquo;</li>
            <li>&ldquo;Add green cylinder&rdquo;</li>
            <li>&ldquo;Delete&rdquo; &mdash; removes selected object</li>
            <li>&ldquo;Clear all&rdquo; &mdash; removes everything</li>
          </ul>
        </div>
      )}
    </div>
  );
}
