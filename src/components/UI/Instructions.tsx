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
          <h3>Gestures</h3>
          <ul>
            <li>
              <strong>Point</strong> &mdash; extend index finger to aim the pointer
            </li>
            <li>
              <strong>Pinch object</strong> &mdash; thumb + index to grab &amp; move
            </li>
            <li>
              <strong>Twist hand</strong> &mdash; rotate grabbed object in view plane
            </li>
            <li>
              <strong>Hand closer / farther</strong> &mdash; push / pull object depth
            </li>
            <li>
              <strong>Second hand spread</strong> &mdash; scale grabbed object up / down
            </li>
            <li>
              <strong>Pinch empty space</strong> &mdash; orbit the camera
            </li>
            <li>
              <strong>Pinch spawn tile</strong> &mdash; spawn an object from the top bar
            </li>
            <li>
              <strong>Release quickly</strong> &mdash; throw the object
            </li>
            <li>
              <strong>Mouse</strong> &mdash; left-drag to orbit, scroll to zoom
            </li>
          </ul>

          <h3>Voice Commands</h3>
          <p className="voice-help-intro">
            Say a <strong>verb</strong>, optional <strong>color</strong>, and a <strong>shape</strong>:
          </p>
          <ul>
            <li>
              <strong>Verbs:</strong> create, make, add, spawn
            </li>
            <li>
              <strong>Shapes:</strong> cube (box), sphere, cylinder, cone, torus (ring/donut), pyramid
            </li>
            <li>
              <strong>Colors:</strong> red, blue, green, yellow, orange, purple, white, gray
            </li>
          </ul>
          <p className="voice-help-examples">Examples:</p>
          <ul>
            <li>&ldquo;Create red cube&rdquo;</li>
            <li>&ldquo;Make blue sphere&rdquo;</li>
            <li>&ldquo;Spawn big orange torus&rdquo;</li>
            <li>&ldquo;Add green cylinder&rdquo;</li>
          </ul>
          <p className="voice-help-examples">Other commands:</p>
          <ul>
            <li>&ldquo;Delete&rdquo; / &ldquo;Remove&rdquo; &mdash; removes selected object</li>
            <li>&ldquo;Clear all&rdquo; / &ldquo;Clear everything&rdquo; &mdash; removes all</li>
          </ul>
        </div>
      )}
    </div>
  );
}
