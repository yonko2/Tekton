import { useState } from "react";

export function Instructions() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`instructions-panel glass ${collapsed ? "collapsed" : ""}`}>
      <button
        className="instructions-toggle"
        onClick={() => setCollapsed((c) => !c)}
      >
        {collapsed ? "Show Help" : "Hide Help"}
      </button>

      {!collapsed && (
        <div className="instructions-content">
          <h3>Gestures</h3>
          <ul>
            <li>
              <strong>Point</strong> &mdash; extend index finger to aim the
              pointer
            </li>
            <li>
              <strong>Pinch object</strong> &mdash; thumb + index to grab &amp;
              move
            </li>
            <li>
              <strong>Twist hand</strong> &mdash; rotate grabbed object in view
              plane
            </li>
            <li>
              <strong>Hand closer / farther</strong> &mdash; push / pull object
              depth
            </li>
            <li>
              <strong>Second hand spread</strong> &mdash; scale grabbed object
              up / down
            </li>
            <li>
              <strong>Pinch empty space</strong> &mdash; orbit the camera
            </li>
            <li>
              <strong>Pinch spawn tile</strong> &mdash; spawn an object from the
              top bar
            </li>
            <li>
              <strong>Mouse</strong> &mdash; left-drag to orbit, scroll to zoom
            </li>
          </ul>

          <h3>Voice Commands</h3>
          <ul>
            <li>
              <strong>Verbs:</strong> create, make, add, spawn, delete, remove,
              clear all, clear everything
            </li>
            <li>
              <strong>Shapes:</strong> cube, sphere, cylinder, pyramid
            </li>
            <li>
              <strong>Colors:</strong> red, blue, green, yellow, orange, purple,
              white, gray
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
