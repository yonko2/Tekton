export function Instructions() {
  return (
    <div className="instructions">
      <h3>Controls</h3>
      <ul>
        <li><strong>Point</strong> - Move pointer, hover objects</li>
        <li><strong>Pinch</strong> - Grab and move objects</li>
        <li><strong>Open Palm</strong> - Pan camera view</li>
        <li><strong>Fist</strong> - Reset camera</li>
        <li><strong>Two-Finger Pinch</strong> - Scale selected object</li>
      </ul>
      
      <h3 style={{ marginTop: '16px' }}>Voice Commands</h3>
      <ul>
        <li>"Create [shape]" - Add object</li>
        <li>"Create [color] [shape]"</li>
        <li>"Delete" - Remove selected</li>
        <li>"Clear all" - Remove all</li>
      </ul>
      
      <p style={{ marginTop: '12px', color: '#888', fontSize: '11px' }}>
        Shapes: cube, sphere, cylinder, cone, torus, pyramid<br/>
        Colors: red, blue, green, yellow, orange, purple, white, black
      </p>
    </div>
  );
}

export default Instructions;
