# 3D Gesture Sandbox

A 3D sandbox application where users can create, manipulate, and stack geometric objects using hand gestures and voice commands.

## Features

- **Hand Gesture Controls**: Use your webcam to track hand gestures for manipulating objects
  - **Point** - Move the 3D pointer and hover over objects
  - **Pinch** - Grab and move objects in 3D space
  - **Open Palm** - Pan the camera view
  - **Fist** - Reset camera to default position
  - **Two-Finger Pinch** - Scale the selected object

- **Voice Commands**: Control the scene with your voice
  - "Create [shape]" - Create a new object at the pointer position
  - "Create [color] [shape]" - Create a colored object
  - "Delete" - Remove the selected object
  - "Clear all" - Remove all objects from the scene

- **Available Shapes**: cube, sphere, cylinder, cone, torus, pyramid
- **Available Colors**: red, blue, green, yellow, orange, purple, white, black

## Tech Stack

- React 18 with TypeScript
- Three.js via @react-three/fiber and @react-three/drei
- MediaPipe Hands for hand tracking
- Web Speech API for voice recognition
- Vite for build tooling

## Getting Started

### Prerequisites

- Node.js 18+ 
- A modern browser with WebGL support
- Webcam for hand tracking
- Microphone for voice commands (optional)

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build

```bash
npm run build
```

## Usage

1. **Grant Permissions**: When the app loads, click "Grant Permissions" to enable camera and microphone access.

2. **Hand Tracking**: Position your hand in view of the webcam. The app will detect and visualize your hand landmarks.

3. **Create Objects**: Use voice commands like "Create blue cube" or "Create sphere" to add objects to the scene.

4. **Manipulate Objects**:
   - Point at an object to highlight it
   - Pinch to grab and move objects
   - Use two-finger pinch to scale

5. **Navigate the Scene**:
   - Use mouse/trackpad for orbit controls
   - Open palm gesture to pan
   - Fist gesture to reset view

## Browser Support

- Chrome/Edge (recommended) - Full support
- Firefox - Partial voice recognition support
- Safari - Limited MediaPipe support

## License

MIT
