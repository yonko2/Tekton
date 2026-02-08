# Tekton 3D Sandbox

A browser-based 3D sandbox application where users interact with geometric objects through **hand gestures** and **voice commands**. Objects are physics-simulated and can be created, grabbed, moved, rotated, scaled, thrown, stacked, and deleted — all without touching a keyboard.

---

## Table of Contents

1. [Functionality](#functionality)
2. [Architecture](#architecture)
3. [Realization](#realization)
4. [Data Model](#data-model)
5. [Used Datasets](#used-datasets)
6. [Configuration](#configuration)
7. [Technologies and Libraries](#technologies-and-libraries)
8. [Problems and Solutions](#problems-and-solutions)
9. [References](#references)

---

## Functionality

### Hand Gesture Interactions

| Gesture | Action |
|---|---|
| **Point** (index finger extended, others curled) | Aim a 3D pointer in the scene |
| **Pinch object** (thumb + index finger together) | Grab and move an object in 3D space |
| **Pinch empty space** | Orbit the camera around the scene |
| **Hand closer / farther** (while pinching object) | Push or pull the object along the camera depth axis |
| **Twist hand** (while pinching object) | Rotate the grabbed object in the camera view plane |
| **Second hand spread/pinch** (while first hand grabs) | Scale the grabbed object up or down |
| **Hand closer / farther** (while pinching empty space) | Zoom the camera in or out |
| **Quick release** | Throw the object with velocity |
| **Pinch over spawn tile** | Spawn a predefined object from the top bar |

### Voice Commands

Voice recognition listens continuously once permissions are granted. Commands follow the pattern:

```
<verb> [color] <shape>
```

- **Verbs**: `create`, `make`, `add`, `spawn`
- **Shapes**: `cube` (or `box`), `sphere`, `cylinder`, `pyramid`
- **Colors**: `red`, `blue`, `green`, `yellow`, `orange`, `purple`, `white`, `gray` (or `grey`)

Additional commands:
- `delete` / `remove` — removes the currently selected object
- `clear all` / `clear everything` — removes all objects from the scene

Objects created by voice are placed at the current pointer position.

### Mouse Fallback

- **Left-drag** — orbit the camera (via OrbitControls)
- **Scroll** — zoom in/out
- **Click spawn tile** — spawn an object

### UI Panels

- **Status Panel** (top-left): shows tracking status, current gesture, object count, selected object
- **Help Panel** (top-right): collapsible panel listing all gestures and voice command parameters
- **Spawn Bar** (top-center): three clickable/pinchable tiles for quick object creation
- **Hand Visualization** (bottom-right): mirrored webcam feed with hand skeleton overlay
- **Voice Indicator** (bottom-left): microphone status and last recognized command

---

## Architecture

The application follows a layered architecture separating concerns into engine logic, React state management, hooks, and presentational components.

```
src/
├── engine/                   # Pure logic (no React)
│   ├── gestures.ts           # Gesture detection, smoothing, screen-to-world projection
│   ├── physics.ts            # Velocity tracking, ground constraints
│   └── grabStore.ts          # Module-level mutable store (gesture → physics bridge)
├── types/
│   └── index.ts              # TypeScript types, constants, configuration
├── constants/
│   └── shapes.ts             # Shape/color definitions, voice parsers
├── context/
│   └── SandboxContext.tsx     # React Context + useReducer state management
├── hooks/
│   ├── useHandTracking.ts    # MediaPipe HandLandmarker integration
│   ├── useGestureRecognition.ts  # Maps hand landmarks to 3D interactions
│   └── useVoiceRecognition.ts    # Web Speech API integration
├── components/
│   ├── Scene/
│   │   ├── Scene.tsx         # R3F Canvas setup, physics world
│   │   ├── Ground.tsx        # Physics ground plane
│   │   ├── Lighting.tsx      # Directional, ambient, hemisphere lights
│   │   ├── SceneObjects.tsx  # Iterates state.objects → PhysicsObject
│   │   ├── PhysicsObject.tsx # Rigid body, geometry, grab/release logic
│   │   └── Pointer3D.tsx     # 3D pointer visualization
│   ├── Camera/
│   │   └── CameraController.tsx  # Gesture-driven + OrbitControls camera
│   └── UI/
│       ├── Overlay.tsx       # Top-level overlay container
│       ├── SpawnBar.tsx      # Spawn tiles with pinch detection
│       ├── StatusPanel.tsx   # Tracking/gesture status display
│       ├── Instructions.tsx  # Help panel
│       ├── HandVisualization.tsx  # Webcam feed + skeleton
│       └── VoiceIndicator.tsx    # Mic status + last command
├── App.tsx                   # Root orchestration, permissions, tracking loop
├── main.tsx                  # React entry point
└── styles.css                # Global styles
```

### Data Flow

```
Camera Webcam
     │
     ▼
MediaPipe HandLandmarker  (useHandTracking)
     │
     ▼
Hand Landmarks [ NormalizedLandmark[] ]
     │
     ├──▶ Gesture Engine (gestures.ts)
     │         │
     │         ├── isPinching()  — hysteresis + EMA + grace frames
     │         ├── detectGesture()  — point / pinch / none
     │         ├── getHandScale()  — proximity estimation
     │         ├── getHandRollAngle()  — wrist twist angle
     │         ├── getFingerSpread()  — second-hand spread
     │         └── screenToWorld / screenToWorldAtDistance
     │
     ▼
useGestureRecognition  (runs at ~30 fps via setInterval)
     │
     ├──▶ grabStore  (module-level mutable state)
     │         │
     │         ├── grabState.objectId
     │         ├── grabState.targetPosition
     │         ├── grabState.twistAngle / twistAxis
     │         ├── grabState.scaleFactor
     │         ├── grabState.releaseVelocity
     │         └── cameraZoomState.radius
     │
     ├──▶ React State (SandboxContext dispatch)
     │         │
     │         ├── gesture, pointer, camera state
     │         └── object CRUD (add, remove, update, select)
     │
     ▼
useFrame (60 fps render loop)
     │
     ├── PhysicsObject reads grabStore → setNextKinematicTranslation/Rotation
     ├── CameraController reads camera state → camera.position
     ├── Pointer3D reads pointer state → mesh position
     └── SpawnBar reads gesture state → spawn detection
```

### Key Design Decision: grabStore

The gesture recognition hook runs in a `setInterval` (~30 fps), while physics objects update in `useFrame` (~60 fps). Passing grab positions through React state would introduce a one-frame lag and unnecessary re-renders. The `grabStore` module provides a shared mutable store that the gesture hook writes to and `PhysicsObject` reads from directly, enabling smooth, lag-free object movement.

---

## Realization

### Gesture Detection Pipeline

1. **Pinch detection** uses hysteresis thresholds (`PINCH_ON = 0.06`, `PINCH_OFF = 0.09`) to prevent rapid toggling, an exponential moving average (EMA) on the raw thumb-index distance for noise reduction, and a grace period of 4 frames before releasing.

2. **Point detection** checks that the index finger is extended (tip farther from wrist than PIP joint) while middle, ring, and pinky fingers are curled.

3. **Screen position smoothing** applies EMA (`POS_SMOOTH = 0.3`) to all screen coordinates. The webcam mirror is accounted for by flipping the x-axis (`1 - landmark.x`).

4. **Hand scale** (proximity estimation) averages two palm distances and applies EMA smoothing to serve as a proxy for hand-to-camera distance. This drives depth movement of grabbed objects and camera zoom.

5. **Hand roll angle** measures the 2D screen-space angle of the WRIST→MIDDLE_MCP vector using `atan2`. EMA with angle wrapping prevents discontinuities at ±π. The sensitivity multiplier (`TWIST_SENSITIVITY = 5.0`) converts subtle wrist twists into visible rotations.

6. **Finger spread** (second hand) measures the thumb-index distance on the non-primary hand with independent EMA smoothing, driving object scaling.

### Object Interaction States

When a pinch starts:
- **Near an object**: enter `grab` mode. The object's rigid body switches to `kinematic`, capturing the grab offset and initial rotation quaternion.
- **No object nearby**: enter `camera` mode. Lateral hand movement orbits, hand proximity zooms.

During a grab:
- Screen position is projected along the camera ray at the current grab distance (`screenToWorldAtDistance`), allowing free 3D movement.
- Hand scale ratio modulates the grab distance, pushing/pulling the object along the depth axis.
- Hand roll delta is multiplied by sensitivity and applied as a quaternion rotation around the camera's forward vector, composed with the grab-start rotation.
- If a second hand appears, its finger spread ratio drives live visual scaling of the mesh.

On release:
- The `VelocityTracker` computes an average velocity from the last 4 position samples. If speed is below `MIN_THROW_SPEED` (1.5), velocity is zeroed (preventing accidental throws). If above `MAX_THROW_SPEED` (30), it's capped.
- The rigid body is switched back to `dynamic` and the computed velocity is applied as a linear impulse.
- If scaling was applied, the final scale is committed to React state.

### Camera Control

Camera position is computed in spherical coordinates (theta, phi, radius) and converted to Cartesian. During gesture-driven orbit, `OrbitControls` is disabled. When not gesture-controlled, `OrbitControls` provides mouse-based interaction. Dead zones (`ORBIT_DEADZONE = 0.003`, `SCALE_DEADZONE = 0.03`) suppress jitter from landmark noise.

### Spawn Bar

The spawn bar detects pinch-over-tile by converting the normalized screen position from the gesture system to pixel coordinates and comparing against each tile's `getBoundingClientRect()` with a 20px tolerance padding. Objects spawn at the pointer's current world position.

### Voice Recognition

The Web Speech API runs in continuous mode, auto-restarting on end. Transcripts are parsed against known shape types, color names, and command verbs. Synonyms are supported (`box` → `cube`, `grey` → `gray`).

---

## Data Model

### Core Types

```typescript
type ShapeType = 'cube' | 'sphere' | 'cylinder' | 'pyramid';
type ColorName = 'red' | 'blue' | 'green' | 'yellow' | 'orange' | 'purple' | 'white' | 'gray';
type Vector3Tuple = [number, number, number];

interface SceneObject {
  id: string;            // Unique ID (timestamp + random)
  type: ShapeType;
  position: Vector3Tuple;
  rotation: Vector3Tuple;
  scale: Vector3Tuple;
  color: string;         // Hex color
  isSelected: boolean;
}
```

### Application State (`SandboxState`)

| Field | Type | Description |
|---|---|---|
| `objects` | `SceneObject[]` | All objects in the scene |
| `selectedObjectId` | `string \| null` | Currently selected object |
| `pointer` | `PointerState` | 3D pointer visibility, position, mode |
| `gesture` | `GestureState` | Current gesture type, pinch state, screen/world position |
| `voice` | `VoiceState` | Listening status, last command, errors |
| `camera` | `CameraState` | Camera position, target, zoom |
| `handTracking` | `HandTrackingState` | Tracking status, hand data |
| `isLoading` | `boolean` | MediaPipe initialization in progress |
| `hasPermissions` | `boolean` | Camera/mic permissions granted |

### Grab Store (Module-level, Mutable)

```typescript
interface GrabState {
  objectId: string | null;
  targetPosition: Vector3Tuple;
  twistAngle: number;
  twistAxis: Vector3Tuple;
  scaleFactor: number;
  releaseVelocity: Vector3Tuple;
  pendingRelease: boolean;
}
```

### Reducer Actions

| Action | Payload | Effect |
|---|---|---|
| `ADD_OBJECT` | `Omit<SceneObject, 'id' \| 'isSelected'>` | Creates object with generated ID |
| `REMOVE_OBJECT` | `string` (id) | Removes object, clears selection if matched |
| `UPDATE_OBJECT` | `{ id, updates: Partial<SceneObject> }` | Partial update (position, rotation, scale) |
| `SELECT_OBJECT` | `string \| null` | Sets selection, updates `isSelected` flags |
| `CLEAR_ALL_OBJECTS` | — | Removes all objects |
| `SET_POINTER` | `Partial<PointerState>` | Updates pointer state |
| `SET_GESTURE` | `Partial<GestureState>` | Updates gesture state |
| `SET_VOICE` | `Partial<VoiceState>` | Updates voice state |
| `SET_CAMERA` | `Partial<CameraState>` | Updates camera state |
| `SET_HAND_TRACKING` | `Partial<HandTrackingState>` | Updates tracking state |
| `SET_LOADING` | `boolean` | Loading flag |
| `SET_PERMISSIONS` | `boolean` | Permissions flag |

---

## Used Datasets

### MediaPipe Hand Landmarker Model

- **Model**: `hand_landmarker.task` (float16)
- **Source**: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- **WASM Runtime**: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm`
- **Output**: 21 3D landmarks per hand (normalized coordinates), handedness classification
- **Landmark schema**: WRIST (0), THUMB_CMC (1) through THUMB_TIP (4), INDEX_MCP (5) through INDEX_TIP (8), MIDDLE_MCP (9) through MIDDLE_TIP (12), RING_MCP (13) through RING_TIP (16), PINKY_MCP (17) through PINKY_TIP (20)

### HDRI Environment Map

- **Preset**: `"city"` (from `@react-three/drei`'s `<Environment>` component)
- **Source**: Loaded from the Poly Haven CDN at runtime
- **Purpose**: Provides image-based lighting and reflections for realistic material appearance

---

## Configuration

### Scene Configuration (`SCENE_CONFIG`)

| Parameter | Value | Description |
|---|---|---|
| `ground.size` | `[30, 30]` | Ground plane dimensions |
| `ground.color` | `#3a3a4a` | Ground material color |
| `lighting.ambient.intensity` | `0.4` | Ambient light intensity |
| `lighting.directional.position` | `[10, 15, 10]` | Main light position |
| `lighting.directional.intensity` | `1.2` | Main light intensity |
| `lighting.directional.shadowMapSize` | `2048` | Shadow map resolution |
| `camera.position` | `[0, 8, 12]` | Initial camera position |
| `camera.fov` | `60` | Field of view (degrees) |

### Gesture Tuning Constants

| Constant | Value | Location | Purpose |
|---|---|---|---|
| `PINCH_ON` | `0.06` | `gestures.ts` | Thumb-index distance to start pinch |
| `PINCH_OFF` | `0.09` | `gestures.ts` | Thumb-index distance to end pinch |
| `SMOOTH_FACTOR` | `0.35` | `gestures.ts` | EMA factor for pinch distance |
| `RELEASE_GRACE_FRAMES` | `4` | `gestures.ts` | Frames before pinch release |
| `POS_SMOOTH` | `0.3` | `gestures.ts` | EMA factor for screen position |
| `HAND_SCALE_SMOOTH` | `0.2` | `gestures.ts` | EMA factor for hand scale |
| `ROLL_SMOOTH` | `0.6` | `gestures.ts` | EMA factor for roll angle |
| `SPREAD_SMOOTH` | `0.4` | `gestures.ts` | EMA factor for finger spread |
| `ORBIT_SENSITIVITY` | `3` | `useGestureRecognition.ts` | Camera orbit speed |
| `ORBIT_DEADZONE` | `0.003` | `useGestureRecognition.ts` | Minimum screen delta for orbit |
| `SCALE_DEADZONE` | `0.03` | `useGestureRecognition.ts` | Minimum scale ratio change |
| `TWIST_SENSITIVITY` | `5.0` | `useGestureRecognition.ts` | Roll-to-rotation multiplier |
| `DEPTH_SENSITIVITY` | `1.2` | `useGestureRecognition.ts` | Hand-scale-to-depth multiplier |
| `ZOOM_SENSITIVITY` | `1.4` | `useGestureRecognition.ts` | Hand-scale-to-zoom multiplier |
| `MIN_GRAB_DIST` / `MAX_GRAB_DIST` | `2` / `25` | `useGestureRecognition.ts` | Object depth range |
| `MIN_ZOOM` / `MAX_ZOOM` | `5` / `30` | `useGestureRecognition.ts` | Camera orbit radius range |
| `MIN_SCALE` / `MAX_SCALE` | `0.3` / `4.0` | `useGestureRecognition.ts` | Object scale range |
| `GRAB_SCREEN_RADIUS` | `0.12` | `useGestureRecognition.ts` | Proximity pick radius (normalized) |

### Physics Constants

| Constant | Value | Location | Purpose |
|---|---|---|---|
| `VELOCITY_SCALE` | `2.5` | `physics.ts` | Throw velocity multiplier |
| `MIN_THROW_SPEED` | `1.5` | `physics.ts` | Below this, velocity is zeroed |
| `MAX_THROW_SPEED` | `30` | `physics.ts` | Speed cap for throws |
| `GROUND_HALF` | `14` | `physics.ts` | Scene boundary half-extent |

### Vite Configuration

| Setting | Value |
|---|---|
| Dev server port | `5173` |
| Auto-open browser | `true` |
| Path alias | `@` → `./src` |
| Source maps | Enabled in production build |

---

## Technologies and Libraries

### Runtime Dependencies

| Library | Version | Purpose |
|---|---|---|
| **React** | `^19.0.0` | UI component framework |
| **React DOM** | `^19.0.0` | React renderer for the browser |
| **Three.js** | `^0.182.0` | 3D graphics engine (WebGL) |
| **@react-three/fiber** | `^9.5.0` | React renderer for Three.js — declarative 3D scene graph |
| **@react-three/drei** | `^10.7.7` | Utility components (OrbitControls, Environment) |
| **@react-three/rapier** | `^2.2.0` | WASM-based physics engine (Rapier) with React bindings — rigid bodies, colliders, gravity |
| **@mediapipe/tasks-vision** | `^0.10.32` | Hand landmark detection via GPU-accelerated ML model |

### Development Dependencies

| Library | Version | Purpose |
|---|---|---|
| **TypeScript** | `^5.7.0` | Static type checking |
| **Vite** | `^6.0.0` | Build tool and dev server (ESM-native, HMR) |
| **@vitejs/plugin-react** | `^4.0.0` | React Fast Refresh for Vite |
| **@types/three** | `^0.182.0` | TypeScript definitions for Three.js |
| **@types/react** | `^19.0.0` | TypeScript definitions for React |
| **@types/react-dom** | `^19.0.0` | TypeScript definitions for React DOM |

### Browser APIs

| API | Purpose |
|---|---|
| **MediaDevices (getUserMedia)** | Camera access for hand tracking |
| **Web Speech API (SpeechRecognition)** | Continuous voice command recognition |
| **WebGL 2** | 3D rendering (via Three.js) |
| **WebAssembly** | Physics simulation (Rapier WASM) and ML inference (MediaPipe WASM) |

---

## Problems and Solutions

### 1. App Loading Flicker

**Problem**: On startup, the loading screen and loaded state flickered recursively. Conditional rendering of the loading screen unmounted the `<video>` element, setting `videoRef.current` to `null`, which re-triggered a loading state in a loop.

**Solution**: Changed the loading indicator from a conditional mount/unmount to a transparent overlay that sits on top of the scene. The video element stays mounted at all times. Used refs (`isLoadingRef`, `isTrackingRef`) instead of state values in `useCallback` dependencies to keep callback identities stable.

### 2. Pinch Gesture Instability

**Problem**: The raw pinch detection toggled rapidly between pinching and not-pinching states due to per-frame noise in MediaPipe landmark positions.

**Solution**: Implemented a three-layer stabilization system:
- **EMA smoothing** on the thumb-index distance to dampen noise
- **Hysteresis thresholds** with separate on/off values (`PINCH_ON < PINCH_OFF`) to prevent toggling at the boundary
- **Grace period** of 4 consecutive "not pinching" frames required before releasing

### 3. Cannot Pinch Objects

**Problem**: Two issues — (a) calling `isPinching()` twice per frame corrupted its module-level state, and (b) pixel-precise raycasting was unreliable for selecting objects with hand gestures.

**Solution**: (a) Refactored to compute the pinch boolean once and pass it to `detectGesture()`. (b) Replaced raycasting with proximity-based object selection: all objects are projected to screen space and the nearest one within a generous radius (`GRAB_SCREEN_RADIUS = 0.12`) is picked. Introduced `grabStore` as a module-level mutable bridge between the gesture hook's `setInterval` and the physics engine's `useFrame`.

### 4. Camera/Object Jitter When Stationary

**Problem**: Even when the hand was still, MediaPipe landmarks fluctuated slightly each frame, causing the camera to orbit or grabbed objects to vibrate.

**Solution**: Applied EMA smoothing to all screen positions (`POS_SMOOTH = 0.3`), hand scale (`HAND_SCALE_SMOOTH = 0.2`), and introduced dead zones (`ORBIT_DEADZONE`, `SCALE_DEADZONE`) that ignore changes below a minimum threshold.

### 5. Objects Locked to Ground Plane

**Problem**: Object movement was projected onto a fixed horizontal plane, preventing vertical lifting.

**Solution**: Implemented `screenToWorldAtDistance()` which projects screen coordinates to a 3D point at a specified distance along the camera ray. On grab start, the camera-to-object distance is stored. Hand movement in any direction (including vertical) now moves the object freely in the camera's view plane.

### 6. Hand Twist Rotation Not Working

**Problem**: Multiple iterations were needed. Initially, the roll angle measurement used unreliable landmark combinations (INDEX_MCP→PINKY_MCP), which collapsed during pinching. The rotation was also applied around the world Y-axis instead of the camera's view axis.

**Solution**: Simplified `getHandRollAngle()` to measure only the 2D screen-space angle of the WRIST→MIDDLE_MCP vector (the longest stable hand axis during pinch). EMA smoothing with angle wrapping handles the ±π discontinuity. The rotation axis is captured as the camera's forward direction at grab start and stored in `grabState.twistAxis`, so rotation is always relative to the user's point of view.

### 7. Stationary Drop Sends Objects Flying

**Problem**: The `VelocityTracker` amplified minor EMA smoothing drift into a throw velocity, even when the hand was essentially still.

**Solution**: Added a minimum speed threshold (`MIN_THROW_SPEED = 1.5`) — velocities below this are zeroed, so stationary releases simply drop under gravity. Added a maximum cap (`MAX_THROW_SPEED = 30`) and reduced the sample window to the last 4 positions for responsiveness.

### 8. Peer Dependency Conflicts

**Problem**: `@react-three/drei@9.x` had peer dependency conflicts with `@react-three/fiber@9.x` during `npm install`.

**Solution**: Upgraded `@react-three/drei` to `^10.x` and aligned `three` and `@types/three` versions.

---

## References

### Core Libraries

- Three.js — https://threejs.org/docs/
- React Three Fiber — https://r3f.docs.pmnd.rs/
- React Three Drei — https://drei.docs.pmnd.rs/
- React Three Rapier — https://github.com/pmndrs/react-three-rapier
- Rapier Physics Engine — https://rapier.rs/docs/

### Hand Tracking

- MediaPipe Hand Landmarker — https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- MediaPipe Tasks Vision (npm) — https://www.npmjs.com/package/@mediapipe/tasks-vision
- Hand Landmark Model Card — https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20Lite%20and%20Full.pdf

### Voice Recognition

- Web Speech API (MDN) — https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition

### Build Tooling

- Vite — https://vite.dev/guide/
- TypeScript — https://www.typescriptlang.org/docs/

### 3D Math

- Three.js Quaternion — https://threejs.org/docs/#api/en/math/Quaternion
- Spherical Coordinates — https://en.wikipedia.org/wiki/Spherical_coordinate_system
- Exponential Moving Average — https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
