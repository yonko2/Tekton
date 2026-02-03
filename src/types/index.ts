import type { NormalizedLandmark } from '@mediapipe/hands';

// Shape types
export type ShapeType = 'cube' | 'sphere' | 'cylinder' | 'cone' | 'torus' | 'pyramid';

// Predefined colors
export type ColorName = 'red' | 'blue' | 'green' | 'yellow' | 'orange' | 'purple' | 'white' | 'black';

// 3D Vector types
export type Vector3Tuple = [number, number, number];

// Scene Object
export interface SceneObject {
  id: string;
  type: ShapeType;
  position: Vector3Tuple;
  rotation: Vector3Tuple;
  scale: Vector3Tuple;
  color: string;
  isSelected: boolean;
}

// Hand Tracking
export interface HandLandmarks {
  landmarks: NormalizedLandmark[];
  handedness: 'Left' | 'Right';
}

export interface HandTrackingState {
  isTracking: boolean;
  hands: HandLandmarks[];
  primaryHand: HandLandmarks | null;
}

// Gesture Types
export type GestureType = 
  | 'none'
  | 'point'
  | 'pinch'
  | 'open_palm'
  | 'fist'
  | 'two_finger_pinch'
  | 'swipe_left'
  | 'swipe_right';

export interface GestureState {
  currentGesture: GestureType;
  pinchDistance: number;
  isPinching: boolean;
  pointerPosition: Vector3Tuple | null;
  screenPosition: { x: number; y: number } | null;
  confidence: number;
}

// Pointer State
export type PointerMode = 'idle' | 'hovering' | 'grabbing' | 'scaling';

export interface PointerState {
  visible: boolean;
  position: Vector3Tuple;
  mode: PointerMode;
  hoveredObjectId: string | null;
  grabbedObjectId: string | null;
}

// Voice Recognition
export interface VoiceCommand {
  type: 'create' | 'delete' | 'clear';
  shape?: ShapeType;
  color?: ColorName;
}

export interface VoiceState {
  isListening: boolean;
  isSupported: boolean;
  lastCommand: string | null;
  lastParsedCommand: VoiceCommand | null;
  error: string | null;
}

// Camera State
export interface CameraState {
  position: Vector3Tuple;
  target: Vector3Tuple;
  zoom: number;
}

// Application State
export interface SandboxState {
  objects: SceneObject[];
  selectedObjectId: string | null;
  pointer: PointerState;
  gesture: GestureState;
  voice: VoiceState;
  camera: CameraState;
  handTracking: HandTrackingState;
  isLoading: boolean;
  hasPermissions: boolean;
}

// Action Types
export type SandboxAction =
  | { type: 'ADD_OBJECT'; payload: Omit<SceneObject, 'id' | 'isSelected'> }
  | { type: 'REMOVE_OBJECT'; payload: string }
  | { type: 'UPDATE_OBJECT'; payload: { id: string; updates: Partial<SceneObject> } }
  | { type: 'SELECT_OBJECT'; payload: string | null }
  | { type: 'CLEAR_ALL_OBJECTS' }
  | { type: 'SET_POINTER'; payload: Partial<PointerState> }
  | { type: 'SET_GESTURE'; payload: Partial<GestureState> }
  | { type: 'SET_VOICE'; payload: Partial<VoiceState> }
  | { type: 'SET_CAMERA'; payload: Partial<CameraState> }
  | { type: 'SET_HAND_TRACKING'; payload: Partial<HandTrackingState> }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_PERMISSIONS'; payload: boolean };

// Landmark indices for MediaPipe Hands
export const HAND_LANDMARKS = {
  WRIST: 0,
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
} as const;

// Scene configuration
export const SCENE_CONFIG = {
  ground: {
    size: [20, 20] as [number, number],
    color: '#505050',
    receiveShadow: true,
  },
  lighting: {
    ambient: { intensity: 0.4 },
    directional: {
      position: [10, 15, 10] as Vector3Tuple,
      intensity: 1,
      castShadow: true,
      shadowMapSize: 2048,
    },
  },
  camera: {
    position: [0, 8, 12] as Vector3Tuple,
    fov: 60,
  },
} as const;
