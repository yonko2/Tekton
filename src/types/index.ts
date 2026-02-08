export type ShapeType = 'cube' | 'sphere' | 'cylinder' | 'pyramid';

export type ColorName = 'red' | 'blue' | 'green' | 'yellow' | 'orange' | 'purple' | 'white' | 'gray';

export type Vector3Tuple = [number, number, number];

export interface SceneObject {
  id: string;
  type: ShapeType;
  position: Vector3Tuple;
  rotation: Vector3Tuple;
  scale: Vector3Tuple;
  color: string;
  isSelected: boolean;
}

export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface HandData {
  landmarks: NormalizedLandmark[];
  worldLandmarks: NormalizedLandmark[];
  handedness: 'Left' | 'Right';
}

export interface HandTrackingState {
  isTracking: boolean;
  hands: HandData[];
  primaryHand: HandData | null;
}

export type GestureType = 'none' | 'point' | 'pinch';

export interface GestureState {
  currentGesture: GestureType;
  pinchDistance: number;
  isPinching: boolean;
  pointerPosition: Vector3Tuple | null;
  screenPosition: { x: number; y: number } | null;
  confidence: number;
}

export type PointerMode = 'idle' | 'hovering' | 'grabbing' | 'camera';

export interface PointerState {
  visible: boolean;
  position: Vector3Tuple;
  mode: PointerMode;
  hoveredObjectId: string | null;
  grabbedObjectId: string | null;
}

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

export interface CameraState {
  position: Vector3Tuple;
  target: Vector3Tuple;
  zoom: number;
}

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

export const SCENE_CONFIG = {
  ground: {
    size: [30, 30] as [number, number],
    color: '#3a3a4a',
    receiveShadow: true,
  },
  lighting: {
    ambient: { intensity: 0.4 },
    directional: {
      position: [10, 15, 10] as Vector3Tuple,
      intensity: 1.2,
      castShadow: true,
      shadowMapSize: 2048,
    },
  },
  camera: {
    position: [0, 8, 12] as Vector3Tuple,
    fov: 60,
  },
} as const;
