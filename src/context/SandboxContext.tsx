import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  type ReactNode,
  type Dispatch,
} from 'react';
import type {
  SandboxState,
  SandboxAction,
  SceneObject,
  PointerState,
  GestureState,
  VoiceState,
  CameraState,
  HandTrackingState,
  Vector3Tuple,
  ShapeType,
} from '@/types';
import { SCENE_CONFIG } from '@/types';
import { getRandomColor } from '@/constants/shapes';


const uid = (): string => `obj_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;


const starterObjects: SceneObject[] = [
  {
    id: 'starter_cube',
    type: 'cube',
    position: [-2, 0.5, 0],
    rotation: [0, 0, 0],
    scale: [1, 1, 1],
    color: '#e53935',
    isSelected: false,
  },
  {
    id: 'starter_sphere',
    type: 'sphere',
    position: [0, 0.5, 0],
    rotation: [0, 0, 0],
    scale: [1, 1, 1],
    color: '#1e88e5',
    isSelected: false,
  },
  {
    id: 'starter_cylinder',
    type: 'cylinder',
    position: [2, 0.75, 0],
    rotation: [0, 0, 0],
    scale: [1, 1, 1],
    color: '#43a047',
    isSelected: false,
  },
];


const initialState: SandboxState = {
  objects: starterObjects,
  selectedObjectId: null,
  pointer: {
    visible: false,
    position: [0, 0, 0],
    mode: 'idle',
    hoveredObjectId: null,
    grabbedObjectId: null,
  },
  gesture: {
    currentGesture: 'none',
    pinchDistance: 1,
    isPinching: false,
    pointerPosition: null,
    screenPosition: null,
    confidence: 0,
  },
  voice: {
    isListening: false,
    isSupported:
      typeof window !== 'undefined' &&
      ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window),
    lastCommand: null,
    lastParsedCommand: null,
    error: null,
  },
  camera: {
    position: [...SCENE_CONFIG.camera.position],
    target: [0, 0, 0],
    zoom: 1,
  },
  handTracking: {
    isTracking: false,
    hands: [],
    primaryHand: null,
  },
  isLoading: true,
  hasPermissions: false,
};


function reducer(state: SandboxState, action: SandboxAction): SandboxState {
  switch (action.type) {
    case 'ADD_OBJECT': {
      const obj: SceneObject = { ...action.payload, id: uid(), isSelected: false };
      return { ...state, objects: [...state.objects, obj] };
    }
    case 'REMOVE_OBJECT':
      return {
        ...state,
        objects: state.objects.filter((o) => o.id !== action.payload),
        selectedObjectId: state.selectedObjectId === action.payload ? null : state.selectedObjectId,
      };
    case 'UPDATE_OBJECT':
      return {
        ...state,
        objects: state.objects.map((o) =>
          o.id === action.payload.id ? { ...o, ...action.payload.updates } : o,
        ),
      };
    case 'SELECT_OBJECT':
      return {
        ...state,
        selectedObjectId: action.payload,
        objects: state.objects.map((o) => ({ ...o, isSelected: o.id === action.payload })),
      };
    case 'CLEAR_ALL_OBJECTS':
      return { ...state, objects: [], selectedObjectId: null };
    case 'SET_POINTER':
      return { ...state, pointer: { ...state.pointer, ...action.payload } };
    case 'SET_GESTURE':
      return { ...state, gesture: { ...state.gesture, ...action.payload } };
    case 'SET_VOICE':
      return { ...state, voice: { ...state.voice, ...action.payload } };
    case 'SET_CAMERA':
      return { ...state, camera: { ...state.camera, ...action.payload } };
    case 'SET_HAND_TRACKING':
      return { ...state, handTracking: { ...state.handTracking, ...action.payload } };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_PERMISSIONS':
      return { ...state, hasPermissions: action.payload };
    default:
      return state;
  }
}


interface SandboxContextValue {
  state: SandboxState;
  dispatch: Dispatch<SandboxAction>;
  addObject: (type: ShapeType, position: Vector3Tuple, color?: string) => void;
  removeObject: (id: string) => void;
  updateObjectPosition: (id: string, position: Vector3Tuple) => void;
  updateObjectScale: (id: string, scale: Vector3Tuple) => void;
  updateObjectRotation: (id: string, rotation: Vector3Tuple) => void;
  selectObject: (id: string | null) => void;
  clearAllObjects: () => void;
  setPointer: (p: Partial<PointerState>) => void;
  setGesture: (g: Partial<GestureState>) => void;
  setVoice: (v: Partial<VoiceState>) => void;
  setCamera: (c: Partial<CameraState>) => void;
  setHandTracking: (h: Partial<HandTrackingState>) => void;
  setLoading: (l: boolean) => void;
  setPermissions: (p: boolean) => void;
}

const SandboxContext = createContext<SandboxContextValue | null>(null);


export function SandboxProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const addObject = useCallback(
    (type: ShapeType, position: Vector3Tuple, color?: string) =>
      dispatch({
        type: 'ADD_OBJECT',
        payload: { type, position, rotation: [0, 0, 0], scale: [1, 1, 1], color: color ?? getRandomColor() },
      }),
    [],
  );

  const removeObject = useCallback((id: string) => dispatch({ type: 'REMOVE_OBJECT', payload: id }), []);
  const updateObjectPosition = useCallback(
    (id: string, position: Vector3Tuple) =>
      dispatch({ type: 'UPDATE_OBJECT', payload: { id, updates: { position } } }),
    [],
  );
  const updateObjectScale = useCallback(
    (id: string, scale: Vector3Tuple) =>
      dispatch({ type: 'UPDATE_OBJECT', payload: { id, updates: { scale } } }),
    [],
  );
  const updateObjectRotation = useCallback(
    (id: string, rotation: Vector3Tuple) =>
      dispatch({ type: 'UPDATE_OBJECT', payload: { id, updates: { rotation } } }),
    [],
  );
  const selectObject = useCallback((id: string | null) => dispatch({ type: 'SELECT_OBJECT', payload: id }), []);
  const clearAllObjects = useCallback(() => dispatch({ type: 'CLEAR_ALL_OBJECTS' }), []);
  const setPointer = useCallback((p: Partial<PointerState>) => dispatch({ type: 'SET_POINTER', payload: p }), []);
  const setGesture = useCallback((g: Partial<GestureState>) => dispatch({ type: 'SET_GESTURE', payload: g }), []);
  const setVoice = useCallback((v: Partial<VoiceState>) => dispatch({ type: 'SET_VOICE', payload: v }), []);
  const setCamera = useCallback((c: Partial<CameraState>) => dispatch({ type: 'SET_CAMERA', payload: c }), []);
  const setHandTracking = useCallback(
    (h: Partial<HandTrackingState>) => dispatch({ type: 'SET_HAND_TRACKING', payload: h }),
    [],
  );
  const setLoading = useCallback((l: boolean) => dispatch({ type: 'SET_LOADING', payload: l }), []);
  const setPermissions = useCallback((p: boolean) => dispatch({ type: 'SET_PERMISSIONS', payload: p }), []);

  return (
    <SandboxContext.Provider
      value={{
        state,
        dispatch,
        addObject,
        removeObject,
        updateObjectPosition,
        updateObjectScale,
        updateObjectRotation,
        selectObject,
        clearAllObjects,
        setPointer,
        setGesture,
        setVoice,
        setCamera,
        setHandTracking,
        setLoading,
        setPermissions,
      }}
    >
      {children}
    </SandboxContext.Provider>
  );
}


export function useSandbox(): SandboxContextValue {
  const ctx = useContext(SandboxContext);
  if (!ctx) throw new Error('useSandbox must be used within <SandboxProvider>');
  return ctx;
}
