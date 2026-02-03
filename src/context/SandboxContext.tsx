import React, { createContext, useContext, useReducer, useCallback, type ReactNode } from 'react';
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

// Generate unique ID
const generateId = (): string => {
  return `obj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

// Initial state
const initialState: SandboxState = {
  objects: [],
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
    isSupported: typeof window !== 'undefined' && 'webkitSpeechRecognition' in window,
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

// Reducer function
function sandboxReducer(state: SandboxState, action: SandboxAction): SandboxState {
  switch (action.type) {
    case 'ADD_OBJECT': {
      const newObject: SceneObject = {
        ...action.payload,
        id: generateId(),
        isSelected: false,
      };
      return {
        ...state,
        objects: [...state.objects, newObject],
      };
    }

    case 'REMOVE_OBJECT': {
      return {
        ...state,
        objects: state.objects.filter(obj => obj.id !== action.payload),
        selectedObjectId: state.selectedObjectId === action.payload ? null : state.selectedObjectId,
      };
    }

    case 'UPDATE_OBJECT': {
      return {
        ...state,
        objects: state.objects.map(obj =>
          obj.id === action.payload.id
            ? { ...obj, ...action.payload.updates }
            : obj
        ),
      };
    }

    case 'SELECT_OBJECT': {
      return {
        ...state,
        selectedObjectId: action.payload,
        objects: state.objects.map(obj => ({
          ...obj,
          isSelected: obj.id === action.payload,
        })),
      };
    }

    case 'CLEAR_ALL_OBJECTS': {
      return {
        ...state,
        objects: [],
        selectedObjectId: null,
      };
    }

    case 'SET_POINTER': {
      return {
        ...state,
        pointer: { ...state.pointer, ...action.payload },
      };
    }

    case 'SET_GESTURE': {
      return {
        ...state,
        gesture: { ...state.gesture, ...action.payload },
      };
    }

    case 'SET_VOICE': {
      return {
        ...state,
        voice: { ...state.voice, ...action.payload },
      };
    }

    case 'SET_CAMERA': {
      return {
        ...state,
        camera: { ...state.camera, ...action.payload },
      };
    }

    case 'SET_HAND_TRACKING': {
      return {
        ...state,
        handTracking: { ...state.handTracking, ...action.payload },
      };
    }

    case 'SET_LOADING': {
      return {
        ...state,
        isLoading: action.payload,
      };
    }

    case 'SET_PERMISSIONS': {
      return {
        ...state,
        hasPermissions: action.payload,
      };
    }

    default:
      return state;
  }
}

// Context type
interface SandboxContextType {
  state: SandboxState;
  dispatch: React.Dispatch<SandboxAction>;
  // Helper functions
  addObject: (type: ShapeType, position: Vector3Tuple, color?: string) => void;
  removeObject: (id: string) => void;
  updateObjectPosition: (id: string, position: Vector3Tuple) => void;
  updateObjectScale: (id: string, scale: Vector3Tuple) => void;
  selectObject: (id: string | null) => void;
  clearAllObjects: () => void;
  setPointer: (pointer: Partial<PointerState>) => void;
  setGesture: (gesture: Partial<GestureState>) => void;
  setVoice: (voice: Partial<VoiceState>) => void;
  setCamera: (camera: Partial<CameraState>) => void;
  setHandTracking: (tracking: Partial<HandTrackingState>) => void;
  setLoading: (loading: boolean) => void;
  setPermissions: (hasPermissions: boolean) => void;
}

// Create context
const SandboxContext = createContext<SandboxContextType | null>(null);

// Provider component
interface SandboxProviderProps {
  children: ReactNode;
}

export function SandboxProvider({ children }: SandboxProviderProps) {
  const [state, dispatch] = useReducer(sandboxReducer, initialState);

  // Helper functions
  const addObject = useCallback((type: ShapeType, position: Vector3Tuple, color?: string) => {
    dispatch({
      type: 'ADD_OBJECT',
      payload: {
        type,
        position,
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
        color: color || getRandomColor(),
      },
    });
  }, []);

  const removeObject = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_OBJECT', payload: id });
  }, []);

  const updateObjectPosition = useCallback((id: string, position: Vector3Tuple) => {
    dispatch({
      type: 'UPDATE_OBJECT',
      payload: { id, updates: { position } },
    });
  }, []);

  const updateObjectScale = useCallback((id: string, scale: Vector3Tuple) => {
    dispatch({
      type: 'UPDATE_OBJECT',
      payload: { id, updates: { scale } },
    });
  }, []);

  const selectObject = useCallback((id: string | null) => {
    dispatch({ type: 'SELECT_OBJECT', payload: id });
  }, []);

  const clearAllObjects = useCallback(() => {
    dispatch({ type: 'CLEAR_ALL_OBJECTS' });
  }, []);

  const setPointer = useCallback((pointer: Partial<PointerState>) => {
    dispatch({ type: 'SET_POINTER', payload: pointer });
  }, []);

  const setGesture = useCallback((gesture: Partial<GestureState>) => {
    dispatch({ type: 'SET_GESTURE', payload: gesture });
  }, []);

  const setVoice = useCallback((voice: Partial<VoiceState>) => {
    dispatch({ type: 'SET_VOICE', payload: voice });
  }, []);

  const setCamera = useCallback((camera: Partial<CameraState>) => {
    dispatch({ type: 'SET_CAMERA', payload: camera });
  }, []);

  const setHandTracking = useCallback((tracking: Partial<HandTrackingState>) => {
    dispatch({ type: 'SET_HAND_TRACKING', payload: tracking });
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  }, []);

  const setPermissions = useCallback((hasPermissions: boolean) => {
    dispatch({ type: 'SET_PERMISSIONS', payload: hasPermissions });
  }, []);

  const value: SandboxContextType = {
    state,
    dispatch,
    addObject,
    removeObject,
    updateObjectPosition,
    updateObjectScale,
    selectObject,
    clearAllObjects,
    setPointer,
    setGesture,
    setVoice,
    setCamera,
    setHandTracking,
    setLoading,
    setPermissions,
  };

  return (
    <SandboxContext.Provider value={value}>
      {children}
    </SandboxContext.Provider>
  );
}

// Hook to use context
export function useSandbox(): SandboxContextType {
  const context = useContext(SandboxContext);
  if (!context) {
    throw new Error('useSandbox must be used within a SandboxProvider');
  }
  return context;
}

export default SandboxContext;
