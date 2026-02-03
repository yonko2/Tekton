import type { ShapeType, Vector3Tuple, SceneObject } from '@/types';
import { getRandomColor } from '@/constants/shapes';

// Generate unique ID
export const generateObjectId = (): string => {
  return `obj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

// Create a new scene object
export function createSceneObject(
  type: ShapeType,
  position: Vector3Tuple,
  options?: {
    color?: string;
    scale?: Vector3Tuple;
    rotation?: Vector3Tuple;
  }
): SceneObject {
  const defaultScale = getDefaultScale(type);
  
  return {
    id: generateObjectId(),
    type,
    position,
    rotation: options?.rotation || [0, 0, 0],
    scale: options?.scale || defaultScale,
    color: options?.color || getRandomColor(),
    isSelected: false,
  };
}

// Get default scale for each shape type
export function getDefaultScale(type: ShapeType): Vector3Tuple {
  switch (type) {
    case 'cube':
      return [1, 1, 1];
    case 'sphere':
      return [0.6, 0.6, 0.6];
    case 'cylinder':
      return [0.5, 1, 0.5];
    case 'cone':
      return [0.5, 1, 0.5];
    case 'torus':
      return [0.6, 0.6, 0.6];
    case 'pyramid':
      return [1, 1, 1];
    default:
      return [1, 1, 1];
  }
}

// Get the height of an object for stacking purposes
export function getObjectHeight(type: ShapeType, scale: Vector3Tuple): number {
  const baseHeight = getBaseHeight(type);
  return baseHeight * scale[1];
}

// Get base height before scaling
function getBaseHeight(type: ShapeType): number {
  switch (type) {
    case 'cube':
      return 1;
    case 'sphere':
      return 1; // diameter
    case 'cylinder':
      return 2; // default cylinder height
    case 'cone':
      return 1;
    case 'torus':
      return 0.4; // tube diameter
    case 'pyramid':
      return 1;
    default:
      return 1;
  }
}

// Get the Y offset for positioning (from center to bottom)
export function getObjectYOffset(type: ShapeType, scale: Vector3Tuple): number {
  const height = getObjectHeight(type, scale);
  
  switch (type) {
    case 'sphere':
      return height / 2;
    case 'torus':
      return 0.2 * scale[1]; // Torus sits flat
    default:
      return height / 2;
  }
}

// Calculate position on top of another object
export function getStackPosition(
  baseObject: SceneObject,
  newType: ShapeType,
  newScale: Vector3Tuple
): Vector3Tuple {
  const baseTop = baseObject.position[1] + getObjectYOffset(baseObject.type, baseObject.scale);
  const newBottom = getObjectYOffset(newType, newScale);
  
  return [
    baseObject.position[0],
    baseTop + newBottom,
    baseObject.position[2],
  ];
}
