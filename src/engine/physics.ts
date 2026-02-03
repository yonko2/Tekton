import type { SceneObject, Vector3Tuple } from '@/types';
import { getObjectYOffset } from './objectFactory';

// Calculate bounding box for an object
export interface BoundingBox {
  min: Vector3Tuple;
  max: Vector3Tuple;
}

export function getObjectBoundingBox(object: SceneObject): BoundingBox {
  const [px, py, pz] = object.position;
  const [sx, sy, sz] = object.scale;
  
  // Approximate bounding box based on shape type
  let halfWidth: number, halfHeight: number, halfDepth: number;
  
  switch (object.type) {
    case 'sphere':
      halfWidth = halfHeight = halfDepth = 0.5 * Math.max(sx, sy, sz);
      break;
    case 'cylinder':
      halfWidth = halfDepth = 0.5 * sx;
      halfHeight = sy;
      break;
    case 'cone':
      halfWidth = halfDepth = 0.5 * sx;
      halfHeight = 0.5 * sy;
      break;
    case 'torus':
      halfWidth = halfDepth = 0.55 * sx;
      halfHeight = 0.2 * sy;
      break;
    case 'pyramid':
    case 'cube':
    default:
      halfWidth = 0.5 * sx;
      halfHeight = 0.5 * sy;
      halfDepth = 0.5 * sz;
  }
  
  return {
    min: [px - halfWidth, py - halfHeight, pz - halfDepth],
    max: [px + halfWidth, py + halfHeight, pz + halfDepth],
  };
}

// Check if two bounding boxes intersect
export function boxesIntersect(a: BoundingBox, b: BoundingBox): boolean {
  return (
    a.min[0] <= b.max[0] && a.max[0] >= b.min[0] &&
    a.min[1] <= b.max[1] && a.max[1] >= b.min[1] &&
    a.min[2] <= b.max[2] && a.max[2] >= b.min[2]
  );
}

// Check if point is inside bounding box
export function pointInBox(point: Vector3Tuple, box: BoundingBox): boolean {
  return (
    point[0] >= box.min[0] && point[0] <= box.max[0] &&
    point[1] >= box.min[1] && point[1] <= box.max[1] &&
    point[2] >= box.min[2] && point[2] <= box.max[2]
  );
}

// Find objects below a given position (for stacking)
export function findObjectsBelow(
  position: Vector3Tuple,
  objects: SceneObject[],
  excludeId?: string
): SceneObject[] {
  const [x, y, z] = position;
  
  return objects
    .filter(obj => {
      if (obj.id === excludeId) return false;
      
      const box = getObjectBoundingBox(obj);
      
      // Check if position is above this object's XZ footprint
      const isAboveXZ = x >= box.min[0] && x <= box.max[0] &&
                        z >= box.min[2] && z <= box.max[2];
      
      // Check if position is above this object's top
      const isAboveY = y > box.max[1];
      
      return isAboveXZ && isAboveY;
    })
    .sort((a, b) => {
      // Sort by top Y position (highest first)
      const aTop = getObjectBoundingBox(a).max[1];
      const bTop = getObjectBoundingBox(b).max[1];
      return bTop - aTop;
    });
}

// Calculate the resting position when dropping an object
export function calculateRestingPosition(
  droppedPosition: Vector3Tuple,
  droppedObject: SceneObject,
  otherObjects: SceneObject[]
): Vector3Tuple {
  const [x, _, z] = droppedPosition;
  
  // Find objects below the drop position
  const objectsBelow = findObjectsBelow(droppedPosition, otherObjects, droppedObject.id);
  
  // Get the Y offset for the dropped object (height from center to bottom)
  const yOffset = getObjectYOffset(droppedObject.type, droppedObject.scale);
  
  if (objectsBelow.length > 0) {
    // Stack on top of the highest object below
    const topObject = objectsBelow[0];
    const topBox = getObjectBoundingBox(topObject);
    const restY = topBox.max[1] + yOffset;
    
    return [x, restY, z];
  }
  
  // Rest on the ground
  return [x, yOffset, z];
}

// Check for collision between a moving object and other objects
export function checkCollision(
  movingObject: SceneObject,
  newPosition: Vector3Tuple,
  otherObjects: SceneObject[]
): { collides: boolean; adjustedPosition: Vector3Tuple } {
  let adjustedPos: Vector3Tuple = [...newPosition];
  let hasCollision = false;
  
  const movingYOffset = getObjectYOffset(movingObject.type, movingObject.scale);
  
  for (const other of otherObjects) {
    if (other.id === movingObject.id) continue;
    
    const testObject: SceneObject = {
      ...movingObject,
      position: adjustedPos,
    };
    
    const testBox = getObjectBoundingBox(testObject);
    const otherBox = getObjectBoundingBox(other);
    
    if (boxesIntersect(testBox, otherBox)) {
      hasCollision = true;
      
      // Calculate overlap in each axis
      const overlapX = Math.min(testBox.max[0] - otherBox.min[0], otherBox.max[0] - testBox.min[0]);
      const overlapY = Math.min(testBox.max[1] - otherBox.min[1], otherBox.max[1] - testBox.min[1]);
      const overlapZ = Math.min(testBox.max[2] - otherBox.min[2], otherBox.max[2] - testBox.min[2]);
      
      // Push out along the axis with smallest overlap
      if (overlapY <= overlapX && overlapY <= overlapZ) {
        // Push up (most common for stacking)
        adjustedPos[1] = otherBox.max[1] + movingYOffset;
      } else if (overlapX <= overlapZ) {
        // Push along X
        if (adjustedPos[0] > other.position[0]) {
          adjustedPos[0] = otherBox.max[0] + (testBox.max[0] - testBox.min[0]) / 2 + 0.01;
        } else {
          adjustedPos[0] = otherBox.min[0] - (testBox.max[0] - testBox.min[0]) / 2 - 0.01;
        }
      } else {
        // Push along Z
        if (adjustedPos[2] > other.position[2]) {
          adjustedPos[2] = otherBox.max[2] + (testBox.max[2] - testBox.min[2]) / 2 + 0.01;
        } else {
          adjustedPos[2] = otherBox.min[2] - (testBox.max[2] - testBox.min[2]) / 2 - 0.01;
        }
      }
    }
  }
  
  return {
    collides: hasCollision,
    adjustedPosition: adjustedPos,
  };
}

// Constrain position to ground plane bounds
export function constrainToGround(
  position: Vector3Tuple,
  groundSize: [number, number] = [20, 20]
): Vector3Tuple {
  const halfWidth = groundSize[0] / 2;
  const halfDepth = groundSize[1] / 2;
  
  return [
    Math.max(-halfWidth, Math.min(halfWidth, position[0])),
    Math.max(0, position[1]), // Don't go below ground
    Math.max(-halfDepth, Math.min(halfDepth, position[2])),
  ];
}

// Snap to grid (optional helper)
export function snapToGrid(position: Vector3Tuple, gridSize: number = 0.5): Vector3Tuple {
  return [
    Math.round(position[0] / gridSize) * gridSize,
    position[1],
    Math.round(position[2] / gridSize) * gridSize,
  ];
}
