import type { ShapeType, ColorName } from '@/types';

// Predefined shape configurations
export const SHAPES: Record<ShapeType, { name: string; defaultScale: [number, number, number] }> = {
  cube: {
    name: 'Cube',
    defaultScale: [1, 1, 1],
  },
  sphere: {
    name: 'Sphere',
    defaultScale: [0.5, 0.5, 0.5],
  },
  cylinder: {
    name: 'Cylinder',
    defaultScale: [0.5, 1, 0.5],
  },
  cone: {
    name: 'Cone',
    defaultScale: [0.5, 1, 0.5],
  },
  torus: {
    name: 'Torus',
    defaultScale: [0.5, 0.5, 0.5],
  },
  pyramid: {
    name: 'Pyramid',
    defaultScale: [1, 1, 1],
  },
};

// Predefined colors with hex values
export const COLORS: Record<ColorName, string> = {
  red: '#e53935',
  blue: '#1e88e5',
  green: '#43a047',
  yellow: '#fdd835',
  orange: '#fb8c00',
  purple: '#8e24aa',
  white: '#fafafa',
  black: '#212121',
};

// Color name to hex lookup
export const getColorHex = (colorName: ColorName): string => {
  return COLORS[colorName] || COLORS.blue;
};

// All available shape types
export const SHAPE_TYPES: ShapeType[] = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'pyramid'];

// All available color names
export const COLOR_NAMES: ColorName[] = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black'];

// Parse color name from voice input
export const parseColorName = (input: string): ColorName | null => {
  const normalized = input.toLowerCase().trim();
  const found = COLOR_NAMES.find(c => normalized.includes(c));
  return found || null;
};

// Parse shape type from voice input
export const parseShapeType = (input: string): ShapeType | null => {
  const normalized = input.toLowerCase().trim();
  const found = SHAPE_TYPES.find(s => normalized.includes(s));
  return found || null;
};

// Get random color
export const getRandomColor = (): string => {
  const colorKeys = Object.keys(COLORS) as ColorName[];
  const randomKey = colorKeys[Math.floor(Math.random() * colorKeys.length)];
  return COLORS[randomKey];
};

// Get random shape type
export const getRandomShapeType = (): ShapeType => {
  return SHAPE_TYPES[Math.floor(Math.random() * SHAPE_TYPES.length)];
};
