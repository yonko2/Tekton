import type { ShapeType, ColorName, Vector3Tuple } from '@/types';

// ── Shape configurations ─────────────────────────────────────
export interface ShapeConfig {
  name: string;
  defaultScale: Vector3Tuple;
}

export const SHAPES: Record<ShapeType, ShapeConfig> = {
  cube: { name: 'Cube', defaultScale: [1, 1, 1] },
  sphere: { name: 'Sphere', defaultScale: [1, 1, 1] },
  cylinder: { name: 'Cylinder', defaultScale: [1, 1, 1] },
  pyramid: { name: 'Pyramid', defaultScale: [1, 1, 1] },
};

// ── Color palette ────────────────────────────────────────────
export const COLORS: Record<ColorName, string> = {
  red: '#e53935',
  blue: '#1e88e5',
  green: '#43a047',
  yellow: '#fdd835',
  orange: '#fb8c00',
  purple: '#8e24aa',
  white: '#fafafa',
  gray: '#757575',
};

// ── Lookups ──────────────────────────────────────────────────
export const SHAPE_TYPES: ShapeType[] = ['cube', 'sphere', 'cylinder', 'pyramid'];
export const COLOR_NAMES: ColorName[] = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'gray'];

export const getColorHex = (name: ColorName): string => COLORS[name] ?? COLORS.blue;

// ── Voice-input parsers ──────────────────────────────────────
export const parseShapeType = (input: string): ShapeType | null => {
  const lower = input.toLowerCase().trim();
  // Handle "box" as synonym for "cube"
  if (lower.includes('box')) return 'cube';
  return SHAPE_TYPES.find((s) => lower.includes(s)) ?? null;
};

export const parseColorName = (input: string): ColorName | null => {
  const lower = input.toLowerCase().trim();
  if (lower.includes('grey')) return 'gray';
  return COLOR_NAMES.find((c) => lower.includes(c)) ?? null;
};

// ── Randoms ──────────────────────────────────────────────────
export const getRandomColor = (): string => {
  const keys = Object.keys(COLORS) as ColorName[];
  return COLORS[keys[Math.floor(Math.random() * keys.length)]];
};

export const getRandomShapeType = (): ShapeType =>
  SHAPE_TYPES[Math.floor(Math.random() * SHAPE_TYPES.length)];
