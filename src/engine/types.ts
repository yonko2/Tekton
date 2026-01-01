export type Point2 = { x: number; y: number };

export type PhysicsBody = {
  id: number;
  kind: 'circle' | 'box';
  pos: Point2;
  vel: Point2;
  radius?: number;
  size?: { w: number; h: number };
  color: string;
  restitution: number;
  friction: number;
  mass: number;
};

