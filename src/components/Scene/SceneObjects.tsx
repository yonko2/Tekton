import { useCallback } from 'react';
import { useSandbox } from '@/context/SandboxContext';
import { PhysicsObject } from './PhysicsObject';

export function SceneObjects() {
  const { state, selectObject, setPointer } = useSandbox();

  const handlePointerOver = useCallback(
    (id: string) => {
      setPointer({ hoveredObjectId: id });
    },
    [setPointer],
  );

  const handlePointerOut = useCallback(() => {
    setPointer({ hoveredObjectId: null });
  }, [setPointer]);

  const handleClick = useCallback(
    (id: string) => {
      selectObject(id);
    },
    [selectObject],
  );

  return (
    <>
      {state.objects.map((obj) => (
        <PhysicsObject
          key={obj.id}
          object={obj}
          onPointerOver={() => handlePointerOver(obj.id)}
          onPointerOut={handlePointerOut}
          onClick={() => handleClick(obj.id)}
        />
      ))}
    </>
  );
}
