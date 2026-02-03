import { useSandbox } from '@/context/SandboxContext';
import { SceneObject } from './SceneObject';

export function SceneObjects() {
  const { state, selectObject, setPointer } = useSandbox();

  const handlePointerOver = (objectId: string) => {
    setPointer({ hoveredObjectId: objectId, mode: 'hovering' });
  };

  const handlePointerOut = () => {
    if (state.pointer.mode === 'hovering') {
      setPointer({ hoveredObjectId: null, mode: 'idle' });
    }
  };

  const handleClick = (objectId: string) => {
    selectObject(state.selectedObjectId === objectId ? null : objectId);
  };

  return (
    <group>
      {state.objects.map((object) => (
        <SceneObject
          key={object.id}
          object={object}
          onPointerOver={() => handlePointerOver(object.id)}
          onPointerOut={handlePointerOut}
          onClick={() => handleClick(object.id)}
        />
      ))}
    </group>
  );
}

export default SceneObjects;
