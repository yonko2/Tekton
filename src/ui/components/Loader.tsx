import React from 'react';

type Props = {
  visible: boolean;
};

export default function Loader({ visible }: Props) {
  return (
    <div id="loader" style={{ opacity: visible ? 1 : 0 }}>
      Initializing 2D Vision...
    </div>
  );
}

