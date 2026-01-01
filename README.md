# 3D Hand Gesture Interaction System

This project uses Three.js and MediaPipe Hands to create a real-time interactive 3D system where you can move objects with hand gestures.

## Prerequisites

- Node.js installed.
- A webcam.

## Setup

1.  Install dependencies:
    ```bash
    npm install
    ```

2.  Start the development server:
    ```bash
    npm run dev
    ```

3.  Open the URL provided by Vite (usually `http://localhost:5173`) in your browser.
4.  Allow camera access when prompted.

## Usage

-   **Move Hand**: Move your hand in front of the camera to move the white cursor in the 3D space.
-   **Grab Object**: Pinch your index finger and thumb together while the cursor is near a cube to grab it. The cursor turns yellow and the object turns purple.
-   **Move Object**: Move your hand while pinching to move the object.
-   **Release Object**: Release the pinch to drop the object.
