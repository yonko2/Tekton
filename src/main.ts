import * as THREE from 'three';
import { Hands, Results, NormalizedLandmark } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

// --- Three.js Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202020);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
const container = document.getElementById('canvas-container');
if (container) {
    container.appendChild(renderer.domElement);
}

// Lights
const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(1, 1, 1).normalize();
scene.add(directionalLight);

// Objects
const objects: THREE.Mesh[] = [];
const geometry = new THREE.BoxGeometry(1, 1, 1);

function createCube(color: number, x: number, y: number): THREE.Mesh {
    const material = new THREE.MeshPhongMaterial({ color: color });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.x = x;
    cube.position.y = y;
    cube.userData.originalColor = color;
    scene.add(cube);
    objects.push(cube);
    return cube;
}

createCube(0xff0000, -2, 0);
createCube(0xff0000, -5, 0);
createCube(0xff0000, -6, 0);
createCube(0xff0000, -7, 0);
createCube(0x00ff00, 0, 0);
createCube(0x0000ff, 2, 0);

// Resize handler
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- MediaPipe Setup ---
const videoElement = document.getElementById('input-video') as HTMLVideoElement;

// Cursor Setup
const cursorGeometry = new THREE.SphereGeometry(0.1, 32, 32);
// Disable depthTest to make it always visible
const cursorMaterial = new THREE.MeshBasicMaterial({ 
    color: 0xffffff,
    depthTest: false,
    depthWrite: false
});
const cursor = new THREE.Mesh(cursorGeometry, cursorMaterial);
// High renderOrder to ensure it's drawn on top
cursor.renderOrder = 999;
scene.add(cursor);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function onResults(results: Results) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        
        // Index finger tip is landmark 8
        const indexTip = landmarks[8];
        
        // Estimate depth based on hand size (distance between wrist and middle finger MCP)
        const wrist = landmarks[0];
        const middleFingerMCP = landmarks[9];
        const handSize = Math.sqrt(
            Math.pow(wrist.x - middleFingerMCP.x, 2) +
            Math.pow(wrist.y - middleFingerMCP.y, 2)
        );

        // Map handSize to Z coordinate
        // Larger handSize = closer to camera (positive Z)
        // Smaller handSize = further from camera (negative Z)
        // Tuned values: 0.05 (far) -> -5, 0.2 (close) -> 3
        const minSize = 0.05;
        const maxSize = 0.2;
        const minZ = -5;
        const maxZ = 3;
        
        let z = (handSize - minSize) / (maxSize - minSize) * (maxZ - minZ) + minZ;
        z = Math.max(minZ, Math.min(maxZ, z)); // Clamp Z value

        updateCursorPosition(indexTip.x, indexTip.y, z);
        checkGestures(landmarks);
    }
}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

hands.onResults(onResults);

if (videoElement) {
    const cameraUtils = new Camera(videoElement, {
      onFrame: async () => {
        await hands.send({image: videoElement});
      },
      width: 1280,
      height: 720
    });
    cameraUtils.start();
}

function updateCursorPosition(x: number, y: number, z: number) {
    // Map 2D video coordinates to 3D world coordinates
    // Video x: 0 (left) -> 1 (right)
    // Video y: 0 (top) -> 1 (bottom)
    
    // We need to calculate the visible width and height at the target depth
    // Camera is at z=5.
    const targetZ = z;
    const distance = camera.position.z - targetZ;
    const vFov = camera.fov * Math.PI / 180;
    const height = 2 * Math.tan(vFov / 2) * distance;
    const width = height * camera.aspect;
    
    // Mirror x (1-x) because it's a selfie camera
    const newX = (1 - x) * width - width / 2;
    // Invert y (1-y) because Three.js y is up, MediaPipe y is down
    const newY = (1 - y) * height - height / 2;
    
    cursor.position.set(newX, newY, targetZ);
}

let isPinching = false;
let grabbedObject: THREE.Mesh | null = null;
let initialHandRotation = 0;
let initialObjectRotation = 0;
const grabOffset = new THREE.Vector3();

function getHandRotation(landmarks: NormalizedLandmark[]): number {
    const wrist = landmarks[0];
    const middleFingerMCP = landmarks[9];
    // Calculate angle in radians relative to vertical axis
    // atan2(y, x) returns angle from x-axis.
    // We want rotation around Z.
    // Note: MediaPipe y is down, x is right.
    const dx = middleFingerMCP.x - wrist.x;
    const dy = middleFingerMCP.y - wrist.y;
    return -Math.atan2(dx, dy); // Negative to match Three.js rotation direction
}

function checkGestures(landmarks: NormalizedLandmark[]) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    
    // Calculate distance between thumb and index finger
    const distance = Math.sqrt(
        Math.pow(thumbTip.x - indexTip.x, 2) +
        Math.pow(thumbTip.y - indexTip.y, 2)
    );
    
    // Threshold for pinch
    if (distance < 0.1) {
        if (!isPinching) {
            isPinching = true;
            cursorMaterial.color.set(0xffff00); // Yellow when pinching
            tryGrab(landmarks);
        }
    } else {
        if (isPinching) {
            isPinching = false;
            cursorMaterial.color.set(0xffffff); // White when released
            releaseObject();
        }
    }
    
    if (grabbedObject) {
        grabbedObject.position.copy(cursor.position).add(grabOffset);
        
        // Apply rotation
        const currentHandRotation = getHandRotation(landmarks);
        const deltaRotation = currentHandRotation - initialHandRotation;
        grabbedObject.rotation.z = initialObjectRotation + deltaRotation;
    }
}

function tryGrab(landmarks: NormalizedLandmark[]) {
    const indexTip = landmarks[8];

    // Calculate Normalized Device Coordinates (NDC)
    // MediaPipe x: 0 (left) -> 1 (right). Mirrored: 1 (left) -> 0 (right)?
    // Wait, updateCursorPosition logic:
    // newX = (1 - x) * width - width / 2;
    // If x=0 (left of video), 1-x=1. newX is positive (Right of screen).
    // So x=0 is Right of screen.
    // NDC x: -1 (left) to 1 (right).
    // If x=0 -> Right -> NDC 1.
    // If x=1 -> Left -> NDC -1.
    // Formula: (1 - x) * 2 - 1
    // x=0 -> 1*2 - 1 = 1. Correct.
    // x=1 -> 0*2 - 1 = -1. Correct.
    
    pointer.x = (1 - indexTip.x) * 2 - 1;
    
    // MediaPipe y: 0 (top) -> 1 (bottom).
    // NDC y: 1 (top) -> -1 (bottom).
    // Formula: (1 - y) * 2 - 1
    // y=0 -> 1*2 - 1 = 1. Correct.
    // y=1 -> 0*2 - 1 = -1. Correct.
    pointer.y = (1 - indexTip.y) * 2 - 1;

    raycaster.setFromCamera(pointer, camera);

    const intersects = raycaster.intersectObjects(objects);

    if (intersects.length > 0) {
        // intersects is sorted by distance, so the first one is the closest (front)
        grabbedObject = intersects[0].object as THREE.Mesh;
        (grabbedObject.material as THREE.MeshPhongMaterial).color.set(0xff00ff); // Highlight grabbed object
        
        grabOffset.copy(grabbedObject.position).sub(cursor.position);

        // Store initial rotation state
        initialHandRotation = getHandRotation(landmarks);
        initialObjectRotation = grabbedObject.rotation.z;
    }
}

function releaseObject() {
    if (grabbedObject) {
        (grabbedObject.material as THREE.MeshPhongMaterial).color.setHex(grabbedObject.userData.originalColor);
        grabbedObject = null;
    }
}

// Animation Loop
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

animate();
