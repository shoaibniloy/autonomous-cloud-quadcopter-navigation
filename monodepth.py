import cv2
import torch
import numpy as np
import time

from depth_anything_v2.dpt import DepthAnythingV2

# Check if CUDA is available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
}

# Choose encoder
encoder = 'vits'

# Load the model
model = DepthAnythingV2(**model_configs[encoder])

# Load model weights and move to the chosen device (GPU or CPU)
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=DEVICE))
model = model.to(DEVICE).eval()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Resize to a lower resolution, e.g., 320x240
cap.set(3, 160)  # Set the width
cap.set(4, 120)  # Set the height

# Prepare to save the video output
out = cv2.VideoWriter('depth_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (320, 240))

# Start the frame rate counter
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to a NumPy array (already in BGR format from the webcam)
    frame_np = np.array(frame)

    # Infer depth on the captured frame
    with torch.no_grad():
        depth = model.infer_image(frame_np)  # Pass the NumPy array directly to the model

    # Normalize the depth map to fit within the 0-255 range
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)

    # Apply color map to the depth image (using JET for color scaling)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Compute the nearest and farthest depth values (in mm)
    min_depth = np.min(depth)  # Nearest depth
    max_depth = np.max(depth)  # Farthest depth
    
    # Convert the nearest and farthest depth to millimeters
    min_depth_in_mm = min_depth  # Assumes the model outputs depth in millimeters
    max_depth_in_mm = max_depth  # Assumes the model outputs depth in millimeters

    # Compute the Hue of the nearest and farthest pixels
    # For nearest depth
    nearest_depth_location = np.unravel_index(np.argmin(depth), depth.shape)
    nearest_bgr = depth_colored[nearest_depth_location]  # Color at nearest depth
    nearest_hsv = cv2.cvtColor(np.uint8([[nearest_bgr]]), cv2.COLOR_BGR2HSV)
    nearest_hue = nearest_hsv[0][0][0]  # Extract hue value

    # For farthest depth
    farthest_depth_location = np.unravel_index(np.argmax(depth), depth.shape)
    farthest_bgr = depth_colored[farthest_depth_location]  # Color at farthest depth
    farthest_hsv = cv2.cvtColor(np.uint8([[farthest_bgr]]), cv2.COLOR_BGR2HSV)
    farthest_hue = farthest_hsv[0][0][0]  # Extract hue value

    # Display FPS
    fps = 1 / (time.time() - start_time)
    start_time = time.time()

    # Display the FPS, nearest and farthest depths, and corresponding hues
    cv2.putText(depth_colored, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(depth_colored, f'Nearest Depth: {min_depth_in_mm:.2f} mm', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(depth_colored, f'Nearest Hue: {nearest_hue}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(depth_colored, f'Farthest Depth: {max_depth_in_mm:.2f} mm', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(depth_colored, f'Farthest Hue: {farthest_hue}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the resulting frame with the colored depth map
    cv2.imshow('Webcam Depth', depth_colored)

    # Save video
    out.write(depth_colored)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()