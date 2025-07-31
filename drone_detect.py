import cv2
from ultralytics import YOLO
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
ws, hs = 860, 640  # Reduced resolution for faster processing
cap.set(3, ws)  # Set width
cap.set(4, hs)  # Set height

if not cap.isOpened():
    print("Error: Could not access camera.")
    exit()

# Load the trained YOLOv11 model
model = YOLO("drone_detect.pt")  # Path to your model file

# Define rectangle in the middle of the screen (larger for better coverage)
rect_width, rect_height = 300, 300  # Increased rectangle size
rect_top_left = (ws // 2 - rect_width // 2, hs // 2 - rect_height // 2)
rect_bottom_right = (ws // 2 + rect_width // 2, hs // 2 + rect_height // 2)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame.")
        continue

    # Preprocess image: Enhance contrast
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # Increase contrast and brightness

    # Run drone detection with lower confidence and adjusted IoU
    results = model.predict(
        img,
        conf=0.3,  # Lowered confidence threshold for higher sensitivity
        iou=0.5,   # Adjusted IoU threshold for overlapping boxes
        augment=True,  # Enable test-time augmentation
        verbose=False
    )
    bboxs = results[0].boxes  # Get bounding boxes

    # Count detected drones
    num_drones = len(bboxs)

    # Check if any part of a drone's bounding box overlaps with the rectangle
    inside_rect = False
    for box in bboxs:
        if box.cls == 0:  # Ensure it's a drone (class 0)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Box coordinates

            # Check if any part of the bounding box overlaps with the rectangle
            if not (x2 < rect_top_left[0] or x1 > rect_bottom_right[0] or
                    y2 < rect_top_left[1] or y1 > rect_bottom_right[1]):
                inside_rect = True

            # Draw green rectangle for detected drone
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label with confidence
            conf = box.conf.cpu().numpy()[0]  # Confidence score
            label = f"drone {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # Draw center rectangle (green if drone inside, red otherwise)
    rect_color = (0, 255, 0) if inside_rect else (0, 0, 255)
    cv2.rectangle(img, rect_top_left, rect_bottom_right, rect_color, 2)

    # Display number of detected drones
    text = f"Drones Detected: {num_drones}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Drone Detection", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()