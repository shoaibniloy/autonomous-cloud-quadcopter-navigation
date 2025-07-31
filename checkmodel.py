import cv2
from ultralytics import YOLO
import torch

# Ensure CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv8 model (epoch25.pt) and move it to GPU (CUDA)
model = YOLO('fighterjets.pt').to(device)

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 640x640 (or another size divisible by 32)
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert the resized frame to a float32 tensor and normalize the pixel values
    frame_tensor = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Move the frame tensor to GPU
    frame_tensor = frame_tensor.to(device)

    # Perform inference with the YOLO model on the GPU
    results = model(frame_tensor)

    # Extract the detections
    detections = results[0]

    # If there are detections, draw bounding boxes
    if detections:
        for det in detections.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()  # Get bounding box coordinates
            conf = det.conf[0].item()  # Get confidence score
            cls = det.cls[0].item()  # Get class ID

            # Only draw the bounding box if confidence is greater than 0.5
            if conf > 0.5:
                # Draw the bounding box on the frame (Green color)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Display the class name and confidence score
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
