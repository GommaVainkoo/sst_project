import torch
import cv2
import numpy as np
import os

# Load the YOLOv5 model (you need to replace 'model.pt' with the path to your YOLOv8 model)
model_path1 = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'drone.pt')

model = torch.hub.load('ultralytics/yolov5', 'custom', model_path1)


# Define the function to call when an object is detected
def on_object_detected(class_name):
    # Replace this with your desired function to call when an object is detected
    print(f"Object detected: {class_name}")


# Open the webcam
cap = cv2.VideoCapture(0)  # 0 indicates the default camera

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Access the detected objects
    for pred in results.pred[0]:
        class_id = int(pred[5])
        class_name = model.names[class_id]
        confidence = float(pred[4])
        bbox = pred[:4].tolist()

        if confidence > 0.38:  # You can adjust the confidence threshold as needed
            # Call the function when an object is detected
            on_object_detected(class_name)

            # Draw bounding boxes on detected objects
            color = (0, 255, 0)  # Green
            thickness = 2
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Display the webcam feed with bounding boxes
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
