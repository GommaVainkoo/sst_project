import cv2
import io
import os
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
from datetime import datetime, timedelta


def save_photo(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

    bio = io.BytesIO()
    bio.name = 'image.jpeg'
    img.save(bio, 'JPEG')
    bio.seek(0)

model_path1 = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'drone.pt')
model = YOLO(model_path1)
model1 = YOLO("yolov8n.pt")
classes1 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

classes=['drone']
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture(0)
fps_camera = cap.get(cv2.CAP_PROP_FPS)
target_fps = 10
n = int(fps_camera / target_fps)
frame_counter, not_detected = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % n == 0:
        outs = model(frame, task='detect', iou=0.2, conf=0.3, show=True, save_conf=True, classes=[0], boxes=True)

        pred_classes = [classes[int(i.item())] for i in outs[0].boxes.cls]
        pred_bbox = [i.tolist() for i in outs[0].boxes.xywh]


        length = len(pred_classes)
        drone_boxes= []
        drone_flag=0
        for i in range(length):
            if pred_classes[i] in ['drone']:

                outs1 = model1(frame, task='detect', iou=0.2, conf=0.3, classes=[0, 14])
                pred_classes1 = [classes1[int(i.item())] for i in outs1[0].boxes.cls]
                if pred_classes1[i] in ['person','bird']:
                    print("NOT A DRONE")
                else:
                    drone_boxes.append((round(pred_bbox[i][0]), round(pred_bbox[i][1]), round(pred_bbox[i][0] + pred_bbox[i][2]), round(pred_bbox[i][1] + pred_bbox[i][3])))
                    drone_flag = 1
                    print("Drone detected")
                    save_photo(frame)



    drone_flag = 0
    drone_boxes = [(0, 0, 0, 0)]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()