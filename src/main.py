import torch
from ultralytics import YOLO
import os
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2

# Load a model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'bird.pt')
model_path1 = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'drone.pt')
#
model = YOLO(model_path1)

#  # build a new model from scratch
# threshold = 0.5
# image = cv2.imread('C:\\Users\\ajsha\\Desktop\\Drone Detection\\Drone_data\\73.jpg')
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="config.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format~

result = model.predict(source="0", show=True)




