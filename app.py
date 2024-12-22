import streamlit as st
import torch.nn as nn
import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from net import Net as FreshnessNet
from net1 import Net as ClassificationNet
import os

# Global model variables
FRESHNESS_MODEL = None
FV_MODEL = None
YOLO_MODEL = None
model_Yolo = None

# Class labels for fruits and vegetables
class_labels = {
    0: 'apple',
    1: 'banana',
    2: 'cucumber',
    3: 'grape',
    4: 'guava',
    5: 'mango',
    6: 'orange',
    7: 'pineapple',
    8: 'strawberry',
    9: 'tomato',
    10: 'watermelon'
}

# Load the models
def get_freshness_model():
    global FRESHNESS_MODEL
    if FRESHNESS_MODEL is None:
        FRESHNESS_MODEL = FreshnessNet()
        FRESHNESS_MODEL.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
        FRESHNESS_MODEL.eval()
    return FRESHNESS_MODEL

def get_fv_model():
    global FV_MODEL
    if FV_MODEL is None:
        FV_MODEL = ClassificationNet(num_classes=11)
        FV_MODEL.load_state_dict(torch.load("final.pth", map_location=torch.device("cpu")))
        FV_MODEL.eval()
    return FV_MODEL

def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO("freshify.pt")  # Path to your YOLO model
    return YOLO_MODEL

# def get_yolov8n_model():
#     global model_Yolo
#     if model_Yolo is None:
#         model_Yolo = YOLO('yolov8n.pt')  # Path to your YOLO model
#     return model_Yolo

# Freshness conditions
def freshness_label(freshness_percentage):
    if freshness_percentage > 90:
        return "Fresh!"
    elif freshness_percentage > 65:
        return "Good!"
    elif freshness_percentage > 50:
        return "Fair!"
    elif freshness_percentage > 0 and freshness_percentage < 10:
        return "Poor!"
    else:
        return "Fresh!"  # Default to fresh if percentage is invalid

def freshness_percentage_by_cv_image(cv_image):
    mean = (0.7369, 0.6360, 0.5318)
    std = (0.3281, 0.3417, 0.3704)
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    model = get_freshness_model()
    with torch.no_grad():
        out = model(batch)
    s = nn.Softmax(dim=1)
    result = s(out)
    return int(result[0][0].item() * 100)

# Classify fruit/vegetable
def classify_fruit_vegetable(cv_image):
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    model = get_fv_model()
    with torch.no_grad():
        out = model(batch)
    _, predicted = torch.max(out, 1)
    return class_labels[predicted.item()]

# Object detection using YOLO
def detect_objects_with_yolo(cv_image):
    model = get_yolo_model()
    results = model(cv_image)
    return results

# Streamlit interface
st.title("Freshness and Object Detection App")

# Upload image
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "MOV"])

if uploaded_file is not None:
    # Read the image
    file_bytes = uploaded_file.read()
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Show the uploaded image
    st.image(img, channels="BGR", caption="Uploaded Image")

    # Perform object detection
    detection_results = detect_objects_with_yolo(img)
    
    # Display detected bounding boxes on the image
    for bbox in detection_results.xywh[0]:  # Assuming we are working with the first image in the batch
        x, y, w, h, conf, cls = bbox  # Get bounding box coordinates, confidence, and class ID
        class_id = int(cls)
        label = class_labels.get(class_id, "Unknown")
        color = (0, 255, 0)  # Green bounding box color
        
        # Draw bounding box on the image
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the image with bounding boxes
    st.image(img, channels="BGR", caption="Image with Object Detection")

    # Perform classification on the entire image
    class_name = classify_fruit_vegetable(img)
    st.write(f"Classified as: {class_name}")
    
    # Calculate freshness percentage
    freshness_percentage = freshness_percentage_by_cv_image(img)
    
    # Display freshness condition
    freshness_condition = freshness_label(freshness_percentage)
    st.write(f"Freshness: {freshness_condition} ({freshness_percentage}%)")
