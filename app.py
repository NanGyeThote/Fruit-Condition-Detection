import streamlit as st
import requests
from streamlit_lottie import st_lottie
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from net import Net as FreshnessNet
from net1 import Net as ClassificationNet
import os

# Function to load Lottie animation from URL
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()  # Ensure the request was successful
        return r.json()  # Return the JSON data
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Lottie animation: {e}")
        return None  # Return None if there's an error

# Load a safe and valid Lottie animation URL
welcome_animation_url = "https://assets1.lottiefiles.com/packages/lf20_J2iz7Z.json"  # Example Lottie animation URL
welcome_animation = load_lottie_url(welcome_animation_url)

# Setup Streamlit layout with tabs
st.set_page_config(page_title="Fruit Condition Detection", layout="wide")
PAGES = {
    "Home": "home_page",
    "Upload Image": "upload_page",
}

# Sidebar for navigation
page = st.sidebar.radio("Select a Page", list(PAGES.keys()))

# Function to load models
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
        YOLO_MODEL = YOLO("freshify.pt")
    return YOLO_MODEL

# Function for freshness label based on percentage
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
        return "Fresh!"

# Function to calculate best before date based on freshness percentage
def calculate_best_before(freshness_percentage):
    if freshness_percentage > 90:
        return "For about next three days"
    elif freshness_percentage > 75:
        return "For about next two days"
    elif freshness_percentage > 65:
        return "For about one day"
    elif freshness_percentage > 50:
        return "For about Today"
    else:
        return "Dear customer, you should not eat!"

# Image preprocessing for freshness checking
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

# Image classification for fruits and vegetables
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

# Object detection with YOLO
def process_frame(cv_image):
    yolo_model = get_yolo_model()
    classNames = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    results = yolo_model(cv_image)
    detection_results = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = cv_image[y1:y2, x1:x2]
            freshness_percentage = freshness_percentage_by_cv_image(roi)
            fruit_vegetable_name = classify_fruit_vegetable(roi)
            cls = int(box.cls[0])
            txt = "Condition"
            label_text = f"{txt} - {freshness_label(freshness_percentage)}"
            class_name = classNames[cls]
            if class_name == 'person':
                pass
            else:
                cv2.putText(cv_image, f'{label_text}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv_image

# Home page with Lottie animation
def home_page():
    st.title("Welcome to Fruit Condition Detection!")
    if welcome_animation:
        st_lottie(welcome_animation, speed=1, width=600, height=300)
    else:
        st.write("Animation could not be loaded.")
    
    st.write("This app helps you detect the freshness of fruits and vegetables using image and video uploads. Please upload an image to get started.")

# Upload page for image upload and detection
def upload_page():
    st.title("Upload Image for Fruit Condition Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image with OpenCV
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Display uploaded image
        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
        
        # Process image for freshness detection and classification
        processed_img = process_frame(img)  # Process the image with object detection and freshness checking
        
        # Display processed image with bounding boxes and freshness condition
        st.image(processed_img, caption="Processed Image", use_column_width=True)

# Class labels for classification
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

# Render the correct page based on the selected tab
if page == "Home":
    home_page()
elif page == "Upload Image":
    upload_page()
