import streamlit as st
import torch.nn as nn
import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from net import Net as FreshnessNet
from net1 import Net as ClassificationNet
import time
import streamlit.components.v1 as components

# Global model variables
FRESHNESS_MODEL = None
FV_MODEL = None
YOLO_MODEL = None
class_labels = {
    0: 'apple', 1: 'banana', 2: 'cucumber', 3: 'grape', 4: 'guava', 
    5: 'mango', 6: 'orange', 7: 'pineapple', 8: 'strawberry', 9: 'tomato', 
    10: 'watermelon'
}

# Freshness Model
def get_freshness_model():
    global FRESHNESS_MODEL
    if FRESHNESS_MODEL is None:
        FRESHNESS_MODEL = FreshnessNet()
        FRESHNESS_MODEL.load_state_dict(
            torch.load("model.pt", map_location=torch.device("cpu"))
        )
        FRESHNESS_MODEL.eval()
    return FRESHNESS_MODEL

# Fruit/Vegetable Classification Model
def get_fv_model():
    global FV_MODEL
    if FV_MODEL is None:
        FV_MODEL = ClassificationNet(num_classes=11)
        FV_MODEL.load_state_dict(
            torch.load("final.pth", map_location=torch.device("cpu"))
        )
        FV_MODEL.eval()
    return FV_MODEL

# YOLO Model
def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO("freshify.pt")
    return YOLO_MODEL

# Freshness condition based on percentage
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

# Calculate best before date
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

# Freshness percentage based on the image
def freshness_percentage_by_cv_image(cv_image):
    mean = (0.7369, 0.6360, 0.5318)
    std = (0.3281, 0.3417, 0.3704)
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
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

# Classify fruit/vegetable based on the image
def classify_fruit_vegetable(cv_image):
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    model = get_fv_model()
    with torch.no_grad():
        out = model(batch)
    _, predicted = torch.max(out, 1)
    return class_labels[predicted.item()]

# Process frame with both object detection and freshness checking
def process_frame_with_condition(cv_image):
    yolo_model = get_yolo_model()
    classNames = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    results = yolo_model(cv_image)
    detection_results = []

    # Loop through the detected objects from YOLO
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            roi = cv_image[y1:y2, x1:x2]  # Crop the region of interest (ROI)

            # Freshness assessment
            freshness_percentage = freshness_percentage_by_cv_image(roi)
            freshness_condition = freshness_label(freshness_percentage)
            best_before = calculate_best_before(freshness_percentage)

            # Classify the fruit/vegetable in the ROI
            fruit_vegetable_name = classify_fruit_vegetable(roi)

            # Get the class name from YOLO
            cls = int(box.cls[0])
            class_name = classNames[cls]  # This will be the detected object class (e.g., 'apple', 'banana', etc.)

            # Add text to the image (object name, freshness condition, best before)
            label_text = f'{class_name}: {freshness_condition}, {best_before}'
            cv2.putText(cv_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

            # Collect the detection result (can be passed to frontend or logged)
            detection_results.append({
                'name': class_name,
                'freshness': freshness_condition,
                'best_before': best_before,
            })

    return cv_image, detection_results

# Home Page Content
def home_page():
    st.title("Fruit & Vegetable Freshness and Condition Detection")
    st.write("Welcome to our Fruit and Vegetable Freshness and Condition Detection App! Upload an image of a fruit or vegetable and our model will detect the object, assess its freshness, and tell you its condition.")
    st.markdown("""
    ### How It Works:
    1. Upload an image of a fruit or vegetable.
    2. Our YOLO-based detection model identifies the object.
    3. Freshness condition is calculated using a deep learning model.
    4. We provide an estimated best-before date based on the freshness percentage.
    """)
    st.write("Experience real-time analysis of your food's freshness!")

# Detection Page Content
def detection_page():
    st.title("Upload Image for Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Process the image with object detection and freshness checking
        processed_img, detection_results = process_frame_with_condition(img)

        # Show processed image with bounding boxes and labels
        st.image(processed_img, channels='BGR', caption="Processed Image", use_column_width=True)

        # Display detection results
        if detection_results:
            st.subheader("Detection Results:")
            for result in detection_results:
                st.write(f"Object: {result['name']}, Freshness: {result['freshness']}, Best Before: {result['best_before']}")
        else:
            st.write("No objects detected.")

# Streamlit page navigation
PAGES = {
    "Home": home_page,
    "Detection": detection_page,
}

# Select which page to display
page = st.sidebar.selectbox("Choose a page", options=PAGES.keys())
PAGES[page]()

# Animation using Lottie or simple CSS animation
st.markdown("""
<style>
@keyframes fadeIn { 
  0% { opacity: 0; }
  100% { opacity: 1; }
}
.fadeIn {
  animation: fadeIn 3s ease-in-out;
}

.stSelectbox div:hover {
    cursor: pointer;
}
</style>
<div class="fadeIn">
    <h2>Enjoy our Fruit & Vegetable Detection App!</h2>
</div>
""", unsafe_allow_html=True)

