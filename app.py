import streamlit as st
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO
import os
import numpy as np

# Global model variables
YOLO_MODEL = None

# Class labels
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

# Freshness labels based on percentage
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
        return "Spoiled"

# Function to load YOLO model
def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO("freshify.pt")
    return YOLO_MODEL

# Function to classify fruit or vegetable (Example classification model)
def classify_fruit_vegetable(cv_image):
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    # Assuming you have a model for classification
    # Replace with the correct model to classify
    model = torch.load("your_model.pth", map_location=torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    _, predicted = torch.max(out, 1)
    return class_labels[predicted.item()]

# Function to calculate freshness percentage (dummy logic)
def freshness_percentage_by_cv_image(cv_image):
    # Simple dummy logic for freshness percentage
    # You can replace this with an actual model or logic
    return np.random.randint(10, 100)  # Random percentage between 10 and 100

# Function to process frame and add bounding boxes and freshness condition
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
            class_name = classNames[cls]
            
            # Determine the freshness label
            freshness = freshness_label(freshness_percentage)
            
            # Draw bounding box and label on the image
            color = (0, 255, 0)  # Green color for the bounding box and text
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} - {freshness}"
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            detection_results.append({'name': class_name, 'condition': freshness})

    return cv_image, detection_results

# Streamlit Image Upload
st.title("Fruit and Vegetable Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image and process it
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    processed_img, detection_results = process_frame(img)

    # Display the processed image with bounding boxes and freshness condition
    st.image(processed_img, channels="BGR", caption="Processed Image", use_column_width=True)
    
    # Show detection results with condition
    if detection_results:
        st.write("Detected Objects and Conditions:")
        for result in detection_results:
            st.write(f"Class: {result['name']}, Condition: {result['condition']}")
    else:
        st.write("No objects detected.")
