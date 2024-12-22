import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
from torchvision import transforms

# Load your YOLO model and other necessary models (ensure you use correct paths)
yolo_model = YOLO("freshify.pt")  # Ensure your trained model is in the correct path

# Class labels for fruit/vegetable classification (example)
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

# Freshness detection (implement your own model or use an existing one)
def freshness_percentage_by_cv_image(cv_image):
    # Implement the freshness checker model
    # This function should return the freshness percentage of the fruit
    return 85  # Placeholder value for now

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

def show_detected_image(image, results):
    img = np.array(image)
    for result in results[0].boxes:
        bbox = result.xywh  # This should be a tensor or array containing bounding box details
        
        # Display bbox data in Streamlit for inspection
        st.write(f"Bounding box data: {bbox}")  # This will print the bbox to the Streamlit interface
        
        # Check if bbox has the correct number of elements (4)
        if len(bbox) >= 4:
            center_x, center_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Calculate top-left coordinates (x1, y1)
            x1 = int(center_x - w / 2)
            y1 = int(center_y - h / 2)
            
            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
            cv2.putText(img, f'{class_labels[int(result.cls)]} {result.conf:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Freshness Check
            freshness_percentage = freshness_percentage_by_cv_image(img)
            freshness = freshness_label(freshness_percentage)
            cv2.putText(img, f'Freshness: {freshness} ({freshness_percentage}%)', 
                        (x1, y1 + int(h) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            st.write(f"Invalid bbox structure: {bbox}")  # Log the invalid bbox structure
    
    # Convert the image to PIL format for Streamlit
    return Image.fromarray(img)

def main():
    # Title of the Streamlit app
    st.title("Fruit Condition Detection")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image using PIL
        image = Image.open(uploaded_file)
        
        # Run YOLO object detection
        results = yolo_model(image)  # Ensure this returns the detection results
        
        # Show the detected image with bounding boxes and freshness info
        detected_image = show_detected_image(image, results)
        st.image(detected_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
