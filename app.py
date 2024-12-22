import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from net import Net as FreshnessNet  # Assuming your freshness model is loaded here

# Define class labels (update with your class labels)
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

# Load YOLOv8 model
model_Yolo = YOLO('yolov8n.pt')  # Update with your YOLO model file

# Load the Freshness Model
def get_freshness_model():
    model = FreshnessNet()
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

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

def show_detected_image(image, results):
    img = np.array(image)
    for result in results[0].boxes:
        bbox = result.xywh
        confidence = result.conf
        class_id = result.cls
        class_name = class_labels[int(class_id)]
        
        # Get coordinates for bounding box
        x1, y1, w, h = int(bbox[0] - w/2), int(bbox[1] - h/2), int(w), int(h)
        
        # Draw rectangle and label
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Freshness Check
        freshness_percentage = freshness_percentage_by_cv_image(img)
        freshness = freshness_label(freshness_percentage)
        cv2.putText(img, f'Freshness: {freshness} ({freshness_percentage}%)', (x1, y1 + h + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return Image.fromarray(img)

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
        return "Fresh!"  # Default to fresh if something goes wrong

def detect_objects(image):
    # Run detection on the image
    results = model_Yolo(image)
    return results

def main():
    st.title("Object Detection and Freshness Check")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform object detection
        results = detect_objects(image)

        # Show detected image with bounding boxes and freshness
        detected_image = show_detected_image(image, results)
        st.image(detected_image, caption="Detected Objects and Freshness", use_column_width=True)

        # Display detection results and freshness information
        for result in results[0].boxes:
            bbox = result.xywh
            confidence = result.conf
            class_id = result.cls
            class_name = class_labels[int(class_id)]
            freshness_percentage = freshness_percentage_by_cv_image(np.array(image))
            freshness = freshness_label(freshness_percentage)
            st.write(f"Detected {class_name} with confidence {confidence:.2f} at {bbox} - Freshness: {freshness} ({freshness_percentage}%)")

if __name__ == "__main__":
    main()
