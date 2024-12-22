# Fruit Condition Detection Web Application

This is a Streamlit web application for detecting the freshness of fruits and vegetables from images. It uses object detection and freshness classification to determine the condition of various fruits/vegetables and provides a freshness label (Fresh, Good, Fair, Poor) along with the best-before information.

## Models Used

### 1. **YOLO (You Only Look Once) - Object Detection**
   - **Model Type**: Object Detection
   - **Purpose**: Detects fruits and vegetables in the input image.
   - **Model Source**: `freshify.pt` (A pre-trained YOLO model for detecting fruits and vegetables)
   - **Model Framework**: [Ultralytics YOLO](https://github.com/ultralytics/yolov5)

### 2. **Freshness Detection Model**
   - **Model Type**: Image Classification
   - **Purpose**: Classifies the freshness level of fruits/vegetables (Fresh, Good, Fair, Poor).
   - **Model Framework**: PyTorch
   - **Model Architecture**: A custom CNN model (`Net` from `net.py`)
   - **Pre-trained Weights**: `model.pt`

### 3. **Fruit and Vegetable Classification Model**
   - **Model Type**: Image Classification
   - **Purpose**: Classifies fruits and vegetables into various categories (apple, banana, cucumber, etc.).
   - **Model Framework**: PyTorch
   - **Model Architecture**: A custom CNN model (`Net` from `net1.py`)
   - **Pre-trained Weights**: `final.pth`

## Project Setup

### Prerequisites

1. Python 3.7 or higher
2. The following Python packages are required:

   - `streamlit`: To create the web application.
   - `opencv-python`: For image processing.
   - `torch`, `torchvision`: For loading the trained models and performing inference.
   - `requests`: For loading the Lottie animation in the UI.
   - `streamlit-lottie`: To display Lottie animations.

### Installing Dependencies

To install the required dependencies, you can run:

```bash
pip install streamlit opencv-python numpy requests streamlit-lottie torch torchvision ultralytics
```

