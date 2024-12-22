# Fruit Condition Detection Web Application

This is a Streamlit web application for detecting the freshness of fruits and vegetables from images. It uses object detection and freshness classification to determine the condition of various fruits/vegetables and provides a freshness label (Fresh, Good, Fair, Poor) along with the best-before information.

## Screenshots 📸

![mainpage](img1.jpg)

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
## Files Structure
```bash
.
├── app.py                # Main Streamlit app file
├── net.py                # Freshness detection model architecture
├── net1.py               # Fruit/Vegetable classification model architecture
├── model.pt             # Trained model for freshness detection
├── final.pth            # Trained model for fruit/vegetable classification
├── freshify.pt          # YOLO model for object detection
└── assets/
    └── welcome_animation.json   # Lottie animation JSON file (if you prefer local storage)
```

## Running the Application

1. Clone this repository or download the files to your local system.

2. Ensure that all required dependencies are installed (see the "Installing Dependencies" section).

3. Run the application using the following command:

```bash
streamlit run app.py
```

4. The application will be accessible at http://localhost:8501 in your browser.

## Using the Application

1. Home Page:

- The home page introduces the app with a welcoming Lottie animation.
  
2. Upload Image Page:

- Upload an image of fruits or vegetables using the file uploader.
- The app will display the image with bounding boxes around detected objects.
- It will also display the freshness label and the best-before date for each detected item.

3. Detection Process:

- The app uses YOLO for object detection to find fruits and vegetables.
- For each detected object, the app applies the freshness detection model to determine its condition.
- The fruit or vegetable is classified using the fruit/vegetable classification model, and the freshness percentage is calculated.

4. Freshness Labels:

- Freshness levels are classified as:
   - Fresh (above 90%)
   - Good (above 65%)
   - Fair (above 50%)
   - Poor (below 50%)

- The app will also recommend the best-before date based on freshness percentage.

## Lottie Animations

The application uses Lottie animations for a more interactive and engaging UI experience. The animations are displayed using the streamlit-lottie library. If you prefer to use a local file, place your .json animation file in the assets folder.

## Models and Weights

- **YOLO Model (freshify.pt)**: A pre-trained model for detecting fruits and vegetables.
- **Freshness Model (model.pt)**: A custom model for predicting the freshness of fruits/vegetables.
- **Classification Model (final.pth)**: A custom model for classifying fruits and vegetables.
You can replace these models with your own pre-trained models if needed. Just ensure the models are compatible with PyTorch and follow the same structure.

## Troubleshooting

- **Lottie Animation Not Loading**: Make sure you have an active internet connection for the Lottie animation URL to be fetched. You can also download and store animations locally.
- **Model Errors**: Ensure the models (freshify.pt, model.pt, final.pth) are correctly placed in the project directory.
- **Dependencies Not Found**: Ensure all required Python packages are installed as per the instructions above.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```bash

### Explanation:

1. **Models**:
   - **YOLO**: Used for detecting fruits and vegetables in images.
   - **Freshness Detection Model**: Classifies the freshness of fruits/vegetables.
   - **Fruit/Vegetable Classification Model**: Identifies the type of fruit or vegetable.

2. **Setup Instructions**:
   - Describes how to install dependencies and run the application.
   - Provides a folder structure and clarifies where models and the Lottie animation file should be placed.

3. **Usage**:
   - Explains the flow of the app, from uploading an image to receiving freshness and classification results.

4. **Troubleshooting**:
   - Provides solutions for common issues like Lottie animation not loading or model errors.

### Final Steps:
- Just copy this `README.md` file into your project directory, and it will serve as an easy guide for anyone who wants to use your project.

Let me know if you need any further adjustments!
```
