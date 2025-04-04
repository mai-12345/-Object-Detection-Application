# Object Detection Application

This is a Streamlit-based object detection application using MediaPipe's Object Detection module. The application detects objects in uploaded images and displays bounding boxes with labels and confidence scores.

## Features

- Upload an image in `.jpg`, `.png`, `.jpeg`, or `.webp` format.
- Object detection is performed using the EfficientDet Lite model (`efficientdet_lite0.tflite`).
- Bounding boxes are drawn around detected objects along with their category names and confidence scores.

## Requirements

To run the Object Detection Application, you'll need Python 3.x and the following libraries:

1. **Streamlit**: For creating the web interface.
2. **OpenCV**: For image processing.
3. **NumPy**: For handling arrays and image data.
4. **MediaPipe**: For object detection.

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the `efficientdet_lite0.tflite` model file and place it in the same directory as the script. You can download the model from [MediaPipe Models](https://google.github.io/mediapipe/solutions/vision/object_detection.html).

4. Run the application:
    ```bash
    streamlit run object_detection.py
    ```

5. The application will open in your web browser. You can upload an image for object detection.

## How It Works

1. **Image Upload**: Users can upload images via the Streamlit interface.
2. **Object Detection**: The uploaded image is processed using MediaPipe's Object Detection model (`efficientdet_lite0.tflite`).
3. **Visualization**: The detected objects are highlighted with bounding boxes, labels, and confidence scores.

## Acknowledgments

This project uses MediaPipe's Object Detection model and Streamlit for creating the user interface. Special thanks to the authors of these libraries for their contributions.

