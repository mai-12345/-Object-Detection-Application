import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.title("Object Detection Application")
st.write("Upload an image to detect objects.")

upload_image = st.file_uploader(
    "Please Upload an Image",
    type=['jpg', 'png', 'jpeg', 'webp']
)

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)


def visualize(image, detection_result) -> np.ndarray:

    for detection in detection_result.detections:

        # Draw bounding box
        bbox = detection.bounding_box

        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (
            bbox.origin_x + bbox.width,
            bbox.origin_y + bbox.height
        )

        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)

        result_text = f'{category_name} ({probability})'

        text_location = (
            MARGIN + bbox.origin_x,
            MARGIN + ROW_SIZE + bbox.origin_y
        )

        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS
        )

    return image


# STEP 2: Create an ObjectDetector object

MODEL_PATH = 'efficientdet_lite0.tflite'

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5
)

detector = vision.ObjectDetector.create_from_options(options)

st.success("Object Detector loaded successfully!")


# STEP 3: Upload and read image

if upload_image is not None:

    file_bytes = np.frombuffer(
        upload_image.getvalue(),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display Original Image
    st.subheader("Original Image")

    st.image(
        Image.fromarray(img),
        use_container_width=True
    )


    # STEP 4: Detect objects

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img
    )

    detection_result = detector.detect(mp_image)


    # STEP 5: Visualize results

    annotated_image = visualize(
        img.copy(),
        detection_result
    )

    st.subheader("Detection Results")

    st.image(
        Image.fromarray(annotated_image),
        use_container_width=True
    )

    st.write(
        f"Number of detected objects: {len(detection_result.detections)}"
    )

    if detection_result.detections:
        st.success("Object detection completed successfully!")
    else:
        st.warning("No objects detected. Try another image.")
