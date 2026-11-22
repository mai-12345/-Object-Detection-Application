import os
import urllib.request

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.title("Object Detection Application")

MODEL_PATH = "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red, since the image is RGB


def visualize(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width),
                     int(bbox.origin_y + bbox.height))
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        result_text = f"{category.category_name} ({round(category.score, 2)})"
        text_location = (int(MARGIN + bbox.origin_x),
                         int(MARGIN + ROW_SIZE + bbox.origin_y))
        cv2.putText(image, result_text, text_location,
                    cv2.FONT_HERSHEY_PLAIN, FONT_SIZE,
                    TEXT_COLOR, FONT_THICKNESS)
    return image


@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, score_threshold=0.5
    )
    return vision.ObjectDetector.create_from_options(options)


detector = load_detector()

upload_image = st.file_uploader(
    "Please upload an image...", type=["jpg", "png", "jpeg", "webp"]
)

if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the image. Please try another one.")
    else:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("Original Image")
        st.image(img, use_container_width=True)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(img),
        )

        detection_result = detector.detect(mp_image)

        annotated_image = visualize(img.copy(), detection_result)

        st.subheader("Image After Detection")
        st.image(annotated_image, use_container_width=True)

        st.write(f"Number of detected objects: {len(detection_result.detections)}")
