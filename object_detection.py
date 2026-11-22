import os
import urllib.request

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.title("Object Detection Application")

MODEL_PATH = "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # أحمر (لأن الصورة RGB)


def visualize(image, detection_result):
    """Draws bounding boxes and labels on the image and returns it."""
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        result_text = f"{category.category_name} ({round(category.score, 2)})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location,
                    cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image


@st.cache_resource
def load_detector():
    # لو ملف الموديل مش موجود، نزّله تلقائياً
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.5)
    return vision.ObjectDetector.create_from_options(options)


detector = load_detector()

upload_image = st.file_uploader("Please Upload an Image....",
                                type=["jpg", "png", "jpeg", "webp"])

if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("مش قادر أقرأ الصورة، جرب صورة تانية.")
        st.stop()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)

    st.image(Image.fromarray(img), caption="Uploaded Image",
             use_container_width=True)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(mp_image)

    annotated_image = visualize(img.copy(), detection_result)

    st.image(Image.fromarray(annotated_image), caption="Detection Result",
             use_container_width=True)

    st.write(f"عدد الأشياء المكتشفة: {len(detection_result.detections)}")
