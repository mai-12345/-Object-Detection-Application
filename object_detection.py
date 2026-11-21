
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.set_page_config(page_title="Object Detection", page_icon="🔍")

st.title("Object Detection Application")
st.write("Upload an image to detect objects.")


# Model path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "efficientdet_lite0.tflite"


# Visualization settings
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)


@st.cache_resource
def load_detector():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5
    )

    return vision.ObjectDetector.create_from_options(options)


def visualize(image, detection_result):

    annotated_image = image.copy()

    for detection in detection_result.detections:

        bbox = detection.bounding_box

        start_point = (
            bbox.origin_x,
            bbox.origin_y
        )

        end_point = (
            bbox.origin_x + bbox.width,
            bbox.origin_y + bbox.height
        )

        cv2.rectangle(
            annotated_image,
            start_point,
            end_point,
            TEXT_COLOR,
            3
        )

        if detection.categories:

            category = detection.categories[0]

            category_name = category.category_name or "Object"
            probability = round(category.score, 2)

            result_text = f"{category_name} ({probability})"

            text_location = (
                MARGIN + bbox.origin_x,
                MARGIN + ROW_SIZE + bbox.origin_y
            )

            cv2.putText(
                annotated_image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE,
                TEXT_COLOR,
                FONT_THICKNESS
            )

    return annotated_image


# Load detector
try:
    detector = load_detector()
    st.success("Object Detector loaded successfully!")

except Exception as e:
    st.error("Failed to load Object Detector.")
    st.exception(e)
    st.stop()


# Upload image
upload_image = st.file_uploader(
    "Please Upload an Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.subheader("Original Image")
        st.image(Image.fromarray(img))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )

        detection_result = detector.detect(mp_image)
        annotated_image = visualize(img.copy(), detection_result)

        st.subheader("Detection Results")
        st.image(Image.fromarray(annotated_image))

        st.write(f"Number of detected objects: {len(detection_result.detections)}")
    else:
        st.error("Could not read the uploaded image.")
