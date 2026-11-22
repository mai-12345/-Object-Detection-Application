import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide"
)

st.title("Object Detection Application")


# -----------------------------
# Model Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "efficientdet_lite0.tflite")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load Object Detector
# -----------------------------
@st.cache_resource
def load_detector():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5,
        max_results=10
    )

    detector = vision.ObjectDetector.create_from_options(
        options
    )

    return detector


try:
    detector = load_detector()
    st.success("Object Detector loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Please Upload an Image",
    type=["jpg", "jpeg", "png", "webp"]
)


# -----------------------------
# Draw Detection Results
# -----------------------------
def visualize(image, detection_result):

    output_image = image.copy()

    for detection in detection_result.detections:

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        width = bbox.width
        height = bbox.height

        # Draw bounding box
        cv2.rectangle(
            output_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Get category information
        category = detection.categories[0]

        label = (
            category.category_name
            if category.category_name
            else category.display_name
        )

        score = category.score

        text = f"{label}: {score:.2f}"

        # Draw label
        cv2.putText(
            output_image,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return output_image


# -----------------------------
# Process Uploaded Image
# -----------------------------
if uploaded_file is not None:

    try:
        # Read uploaded image
        image_bytes = uploaded_file.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image_bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image_bgr is None:
            st.error("Could not read the uploaded image.")
            st.stop()

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        image_rgb = np.ascontiguousarray(image_rgb)

        # Convert to PIL Image
        original_image = Image.fromarray(image_rgb)

        # Display original image
        st.subheader("Original Image")

        st.image(
            original_image,
            caption="Original Image",
            use_container_width=True
        )

        # -----------------------------
        # Run Object Detection
        # -----------------------------
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        detection_result = detector.detect(mp_image)

        # Visualize results
        annotated_image = visualize(
            image_rgb,
            detection_result
        )

        # Convert result to PIL Image
        result_image = Image.fromarray(annotated_image)

        # Display detection results
        st.subheader("Detection Results")

        st.image(
            result_image,
            caption="Detected Objects",
            use_container_width=True
        )

        # -----------------------------
        # Detection Count
        # -----------------------------
        number_of_objects = len(
            detection_result.detections
        )

        st.write(
            f"Number of detected objects: {number_of_objects}"
        )

        st.success("Object detection completed successfully!")

        # -----------------------------
        # Save Output Image
        # -----------------------------
        output_filename = "detected_image.png"

        output_path = os.path.join(
            OUTPUT_DIR,
            output_filename
        )

        result_image.save(
            output_path,
            format="PNG"
        )

        # Prepare image for download
        from io import BytesIO

        image_buffer = BytesIO()

        result_image.save(
            image_buffer,
            format="PNG"
        )

        image_bytes_to_download = image_buffer.getvalue()

        st.download_button(
            label="Download Detection Result",
            data=image_bytes_to_download,
            file_name=output_filename,
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")
