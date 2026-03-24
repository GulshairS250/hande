import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.models import load_model

try:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # type: ignore
    webrtc_streamer = None  # type: ignore


APP_TITLE = "Hand Gesture Recognition"
DEFAULT_MODEL_NAME = "best_hand_gesture_model.keras"
DEFAULT_IMG_SIZE = 128
FALLBACK_LABELS = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]


def _find_model_path() -> Optional[Path]:
    cwd = Path(".")
    exact = cwd / DEFAULT_MODEL_NAME
    if exact.exists():
        return exact

    keras_files = sorted(cwd.glob("*.keras"))
    if keras_files:
        return keras_files[0]
    return None


def _scan_labels_from_dataset(dataset_root: Path) -> List[str]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []

    labels = set()
    for user_dir in dataset_root.iterdir():
        if not user_dir.is_dir():
            continue
        for gesture_dir in user_dir.iterdir():
            if gesture_dir.is_dir():
                labels.add(gesture_dir.name)

    def sort_key(name: str) -> Tuple[int, str]:
        try:
            prefix = int(name.split("_", 1)[0])
        except Exception:
            prefix = 9999
        return (prefix, name)

    return sorted(labels, key=sort_key)


def _load_labels() -> List[str]:
    labels_json = Path("labels.json")
    if labels_json.exists():
        try:
            with labels_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "class_names" in data:
                data = data["class_names"]
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass

    dataset_root = Path(os.environ.get("DATASET_ROOT", "leapGestRecog"))
    scanned = _scan_labels_from_dataset(dataset_root)
    if scanned:
        return scanned

    return FALLBACK_LABELS


@st.cache_resource
def get_model() -> tf.keras.Model:
    model_path = _find_model_path()
    if model_path is None:
        raise FileNotFoundError(
            f"Model not found. Put '{DEFAULT_MODEL_NAME}' (or any .keras file) in this folder."
        )
    return load_model(model_path)


@st.cache_data
def get_labels() -> List[str]:
    return _load_labels()


def preprocess_image(
    img_rgb: np.ndarray,
    img_size: int = DEFAULT_IMG_SIZE,
    use_vgg_preprocess: bool = False,
) -> np.ndarray:
    resized = cv2.resize(img_rgb, (img_size, img_size))
    arr = resized.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    if use_vgg_preprocess:
        arr = vgg16_preprocess_input(arr)
    else:
        arr = arr / 255.0
    return arr


def get_img_size() -> int:
    return int(st.session_state.get("img_size", DEFAULT_IMG_SIZE))


def predict_image(
    img_rgb: np.ndarray,
    model: tf.keras.Model,
    class_names: List[str],
    img_size: Optional[int] = None,
    use_vgg_preprocess: bool = False,
) -> Tuple[str, float, np.ndarray]:
    resolved_size = img_size if img_size is not None else get_img_size()
    batch = preprocess_image(img_rgb, img_size=resolved_size, use_vgg_preprocess=use_vgg_preprocess)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    conf = float(probs[idx])
    return label, conf, probs


def render_prediction_result(label: str, conf: float) -> None:
    st.success(f"Prediction: {label}")
    st.write(f"Confidence: **{conf * 100:.2f}%**")


def handle_single_image(model: tf.keras.Model, class_names: List[str], use_vgg_preprocess: bool) -> None:
    uploaded = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "bmp"], key="single_uploader")
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)
    st.image(img_rgb, caption="Uploaded image", use_container_width=True)
    label, conf, _ = predict_image(img_rgb, model, class_names, use_vgg_preprocess=use_vgg_preprocess)
    render_prediction_result(label, conf)


def handle_multiple_images(model: tf.keras.Model, class_names: List[str], use_vgg_preprocess: bool) -> None:
    files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="multi_uploader",
    )
    if not files:
        return

    st.subheader("Results")
    cols = st.columns(3)
    for i, uploaded in enumerate(files):
        image = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(image)
        label, conf, _ = predict_image(img_rgb, model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        with cols[i % 3]:
            st.image(img_rgb, caption=uploaded.name, use_container_width=True)
            st.write(f"**{label}**")
            st.caption(f"{conf * 100:.2f}%")


def handle_video(model: tf.keras.Model, class_names: List[str], use_vgg_preprocess: bool) -> None:
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"], key="video_uploader")
    if uploaded is None:
        return

    sample_every_n = st.slider("Sample every N frames", min_value=5, max_value=60, value=15, step=1)
    max_samples = st.slider("Max sampled frames", min_value=10, max_value=300, value=80, step=10)

    temp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    st.video(temp_path)
    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        st.error("Could not read video file.")
        return

    frame_id = 0
    sampled = 0
    rows = []
    labels_counter = Counter()

    with st.spinner("Analyzing video..."):
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_id % sample_every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                label, conf, _ = predict_image(
                    frame_rgb,
                    model,
                    class_names,
                    use_vgg_preprocess=use_vgg_preprocess,
                )
                rows.append({"frame": frame_id, "label": label, "confidence": round(conf * 100, 2)})
                labels_counter[label] += 1
                sampled += 1
                if sampled >= max_samples:
                    break
            frame_id += 1
        cap.release()

    if not rows:
        st.warning("No sampled frames were analyzed.")
        return

    majority_label, count = labels_counter.most_common(1)[0]
    st.success(f"Overall video prediction: {majority_label} ({count}/{sampled} sampled frames)")
    st.dataframe(rows, use_container_width=True)


class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.model = get_model()
        self.class_names = get_labels()
        self.use_vgg_preprocess = bool(st.session_state.get("use_vgg_preprocess", False))
        self.last_label = ""
        self.last_conf = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label, conf, _ = predict_image(
            rgb,
            self.model,
            self.class_names,
            use_vgg_preprocess=self.use_vgg_preprocess,
        )
        self.last_label = label
        self.last_conf = conf
        text = f"{label} ({conf * 100:.1f}%)"
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame.from_ndarray(img, format="bgr24")


def handle_live_webcam() -> None:
    if not WEBRTC_AVAILABLE:
        st.warning(
            "Live webcam needs `streamlit-webrtc`. Install dependencies from requirements.txt and restart."
        )
        st.info("You can still use snapshot mode below.")

    st.subheader("Live Webcam")
    if WEBRTC_AVAILABLE:
        ctx = webrtc_streamer(
            key="gesture-live",
            video_processor_factory=GestureVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx and ctx.video_processor:
            st.write(
                f"Live: **{ctx.video_processor.last_label}** ({ctx.video_processor.last_conf * 100:.2f}%)"
            )

    st.subheader("Snapshot")
    snapshot = st.camera_input("Take a single photo")
    if snapshot is not None:
        image = Image.open(snapshot).convert("RGB")
        img_rgb = np.array(image)
        st.image(img_rgb, caption="Captured snapshot", use_container_width=True)
        model = get_model()
        labels = get_labels()
        use_vgg = bool(st.session_state.get("use_vgg_preprocess", False))
        label, conf, _ = predict_image(img_rgb, model, labels, use_vgg_preprocess=use_vgg)
        render_prediction_result(label, conf)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Predict hand gestures from image, multiple images, video, and live webcam.")

    with st.sidebar:
        st.header("Settings")
        st.write("Run with:")
        st.code("pip install -r requirements.txt\nstreamlit run app.py")

        st.session_state["use_vgg_preprocess"] = st.checkbox(
            "Use VGG16 preprocess_input",
            value=False,
            help="Enable only if your trained model expects VGG16 preprocessing.",
        )

        st.session_state["img_size"] = st.number_input(
            "Image size",
            min_value=64,
            max_value=512,
            value=DEFAULT_IMG_SIZE,
            step=16,
        )

        started = st.session_state.get("started", False)
        if st.button("Start", type="primary"):
            st.session_state["started"] = True
            started = True
        if st.button("Stop"):
            st.session_state["started"] = False
            started = False
        st.write(f"Status: {'Started' if started else 'Stopped'}")

    try:
        model = get_model()
        class_names = get_labels()
    except Exception as e:
        st.error(str(e))
        st.stop()

    mode = st.selectbox("Choose mode", ["Single image", "Multiple images", "Video file", "Live webcam"])

    if not st.session_state.get("started", False):
        st.info("Press **Start** in the sidebar to enable camera and prediction.")
        return

    use_vgg = bool(st.session_state.get("use_vgg_preprocess", False))

    if mode == "Single image":
        handle_single_image(model, class_names, use_vgg)
    elif mode == "Multiple images":
        handle_multiple_images(model, class_names, use_vgg)
    elif mode == "Video file":
        handle_video(model, class_names, use_vgg)
    else:
        handle_live_webcam()


if __name__ == "__main__":
    main()
