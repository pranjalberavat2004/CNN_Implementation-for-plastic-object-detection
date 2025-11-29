import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pickle
import os
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Config
# -------------------------
IMG_SIZE = 128
SAVE_DIR = "saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, "detections.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "confidence", "image_path"])

# -------------------------
# Load CNN + LabelEncoder
# -------------------------
try:
    with open("plastic_cnn.json", "r") as jf:
        model_json = jf.read()
    model = model_from_json(model_json)
    model.load_weights("plastic_cnn.h5")

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # notify user
    st.set_page_config(page_title="Plastic Detector", page_icon="♻", layout="wide")
    st.success("✅ CNN Model Loaded Successfully")
except Exception as e:
    st.set_page_config(page_title="Plastic Detector", page_icon="♻", layout="wide")
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -------------------------
# Plastic info (you already have this)
# -------------------------
plastic_info = {
    "disposable_plastic_cutlery": {"formula":"Polystyrene (PS)", "recyclable":"⚠ Rarely recyclable", "consequences":"Toxic styrene leakage."},
    "plastic_cup_lids": {"formula":"Polystyrene (PS)", "recyclable":"⚠ Hard to recycle", "consequences":"Microplastics pollution."},
    "plastic_detergent_bottles": {"formula":"HDPE", "recyclable":"♻ Recyclable", "consequences":"High energy process."},
    "plastic_food_containers": {"formula":"PP", "recyclable":"♻ Limited recycling", "consequences":"Toxic when heated."},
    "plastic_polyethene_bag": {"formula":"LDPE", "recyclable":"⚠ Limited facilities", "consequences":"Ocean pollution"},
    "plastic_shopping_bags": {"formula":"LDPE", "recyclable":"⚠ Partial", "consequences":"100+ year decomposition"},
    "plastic_soda_bottles": {"formula":"PET", "recyclable":"♻ Yes", "consequences":"Microplastic risk when reused/heated"},
    "plastic_straws": {"formula":"PP", "recyclable":"❌ Usually not", "consequences":"Dangerous for marine life."},
    "plastic_trash_bags": {"formula":"LDPE/HDPE", "recyclable":"⚠ Low", "consequences":"Blocks drains; pollutant."},
    "plastic_water_bottle": {"formula":"PET", "recyclable":"♻ Yes", "consequences":"Releases chemicals when heated."},
    
}

# -------------------------
# Helpers
# -------------------------
def classify_roi_bgr(bgr_img):
    """Takes an OpenCV BGR crop, returns (label, confidence) using your CNN."""
    # convert BGR->RGB PIL
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = np.array(pil).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    label = le.classes_[idx]
    confidence = float(preds[0][idx] * 100.0)
    return label, confidence

def save_detection_to_disk(img_bgr, label, confidence):
    """Save BGR image and append CSV row with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{label}_{int(confidence)}_{ts}.jpg"
    path = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(path, img_bgr)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, label, f"{confidence:.2f}", path])
    return path

# initialize session state keys
if "last_label" not in st.session_state:
    st.session_state["last_label"] = None
if "last_conf" not in st.session_state:
    st.session_state["last_conf"] = None
if "last_frame" not in st.session_state:
    st.session_state["last_frame"] = None
if "camera_running" not in st.session_state:
    st.session_state["camera_running"] = False

# -------------------------
# UI layout
# -------------------------
st.title("♻ Real-Time Plastic Detection Using Custom CNN")
tabs = st.tabs(["🎥 Real-Time Detection", "📸 Capture", "📤 Upload & Classify", "🗂 Saved History"])

# -------------------------
# Tab: REAL-TIME Detection
# -------------------------
with tabs[0]:
    st.subheader("Real-Time Object Detection with Bounding Box")
    col1, col2 = st.columns([3,1])

    with col2:
        start_btn = st.button("Start Camera")
        stop_btn = st.button("Stop Camera")
        if st.button("Save Last Detection"):
            if st.session_state["last_frame"] is not None and st.session_state["last_label"] is not None:
                saved_path = save_detection_to_disk(st.session_state["last_frame"], st.session_state["last_label"], st.session_state["last_conf"])
                st.success(f"Saved last detection to {saved_path}")
            else:
                st.warning("No detection to save yet. Run camera and wait for a detection.")

    frame_slot = col1.empty()

    # start / stop camera control
    if start_btn:
        st.session_state["camera_running"] = True
    if stop_btn:
        st.session_state["camera_running"] = False

    if st.session_state["camera_running"]:
        cap = cv2.VideoCapture(0)
        # small guard
        if not cap.isOpened():
            st.error("Cannot open webcam. Make sure your camera is connected.")
            st.session_state["camera_running"] = False
        else:
            try:
                # run until user clicks Stop
                while st.session_state["camera_running"]:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame from webcam.")
                        break

                    # PREPROCESS -> edges -> largest contour -> bbox
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (7,7), 0)
                    edges = cv2.Canny(blur, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    predicted_label = None
                    confidence = 0.0

                    if contours:
                        # pick largest contour by area and ignore tiny areas
                        c = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(c)
                        h_frame, w_frame = frame.shape[:2]
                        # optional area threshold to avoid noisy detections
                        if area > (w_frame * h_frame) * 0.001:  # adjust threshold if needed
                            x,y,w,h = cv2.boundingRect(c)
                            # pad bbox slightly
                            pad = int(0.03 * max(w,h))
                            x1 = max(0, x-pad)
                            y1 = max(0, y-pad)
                            x2 = min(w_frame, x+w+pad)
                            y2 = min(h_frame, y+h+pad)
                            roi = frame[y1:y2, x1:x2]

                            if roi.size != 0:
                                # classify ROI with your CNN
                                predicted_label, confidence = classify_roi_bgr(roi)

                                # overlay bbox + label
                                cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
                                label_text = f"{predicted_label} ({confidence:.1f}%)"
                                cv2.putText(frame, label_text, (x1, max(15,y1-8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                                # store last detection in session state (BGR image)
                                st.session_state["last_frame"] = frame.copy()
                                st.session_state["last_label"] = predicted_label
                                st.session_state["last_conf"] = confidence

                    # display frame (convert BGR->RGB for correct colors in Streamlit)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(frame_rgb, use_column_width=True)

                    # small wait so UI updates (and to avoid 100% CPU loop)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # release camera when loop ends
            except Exception as ex:
                st.error(f"Camera loop error: {ex}")
            finally:
                cap.release()
                st.session_state["camera_running"] = False

    else:
        # show placeholder image/info when camera not running
        if st.session_state["last_frame"] is not None:
            # show last frame preview if exists
            frame_rgb = cv2.cvtColor(st.session_state["last_frame"], cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, use_column_width=True)
        else:
            frame_slot.info("Camera is stopped. Click Start Camera to begin real-time detection.")

    # Display result card for last prediction (if available)
    if st.session_state["last_label"]:
        lbl = st.session_state["last_label"]
        conf = st.session_state["last_conf"]
        info = plastic_info.get(lbl, {})
        st.markdown(
            f"""
            <div style="
                background-color:#003300;
                padding:14px;
                border-radius:10px;
                color:white;">
                🟢 <b>Prediction:</b> {lbl} ({conf:.2f}%)
                <div style='margin-top:8px; color:#DCEFE6;'>
                <b>Formula:</b> {info.get('formula','N/A')} &nbsp;&nbsp;
                <b>Recyclable:</b> {info.get('recyclable','N/A')}
                <p style='margin-top:6px;'>🌍 {info.get('consequences','N/A')}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------
# Tab: Capture
# -------------------------
with tabs[1]:
    st.subheader("Capture image from webcam and classify")
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        img = Image.open(camera_file).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]  # RGB->BGR for our classify function
        # classify full image (no bbox)
        label, conf = classify_roi_bgr(img_np)
        st.image(img, use_column_width=False, width=400)
        st.success(f"Prediction: {label} ({conf:.2f}%)")
        info = plastic_info.get(label, {})
        st.markdown(f"**Formula:** {info.get('formula','N/A')}  \n**Recyclable:** {info.get('recyclable','N/A')}  \n🌍 {info.get('consequences','N/A')}")
        if st.button("Save this capture"):
            saved = save_detection_to_disk(img_np, label, conf)
            st.success(f"Saved to {saved}")

# -------------------------
# Tab: Upload & Classify
# -------------------------
with tabs[2]:
    st.subheader("Upload an image to classify")
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"])
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        st.image(pil, width=400)
        arr_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        # try to find largest contour in uploaded image to get bbox first
        gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 100:  # simple threshold
                roi = arr_bgr[y:y+h, x:x+w]
                label, conf = classify_roi_bgr(roi)
                st.success(f"Detected (from bbox): {label} ({conf:.2f}%)")
                if st.button("Save this detection from uploaded image"):
                    saved = save_detection_to_disk(arr_bgr, label, conf)
                    st.success(f"Saved to {saved}")
            else:
                # small contour -> classify whole image
                label, conf = classify_roi_bgr(arr_bgr)
                st.success(f"Whole-image classification: {label} ({conf:.2f}%)")
                if st.button("Save whole-image classification"):
                    saved = save_detection_to_disk(arr_bgr, label, conf)
                    st.success(f"Saved to {saved}")
        else:
            # no contours found -> whole image classification
            label, conf = classify_roi_bgr(arr_bgr)
            st.success(f"Whole-image classification: {label} ({conf:.2f}%)")
            if st.button("Save whole-image classification"):
                saved = save_detection_to_disk(arr_bgr, label, conf)
                st.success(f"Saved to {saved}")

# -------------------------
# Tab: Saved History
# -------------------------
with tabs[3]:
    st.subheader("Saved detections")
    if os.path.exists(CSV_PATH):
        df = None
        try:
            import pandas as pd
            df = pd.read_csv(CSV_PATH)
        except Exception:
            st.write("Could not read CSV results file.")
        if df is None or df.empty:
            st.write("No saved detections yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False).reset_index(drop=True))
            if st.button("Download all saved images as ZIP"):
                # create zip in-memory
                import zipfile, io
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as zf:
                    for p in df["image_path"].dropna().unique():
                        if os.path.exists(p):
                            zf.write(p, arcname=os.path.basename(p))
                buffer.seek(0)
                st.download_button("Download ZIP", data=buffer, file_name="saved_detections.zip")
    else:
        st.write("No saved results file found yet.")
