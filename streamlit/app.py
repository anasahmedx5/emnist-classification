import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Tracing Evaluator", layout="wide")

st.title("Tracing Evaluator & CV Engine")
st.write("Draw the selected character inside the canvas to receive real-time intent validation and geometric feedback.")

@st.cache_resource
def load_keras_model():
    model = tf.keras.models.load_model("best_tracing_model.keras")
    return model

try:
    model = load_keras_model()
    st.sidebar.success("Loaded 'best_tracing_model.keras'")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.sidebar.info("Ensure 'best_tracing_model.keras' is in the root directory.")

EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def generate_ideal_template(target_char, canvas_size=280):
    """Generates a perfect binary reference template for IoU alignment."""
    img = Image.new('L', (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", int(canvas_size * 0.7))
    except IOError:
        font = ImageFont.load_default()
        
    bbox = draw.textbbox((0, 0), target_char, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = ((canvas_size - text_w) / 2, (canvas_size - text_h) / 2 - bbox[1])
    
    draw.text(position, target_char, fill=255, font=font)
    return np.array(img)

def compute_iou_and_errors(user_binary, template_binary):
    """Calculates Intersection over Union (IoU) and extracts missing stroke locations."""
    u_mask = (user_binary > 0).astype(np.uint8)
    t_mask = (template_binary > 0).astype(np.uint8)

    intersection = np.logical_and(u_mask, t_mask)
    union = np.logical_or(u_mask, t_mask)
    
    union_sum = np.sum(union)
    iou_score = (np.sum(intersection) / union_sum) * 100 if union_sum > 0 else 0.0

    missing_pixels = cv2.subtract(t_mask * 255, u_mask * 255)
    
    overlay_rgb = cv2.cvtColor(user_binary, cv2.COLOR_GRAY2RGB)
    overlay_rgb[missing_pixels > 0] = [255, 0, 0] 
    
    return round(iou_score, 1), overlay_rgb

st.sidebar.header("Tracing Setup")
target_char = st.sidebar.selectbox("Select Target Character to Trace:", EMNIST_LABELS, index=10) 
stroke_width = st.sidebar.slider("Brush Stroke Width:", 10, 30, 20)

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.subheader(f"1. Draw Target: '{target_char}'")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

template_img = generate_ideal_template(target_char, canvas_size=280)

with col2:
    st.subheader("2. Target Template")
    st.image(template_img, caption=f"Ideal '{target_char}' Path", width=280)


if canvas_result.image_data is not None:
    raw_rgba = canvas_result.image_data.astype(np.uint8)
    gray_canvas = cv2.cvtColor(raw_rgba, cv2.COLOR_RGBA2GRAY)
    _, user_binary = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)
    
    if np.sum(user_binary) > 0:
        
        resized_28 = cv2.resize(user_binary, (28, 28), interpolation=cv2.INTER_AREA)
        normalized_tensor = np.expand_dims(np.expand_dims(resized_28 / 255.0, axis=-1), axis=0)
        
        preds = model.predict(normalized_tensor, verbose=0)
        predicted_class_idx = np.argmax(preds)
        predicted_char = EMNIST_LABELS[predicted_class_idx]
        cnn_confidence = float(preds[0][predicted_class_idx]) * 100
        
        iou_score, error_overlay = compute_iou_and_errors(user_binary, template_img)
        
        with col3:
            st.subheader("3. Error Analysis")
            st.image(error_overlay, caption="Red = Missing Strokes", width=280)
            
        st.markdown("---")
        st.header("Evaluation Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        intent_matched = (predicted_char == target_char)
        
        with res_col1:
            st.metric("Detected Character", f"'{predicted_char}'", delta="Match!" if intent_matched else "Mismatch", delta_color="normal" if intent_matched else "inverse")
            
        with res_col2:
            st.metric("CNN Intent Confidence", f"{cnn_confidence:.1f}%")
            
        with res_col3:
            st.metric("Tracing Score (IoU)", f"{iou_score}%")
            
        if not intent_matched:
            st.error(f"**Feedback:** That doesn't look quite like an '{target_char}'. It looks more like a '{predicted_char}'. Try again!")
        elif iou_score > 70:
            st.success(f"**Feedback:** Excellent job! Your tracing of '{target_char}' is accurate and smooth.")
        elif iou_score > 40:
            st.warning(f"**Feedback:** Good effort! You traced '{target_char}' correctly, but check the red highlighted spots to make your lines straighter.")
        else:
            st.info(f"**Feedback:** You drew an '{target_char}', but you missed a lot of the template path. Try covering the full shape.")