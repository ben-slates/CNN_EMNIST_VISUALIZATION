import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# ================= LOAD MODELS =================
digit_model = tf.keras.models.load_model("digit_model.h5")
letter_model = tf.keras.models.load_model("letter_model.h5")

st.set_page_config(page_title="Handwriting Recognition", layout="centered")
st.title(" Handwriting Recognition (Digit & Alphabet)")

# ================= HELPER FUNCTIONS =================
def center_image(img):
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = img[y0:y1+1, x0:x1+1]
    return cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

def has_loop(img_28):
    """
    Detects closed loops (used to distinguish 9 from 7).
    """
    img_bin = (img_28 > 0.2).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return False

    # Loop exists if any contour has a child contour
    for h in hierarchy[0]:
        if h[2] != -1:
            return True
    return False

# ================= CANVAS =================
canvas = st_canvas(
    stroke_width=15,
    stroke_color="yellow",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ================= PREDICTION =================
if st.button(" Predict"):

    if canvas.image_data is None:
        st.warning("Please draw something first.")
        st.stop()

    img = canvas.image_data[:, :, 0]

    if np.sum(img) < 10:
        st.warning("Canvas is empty. Draw a digit or letter.")
        st.stop()

    # Preprocessing (MATCHES TRAINING)
    img = center_image(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # ================= MODEL PREDICTIONS =================
    digit_probs = digit_model.predict(img, verbose=0)[0]
    letter_probs = letter_model.predict(img, verbose=0)[0]

    digit_conf = float(np.max(digit_probs))
    digit_pred = int(np.argmax(digit_probs))

    letter_conf = float(np.max(letter_probs))
    letter_idx = int(np.argmax(letter_probs))
    letter_pred = chr(letter_idx + 65)

    # ================= DECISION LOGIC =================
    img_28 = img.reshape(28, 28)

    # --- RULE 1: VERY CONFIDENT DIGIT ---
    if digit_conf >= 0.85:
        if digit_pred == 7 and has_loop(img_28):
            final_type = "DIGIT"
            final_pred = 9
            final_conf = digit_conf
        else:
            final_type = "DIGIT"
            final_pred = digit_pred
            final_conf = digit_conf

    # --- RULE 2: LETTER LOOKS LIKE DIGIT ---
    elif letter_pred in ["P", "S", "B", "O"] and digit_conf >= 0.65:
        if digit_pred == 7 and has_loop(img_28):
            final_type = "DIGIT"
            final_pred = 9
            final_conf = digit_conf
        else:
            final_type = "DIGIT"
            final_pred = digit_pred
            final_conf = digit_conf

    # --- RULE 3: CONFIDENCE COMPARISON ---
    else:
        if digit_conf > letter_conf:
            if digit_pred == 7 and has_loop(img_28):
                final_type = "DIGIT"
                final_pred = 9
                final_conf = digit_conf
            else:
                final_type = "DIGIT"
                final_pred = digit_pred
                final_conf = digit_conf
        else:
            final_type = "ALPHABET"
            final_pred = letter_pred
            final_conf = letter_conf

    # ================= OUTPUT =================
    if final_type == "DIGIT":
        st.success(" Type: DIGIT")
        st.write(f"**Prediction:** {final_pred}")
    else:
        st.success(" Type: ALPHABET")
        st.write(f"**Prediction:** {final_pred}")

    st.write(f"**Confidence:** {final_conf:.2f}")

    # Debug info (useful for viva)
    st.caption(
        f"Digit → {digit_pred} ({digit_conf:.2f}) | "
        f"Letter → {letter_pred} ({letter_conf:.2f})"
    )
