# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import re
import time

st.set_page_config(page_title="Smart Plate Scanner 🔮", layout="wide")

# -------------------------
# CSS / Styling (GenZ futuristic)
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #071129 50%, #001219 100%); color: #e6f0ff; }
    .title { font-size:34px; font-weight:700; color: #a7f3d0; letter-spacing:1px; }
    .subtitle { color: #9fb4ff; font-size:14px; margin-bottom:8px; }
    .panel { background: linear-gradient(145deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:16px; padding:18px; box-shadow: 0 8px 30px rgba(2,6,23,0.7); }
    .big-btn { background: linear-gradient(90deg,#7c3aed,#06b6d4); padding:10px 18px; border-radius:12px; color:white; font-weight:600; }
    .muted { color:#9fb4ff; opacity:0.8; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Smart Plate Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Scan vehicle number plate with camera or enter manually — futuristic UI ✨</div>', unsafe_allow_html=True)

# -------------------------
# Session state defaults
# -------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "camera"  # 'camera' or 'manual'

if "captured_number" not in st.session_state:
    st.session_state.captured_number = None

if "ocr_candidates" not in st.session_state:
    st.session_state.ocr_candidates = []

if "manual_input" not in st.session_state:
    st.session_state.manual_input = ""

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "status" not in st.session_state:
    st.session_state.status = ""

# -------------------------
# OCR helper (tries easyocr, falls back to pytesseract)
# -------------------------
@st.cache_resource
def load_easyocr_reader():
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)  # change gpu=True if GPU is available
        return reader
    except Exception:
        return None

def ocr_from_image(image: Image.Image):
    img = np.array(image.convert('RGB'))[:, :, ::-1]  # RGB->BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_proc = cv2.bilateralFilter(gray, 9, 75, 75)
    try:
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(gray_proc, window_size=25)
        binary = (gray_proc > thresh).astype(np.uint8) * 255
    except Exception:
        _, binary = cv2.threshold(gray_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = []
    reader = load_easyocr_reader()
    if reader is not None:
        try:
            raw = reader.readtext(img, detail=0)
            results.extend(raw)
        except Exception:
            pass

    if not results:
        try:
            import pytesseract
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789- '
            txt = pytesseract.image_to_string(gray_proc, config=config)
            for line in txt.splitlines():
                s = line.strip()
                if s:
                    results.append(s)
        except Exception:
            pass

    candidates = []
    for r in results:
        s = r.upper()
        s = re.sub(r'[^A-Z0-9\- ]', '', s).strip()
        if len(s) >= 4:
            candidates.append(s)

    dedup = []
    for c in candidates:
        if c not in dedup:
            dedup.append(c)
    return dedup

# -------------------------
# Layout: two columns
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Live Camera")
    st.markdown('<div class="muted">Camera stays active until you capture a plate or switch to Manual.</div>', unsafe_allow_html=True)

    top_cols = st.columns([1,1,1])
    with top_cols[0]:
        if st.button("Manual", key="manual_btn"):
            st.session_state.mode = "manual"
            st.session_state.status = "Switched to manual input"
    with top_cols[1]:
        if st.button("Camera Mode", key="camera_btn"):
            st.session_state.mode = "camera"
            st.session_state.status = "Camera active"
    with top_cols[2]:
        if st.button("Reset", key="reset_btn"):
            st.session_state.captured_number = None
            st.session_state.ocr_candidates = []
            st.session_state.manual_input = ""
            st.session_state.last_image = None
            st.session_state.status = "Reset"

    if st.session_state.mode == "camera":
        camera_image = st.camera_input("Point the camera at the vehicle plate and click 'Capture photo'", key="cam")
        if camera_image is not None:
            st.session_state.last_image = camera_image
            st.session_state.status = "Image captured — performing OCR..."
            try:
                img_pil = Image.open(camera_image)
                candidates = ocr_from_image(img_pil)
                st.session_state.ocr_candidates = candidates
                if candidates:
                    st.session_state.captured_number = candidates[0]
                    st.session_state.status = "OCR found candidates"
                else:
                    st.session_state.status = "No candidates found — try again or use Manual"
            except Exception as e:
                st.session_state.status = f"OCR error: {e}"

        if st.session_state.last_image is not None:
            st.markdown("---")
            st.image(st.session_state.last_image, caption="Captured frame", use_column_width=True)
            st.markdown("**OCR candidates:**")
            if st.session_state.ocr_candidates:
                for i, c in enumerate(st.session_state.ocr_candidates):
                    st.write(f"{i+1}. {c}")
            else:
                st.write("No text-like candidates detected.")

            st.markdown("**Validation**")
            val_cols = st.columns([1,1,1])
            with val_cols[0]:
                if st.button("Retry OCR", key="retry_ocr"):
                    st.session_state.last_image = None
                    st.session_state.ocr_candidates = []
                    st.session_state.captured_number = None
                    st.session_state.status = "Retry requested"
            with val_cols[1]:
                if st.session_state.captured_number and st.button("Number Captured", key="captured_confirm"):
                    st.session_state.status = "Captured — pending submit"
                    st.session_state.show_confirm = True
            with val_cols[2]:
                if st.session_state.captured_number and st.button("Submit as-is", key="submit_direct"):
                    st.session_state.status = f"Submitted: {st.session_state.captured_number}"
                    st.success(f"Submitted number: {st.session_state.captured_number}")
    else:
        st.markdown('<div style="padding:10px;border-radius:12px;background:rgba(255,255,255,0.02)">', unsafe_allow_html=True)
        st.markdown("### Manual Input Mode")
        st.markdown('<div class="muted">Type the number plate and click Submit.</div>', unsafe_allow_html=True)

        manual_txt = st.text_input("Plate Number", value=st.session_state.manual_input, key="manual_txt")
        st.session_state.manual_input = manual_txt

        if st.button("Submit Manual", key="manual_submit"):
            if st.session_state.manual_input:
                st.session_state.captured_number = st.session_state.manual_input
                st.success(f"Manual submitted: {st.session_state.captured_number}")
                st.session_state.status = "Manual submitted"
            else:
                st.warning("Please enter value before Submit.")

        st.write("")
        st.markdown("**Current input:**")
        st.code(st.session_state.manual_input or "<empty>", language="text")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Status & Preview")
    st.markdown(f"**Mode:** `{st.session_state.mode}`")
    st.markdown(f"**Status:** {st.session_state.status}")

    if st.session_state.get("show_confirm", False) and st.session_state.captured_number:
        st.markdown("#### Confirm detected number")
        confirmed = st.text_input("Edit / Confirm detected number", value=st.session_state.captured_number, key="confirm_edit")
        cols = st.columns([1,1,1])
        with cols[0]:
            if st.button("Submit final", key="final_submit"):
                st.session_state.captured_number = confirmed
                st.session_state.show_confirm = False
                st.session_state.status = f"Final Submitted: {st.session_state.captured_number}"
                st.success(f"Final Submitted: {st.session_state.captured_number}")
        with cols[1]:
            if st.button("Retry OCR (from confirm)", key="confirm_retry"):
                st.session_state.last_image = None
                st.session_state.ocr_candidates = []
                st.session_state.captured_number = None
                st.session_state.show_confirm = False
                st.session_state.status = "Retry from confirm"
        with cols[2]:
            if st.button("Cancel", key="confirm_cancel"):
                st.session_state.show_confirm = False
                st.session_state.status = "Cancelled confirmation"

    st.markdown("---")
    st.markdown("### Captured Number")
    if st.session_state.captured_number:
        st.success(st.session_state.captured_number)
    else:
        st.info("No plate captured yet.")

    st.markdown("---")
    st.markdown("### Tips")
    st.write("- Point camera square at plate, good lighting helps.")
    st.write("- Try different angles if OCR fails.")
    st.write("- If OCR returns junk, switch to Manual and type the plate.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Final output area
# -------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Output")

if st.session_state.captured_number:
    st.markdown(f"**Captured / Submitted Plate:** `{st.session_state.captured_number}`")

    if "fuel_submitted" not in st.session_state:
        st.session_state.fuel_submitted = False

    if not st.session_state.fuel_submitted:
        fuel_litres = st.number_input(
            "Enter fuel filled (litres)", min_value=0.0, step=0.5, key="fuel_input"
        )
        if st.button("Submit Fuel Entry"):
            if fuel_litres > 0:
                st.success(
                    f"Filled {fuel_litres} litres of fuel to {st.session_state.captured_number}"
                )
                st.session_state.fuel_submitted = True
            else:
                st.warning("Please enter a valid fuel amount.")
    else:
        st.info("Fuel entry already submitted.")
else:
    st.markdown("_No plate submitted yet._")
