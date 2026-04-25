import streamlit as st
import cv2
import numpy as np
import time
import os
import tempfile
import json
import pandas as pd
import re

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KnightSight EdgeVision | ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #060912; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2a4a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px; padding: 16px; text-align: center; margin: 4px 0;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #00d4ff; }
    .metric-label { font-size: 0.72rem; color: #667788; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2px; }
    .good { color: #00ff88 !important; }
    .warn { color: #ffcc00 !important; }
    .badge {
        display: inline-block; background: linear-gradient(90deg, #0f3460, #533483);
        color: white; padding: 3px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; margin: 2px;
    }
    .detection-card {
        background: #0d1b2a; border-left: 3px solid #00d4ff;
        border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.85rem;
    }
    .plate-card {
        background: #0d1b2a; border-left: 3px solid #00ff88;
        border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.85rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #533483) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        padding: 10px 20px !important; font-weight: 700 !important; width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #533483, #e94560) !important; }
    .stTabs [data-baseweb="tab"] { color: #667788 !important; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:16px 0 8px 0;">
    <h1 style="color:#00d4ff; font-size:2.2rem; font-weight:800; margin:0; letter-spacing:-1px;">
        🚗 KnightSight EdgeVision
    </h1>
    <p style="color:#445566; margin:4px 0 0 0; font-size:0.9rem;">
        ANPR Vehicle Intelligence Pipeline &nbsp;|&nbsp;
        <span class="badge">YOLO11n</span>
        <span class="badge">BoT-SORT</span>
        <span class="badge">PaddleOCR</span>
        <span class="badge">Temporal Fusion</span>
        <span class="badge">100% Offline</span>
    </p>
</div>
<hr style="border-color:#1a2a4a; margin:10px 0 16px 0;">
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES  = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VEHICLE_COLORS = {2: (0, 200, 255), 3: (255, 130, 0), 5: (0, 255, 160), 7: (180, 0, 255)}
PLATE_COLOR  = (0, 255, 100)

PLATE_MODEL_PATH = "runs/detect/runs/detect/plate_detector_yolo11/weights/best.pt"

# ─── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    v_model = None
    for path in ["yolo11n.engine", "yolo11n.onnx", "yolo11n.pt"]:
        if os.path.exists(path):
            try:
                v_model = YOLO(path)
                break
            except Exception:
                continue
    if v_model is None:
        v_model = YOLO("yolo11n.pt")

    p_model = None
    base = os.path.splitext(PLATE_MODEL_PATH)[0]
    for path in [f"{base}.engine", f"{base}.onnx", PLATE_MODEL_PATH]:
        if os.path.exists(path):
            try:
                p_model = YOLO(path)
                break
            except Exception:
                continue

    return v_model, p_model, device

# ─── OCR Engine ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_ocr():
    try:
        from paddleocr import PaddleOCR
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        return PaddleOCR(use_textline_orientation=True, lang='en')
    except Exception:
        return None

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    mean_val = np.mean(enhanced)
    if mean_val > 5:
        gamma = float(np.clip(np.log(127.5) / np.log(mean_val + 1e-3), 0.5, 2.0))
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
    return enhanced

def validate_plate(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if 6 <= len(cleaned) <= 10:
        return cleaned
    return None

# ─── Core Detection Function ──────────────────────────────────────────────────
def detect_frame(img_bgr, v_model, p_model, ocr, conf_thresh, iou_thresh, use_preprocess=True):
    if use_preprocess:
        processed_img = preprocess(img_bgr)
    else:
        processed_img = img_bgr
        
    annotated = img_bgr.copy()

    t0 = time.perf_counter()
    v_results = v_model(processed_img, conf=conf_thresh, iou=iou_thresh,
                        classes=VEHICLE_CLASSES, verbose=False)[0]
    v_time_ms = (time.perf_counter() - t0) * 1000

    detections = []

    for box in v_results.boxes:
        cls_id   = int(box.cls[0])
        conf_v   = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color    = VEHICLE_COLORS.get(cls_id, (200, 200, 200))
        label    = CLASS_NAMES.get(cls_id, "Vehicle")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf_v:.0%}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, txt, (x1+3, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

        plate_text = None
        plate_conf = None
        p_time_ms  = 0.0

        if p_model:
            crop = processed_img[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0:
                tp = time.perf_counter()
                p_res = p_model(crop, conf=0.10, verbose=False)[0]
                p_time_ms = (time.perf_counter() - tp) * 1000

                for pb in p_res.boxes:
                    px1,py1,px2,py2 = map(int, pb.xyxy[0])
                    
                    pad = 2
                    py1_p = max(0, py1 - pad)
                    py2_p = min(crop.shape[0], py2 + pad)
                    px1_p = max(0, px1 - pad)
                    px2_p = min(crop.shape[1], px2 + pad)
                    plate_crop = crop[py1_p:py2_p, px1_p:px2_p]

                    if plate_crop.size == 0: continue

                    cv2.rectangle(annotated, (x1+px1, y1+py1), (x1+px2, y1+py2), PLATE_COLOR, 2)

                    if ocr is not None:
                        try:
                            # --- 🚀 THE UPSCALING & BORDER HACK ---
                            # Double the image size
                            plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            # Add a solid white 20px border so the OCR engine has breathing room
                            plate_crop = cv2.copyMakeBorder(plate_crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                            # --------------------------------------

                            # Use new predict logic
                            r_gen = ocr.predict(plate_crop)
                            r = list(r_gen) if r_gen else []
                            
                            found_texts = []
                            found_scores = []
                            
                            if r:
                                res_obj = r[0]
                                if isinstance(res_obj, dict) and 'rec_texts' in res_obj:
                                    found_texts = res_obj['rec_texts']
                                    found_scores = res_obj.get('rec_scores', [1.0]*len(found_texts))
                                elif hasattr(res_obj, 'res') and 'rec_texts' in res_obj.res:
                                    found_texts = res_obj.res['rec_texts']
                                    found_scores = res_obj.res.get('rec_scores', [1.0]*len(found_texts))
                                elif isinstance(res_obj, list):
                                    found_texts = [line[1][0] for line in res_obj]
                                    found_scores = [line[1][1] for line in res_obj]

                            if found_texts:
                                combined = " ".join(found_texts)
                                avg_c = sum(found_scores) / len(found_scores)
                                validated = validate_plate(combined)
                                plate_text = validated if validated else combined.strip()
                                plate_conf = avg_c

                                cv2.putText(annotated, plate_text,
                                            (x1+px1, y1+py2 + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, PLATE_COLOR, 2)
                        except Exception:
                            pass

        detections.append({
            "vehicle_type":    label,
            "vehicle_conf":    round(conf_v, 3),
            "vehicle_bbox":    [x1, y1, x2, y2],
            "plate_text":      plate_text or "—",
            "plate_conf":      round(plate_conf, 3) if plate_conf else None,
            "v_infer_ms":      round(v_time_ms, 1),
            "p_infer_ms":      round(p_time_ms, 1),
        })

    # ── FALLBACK: If no vehicles found, run plate detector on FULL image ──
    if len(detections) == 0 and p_model:
        tp = time.perf_counter()
        p_res = p_model(processed_img, conf=0.10, verbose=False)[0]
        p_time_ms = (time.perf_counter() - tp) * 1000

        for pb in p_res.boxes:
            px1, py1, px2, py2 = map(int, pb.xyxy[0])
            plate_conf_val = float(pb.conf[0])

            pad = 2
            py1_p = max(0, py1 - pad)
            py2_p = min(processed_img.shape[0], py2 + pad)
            px1_p = max(0, px1 - pad)
            px2_p = min(processed_img.shape[1], px2 + pad)
            plate_crop = processed_img[py1_p:py2_p, px1_p:px2_p]

            if plate_crop.size == 0:
                continue

            cv2.rectangle(annotated, (px1, py1), (px2, py2), PLATE_COLOR, 2)

            plate_text = None
            plate_conf = plate_conf_val

            if ocr is not None:
                try:
                    plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    plate_crop = cv2.copyMakeBorder(plate_crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                    r_gen = ocr.predict(plate_crop)
                    r = list(r_gen) if r_gen else []

                    found_texts = []
                    found_scores = []

                    if r:
                        res_obj = r[0]
                        if isinstance(res_obj, dict) and 'rec_texts' in res_obj:
                            found_texts = res_obj['rec_texts']
                            found_scores = res_obj.get('rec_scores', [1.0] * len(found_texts))
                        elif hasattr(res_obj, 'res') and 'rec_texts' in res_obj.res:
                            found_texts = res_obj.res['rec_texts']
                            found_scores = res_obj.res.get('rec_scores', [1.0] * len(found_texts))
                        elif isinstance(res_obj, list):
                            found_texts = [line[1][0] for line in res_obj]
                            found_scores = [line[1][1] for line in res_obj]

                    if found_texts:
                        combined = " ".join(found_texts)
                        avg_c = sum(found_scores) / len(found_scores)
                        validated = validate_plate(combined)
                        plate_text = validated if validated else combined.strip()
                        plate_conf = avg_c

                        cv2.putText(annotated, plate_text,
                                    (px1, py2 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLATE_COLOR, 2)
                except Exception:
                    pass

            detections.append({
                "vehicle_type":    "Direct Plate",
                "vehicle_conf":    round(plate_conf, 3),
                "vehicle_bbox":    [px1, py1, px2, py2],
                "plate_text":      plate_text or "—",
                "plate_conf":      round(plate_conf, 3) if plate_conf else None,
                "v_infer_ms":      round(v_time_ms, 1),
                "p_infer_ms":      round(p_time_ms, 1),
            })

    return annotated, detections, v_time_ms

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")
    conf_thresh = st.slider("Vehicle Confidence", 0.10, 0.90, 0.15, 0.05)
    iou_thresh  = st.slider("IoU (NMS)", 0.10, 0.90, 0.45, 0.05)
    use_preprocess = st.checkbox("Night/Glare Preprocessing", value=False)

    st.markdown("---")
    st.markdown("### 🎯 Classes")
    detect_cars   = st.checkbox("🚗 Cars",        value=True)
    detect_motos  = st.checkbox("🏍️ Motorcycles", value=True)
    detect_buses  = st.checkbox("🚌 Buses",        value=True)
    detect_trucks = st.checkbox("🚛 Trucks",       value=True)
    active_classes = [k for k,v in {2: detect_cars, 3: detect_motos, 5: detect_buses, 7: detect_trucks}.items() if v]

    st.markdown("---")
    st.markdown("### 📊 System Metrics")

    with st.spinner("Loading models..."):
        try:
            v_model, p_model, device = load_models()
            ocr = load_ocr()
            model_loaded = True
        except Exception as e:
            st.error(f"Model error: {e}")
            model_loaded = False

    if model_loaded:
        plate_ready = "✅ Loaded" if p_model else "⚠️ Not Found"
        ocr_ready   = "✅ Ready"  if ocr    else "⚠️ Not Installed"
        st.markdown(f"""
        <div class="metric-card"><div class="metric-value" style="font-size:1rem;">YOLO11n</div><div class="metric-label">Vehicle Model</div></div>
        <div class="metric-card"><div class="metric-value" style="font-size:0.9rem;">{'🟢 GPU' if device=='cuda' else '🟡 CPU'}</div><div class="metric-label">Inference Device</div></div>
        <div class="metric-card"><div class="metric-value" style="font-size:0.85rem; color:#00ff88;">99.38%</div><div class="metric-label">Plate Model mAP@50</div></div>
        <div class="metric-card"><div class="metric-value" style="font-size:0.85rem;">{plate_ready}</div><div class="metric-label">Plate Detector</div></div>
        <div class="metric-card"><div class="metric-value" style="font-size:0.85rem;">{ocr_ready}</div><div class="metric-label">PaddleOCR</div></div>
        """, unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷 Image ANPR", "🎬 Video ANPR", "📊 Model Metrics"])

with tab1:
    col_l, col_r = st.columns([1, 1], gap="medium")
    with col_l:
        st.markdown("#### 📤 Upload Image")
        uploaded = st.file_uploader("JPEG / PNG / BMP", type=["jpg","jpeg","png","bmp","webp"])
        if uploaded:
            arr = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            if st.button("🚀 Run Full ANPR Pipeline") and model_loaded:
                annotated, dets, inf_ms = detect_frame(img, v_model, p_model, ocr, conf_thresh, iou_thresh, use_preprocess)
                st.session_state["img_out"] = (annotated, dets, inf_ms)

    with col_r:
        if "img_out" in st.session_state:
            annotated, dets, inf_ms = st.session_state["img_out"]
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(dets)}</div><div class="metric-label">Vehicles</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-card"><div class="metric-value">{inf_ms:.0f}ms</div><div class="metric-label">Inference</div></div>', unsafe_allow_html=True)
            with m3:
                plates = [d for d in dets if d["plate_text"] != "—"]
                st.markdown(f'<div class="metric-card"><div class="metric-value good">{len(plates)}</div><div class="metric-label">Plates Read</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="metric-card"><div class="metric-value">{1000/max(inf_ms,1):.0f}</div><div class="metric-label">FPS Equiv.</div></div>', unsafe_allow_html=True)

            for i, d in enumerate(dets):
                plate_str = f"🟢 **{d['plate_text']}**" if d["plate_text"] != "—" else "⚠️ No plate detected"
                st.markdown(f'<div class="detection-card"><b>#{i+1} {d["vehicle_type"]}</b> — Confidence: {d["vehicle_conf"]:.0%}<br>🔤 Plate: {plate_str}</div>', unsafe_allow_html=True)

with tab2:
    st.info("Upload a video to process frames.")
    vf = st.file_uploader("Upload MP4 / AVI", type=["mp4","avi","mov"])
    if vf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vf.read()); tmp.flush()
        if st.button("🚀 Run Video Pipeline"):
            cap = cv2.VideoCapture(tmp.name)
            ph = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                ann, dets, ms = detect_frame(frame, v_model, p_model, ocr, conf_thresh, iou_thresh, use_preprocess)
                ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()

with tab3:
    st.markdown("#### 📊 Model Performance Report")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><div class="metric-value good">99.38%</div><div class="metric-label">mAP@50</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-value good">98.97%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-value good">97.95%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><div class="metric-value">5.4 MB</div><div class="metric-label">Model Size</div></div>', unsafe_allow_html=True)

st.markdown("""
<hr style="border-color:#1a2a4a; margin:30px 0 10px 0;">
<p style="text-align:center; color:#334455; font-size:0.78rem;">
KnightSight EdgeVision &nbsp;·&nbsp; YOLO11n + PaddleOCR &nbsp;·&nbsp; 100% Offline Edge AI
</p>
""", unsafe_allow_html=True)
