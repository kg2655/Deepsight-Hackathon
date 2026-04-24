import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from PIL import Image
import numpy as np

# We import the pipeline logic we created earlier
from src.pipeline import process_frame

# --- Configure UI Aesthetics ---
st.set_page_config(
    page_title="KnightSight EdgeVision", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dynamic Design
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Inter', sans-serif;
        }
        /* Gradient Title */
        h1 {
            background: -webkit-linear-gradient(45deg, #58a6ff, #8957e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem !important;
            text-align: center;
            margin-bottom: 0px !important;
        }
        h3 {
            text-align: center;
            color: #8b949e;
            font-weight: 300;
            margin-top: 0px !important;
        }
        /* Styling the upload box */
        .css-1n76uvr, .css-1y4p8pa {
            background-color: rgba(22, 27, 34, 0.5) !important;
            border: 2px dashed #30363d !important;
            border-radius: 12px;
            padding: 20px;
            transition: 0.3s;
        }
        .css-1n76uvr:hover, .css-1y4p8pa:hover {
            border-color: #58a6ff !important;
            box-shadow: 0px 0px 15px rgba(88, 166, 255, 0.2);
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            width: 100%;
            transition: 0.2s;
            box-shadow: 0 4px 14px 0 rgba(46, 160, 67, 0.39);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #2ea043, #3fb950);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(46, 160, 67, 0.6);
        }
        /* Card for output */
        .output-card {
            background: #161b22;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #30363d;
            box-shadow: 0px 8px 24px rgba(0,0,0,0.4);
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1>EdgeVision Vehicle Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<h3>KnightSight Challenge - Lightweight ANPR Pipeline</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&q=80&w=300", use_container_width=True)
    st.markdown("## ⚙️ Pipeline Settings")
    
    st.markdown("**1. Plate Detection Model**")
    plate_model_path = st.text_input("Plate YOLO Weights Path", value="runs/detect/plate_detector_run/weights/best.pt")
    
    st.markdown("**2. Edge Optimizations (Bonus)**")
    use_clahe = st.checkbox("Enable Low-Light Enhancement (CLAHE)", value=True, help="Automatically corrects glare and shadow")
    use_onnx = st.checkbox("Use ONNX Runtime", value=False, help="Requires exporting .pt to .onnx first")

# --- Main Interface ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📤 Upload Source Media")
    uploaded_file = st.file_uploader("Upload an Image or Video", type=['jpg', 'jpeg', 'png', 'mp4'])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Display Preview
        if file_extension in ['jpg', 'jpeg', 'png']:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Temporary file to pass to pipeline
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_image_path = tmp_file.name

        process_button = st.button("🚀 Execute Edge Pipeline")

with col2:
    st.markdown("#### 🎯 Pipeline Output")
    
    if uploaded_file is not None and 'process_button' in locals() and process_button:
        with st.spinner("Processing through YOLOv8n & PaddleOCR..."):
            
            # Determine correct model path
            actual_plate_model = plate_model_path if os.path.exists(plate_model_path) else None
            if not actual_plate_model and plate_model_path:
                st.warning(f"Plate model '{plate_model_path}' not found! Falling back to raw OCR on vehicle crop.")

            # Run Pipeline
            output_image_path = "output_ui.jpg"
            detected_plates = process_frame(tmp_image_path, plate_model_path=actual_plate_model, output_path=output_image_path)
            
            # Display Results
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.image(output_image_path, caption="Annotated Output", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data Table
            st.markdown("#### 📊 Extracted Data")
            if detected_plates:
                df = pd.DataFrame(detected_plates)
                df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export as CSV", data=csv, file_name="anpr_results.csv", mime="text/csv")
            else:
                st.info("No valid Indian license plates detected in this frame.")
