import cv2
import time
import re
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import sys

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    print("Warning: PaddleOCR not installed.")
    OCR_AVAILABLE = False

# --- LOAD SOTA YOLO11 MODELS ---
try:
    print("Loading SOTA YOLO11 Vehicle Detection Model...")
    vehicle_model = YOLO("yolo11n.pt") # Upgraded to YOLO11
except Exception as e:
    print(f"Error loading vehicle model: {e}")
    sys.exit(1)

VEHICLE_CLASSES = [2, 3, 5, 7] # COCO Car, Motorcycle, Bus, Truck

def initialize_ocr():
    if not OCR_AVAILABLE: return None
    # Use GPU for PaddleOCR to crush inference time
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)

def apply_zero_dce_lite(img):
    """
    Simulated Zero-DCE lite (Adaptive Gamma Correction + Sharpening).
    Extremely fast edge-optimized enhancement for low-light/night.
    """
    # Auto gamma correction
    mid = 0.5
    mean = np.mean(img)
    gamma = math.log(mid * 255) / math.log(mean + 1e-3) if mean > 0 else 1.0
    gamma = np.clip(gamma, 0.4, 2.5)
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(img, table)
    
    # Unsharp Mask to deblur
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    return enhanced

def validate_plate_format(text):
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(clean_text) >= 7 and len(clean_text) <= 10:
        return clean_text
    return None

import math

class TemporalOCRFusion:
    """
    Instead of calling an LLM API, we aggregate OCR readings across 
    multiple tracked video frames and use confidence-weighted voting.
    """
    def __init__(self):
        self.history = defaultdict(list)
        
    def add_reading(self, track_id, ocr_text, confidence):
        validated = validate_plate_format(ocr_text)
        if validated:
            self.history[track_id].append((validated, confidence))
            
    def get_best_plate(self, track_id):
        if not self.history[track_id]: return None
        
        # Confidence-weighted voting
        votes = defaultdict(float)
        for text, conf in self.history[track_id]:
            votes[text] += conf
            
        # Return string with highest cumulative confidence score
        best_plate = max(votes.items(), key=lambda x: x[1])[0]
        return best_plate

def process_video_stream(video_path, plate_model_path=None, output_path="output_tracking.mp4"):
    """
    Advanced Pipeline: 
    YOLO11 + BoT-SORT Tracking + Zero-DCE Lite + Temporal OCR Fusion
    """
    print(f"--- Running Advanced Edge Pipeline on: {video_path} ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream.")
        return
        
    # Get video specs
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    plate_model = YOLO(plate_model_path) if plate_model_path else None
    ocr_engine = initialize_ocr()
    temporal_fusion = TemporalOCRFusion()
    
    frame_count = 0
    start_time = time.time()

    # We use YOLO's built-in BoT-SORT tracker
    # tracker="botsort.yaml" is vastly superior to bytetrack for handling occlusions
    results = vehicle_model.track(video_path, stream=True, classes=VEHICLE_CLASSES, tracker="botsort.yaml", conf=0.3)
    
    for r in results:
        frame_count += 1
        img = r.orig_img.copy()
        
        # Fast light enhancement
        img = apply_zero_dce_lite(img)
        
        boxes = r.boxes
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.int().cpu().tolist()
            
            for track_id, box in zip(track_ids, xyxys):
                x1, y1, x2, y2 = box
                
                # Draw vehicle box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                if plate_model:
                    vehicle_crop = img[y1:y2, x1:x2]
                    
                    # Ensure crop is valid
                    if vehicle_crop.size == 0: continue
                        
                    p_results = plate_model(vehicle_crop, conf=0.25, verbose=False)[0]
                    
                    for p_box in p_results.boxes:
                        px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        
                        # OCR
                        if ocr_engine is not None and plate_crop.size > 0:
                            ocr_result = ocr_engine.ocr(plate_crop, cls=False)
                            if ocr_result and ocr_result[0]:
                                text = ocr_result[0][0][1][0]
                                conf = ocr_result[0][0][1][1]
                                
                                # Add to temporal fusion buffer
                                temporal_fusion.add_reading(track_id, text, conf)
                                
                                # Get the historically verified best read
                                best_plate = temporal_fusion.get_best_plate(track_id)
                                
                                # Draw Plate
                                abs_px1, abs_py1 = x1 + px1, y1 + py1
                                abs_px2, abs_py2 = x1 + px2, y1 + py2
                                cv2.rectangle(img, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                                cv2.putText(img, best_plate if best_plate else text, 
                                           (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(img)
        
    cap.release()
    out.release()
    
    fps_avg = frame_count / (time.time() - start_time)
    print(f"\n✅ Advanced Video Pipeline Complete!")
    print(f"Processed {frame_count} frames at {fps_avg:.1f} FPS.")
    print(f"Saved optimized tracked output to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced KNIGHTSIGHT Pipeline (BoT-SORT + Temporal OCR)")
    parser.add_argument("--video", type=str, required=True, help="Video stream path")
    parser.add_argument("--plate_model", type=str, default=None, help="Path to trained YOLO11 plate model")
    args = parser.parse_args()
    
    process_video_stream(args.video, args.plate_model)
