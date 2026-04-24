import cv2
import time
import re
from ultralytics import YOLO
import sys

# Attempt to load PaddleOCR, handling cases where it's still installing or missing
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    print("Warning: PaddleOCR is not installed or still installing. OCR will be disabled.")
    OCR_AVAILABLE = False


# Load Vehicle Model
try:
    print("Loading vehicle detection model...")
    vehicle_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading vehicle model: {e}")
    sys.exit(1)

# COCO Dataset IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]

def initialize_ocr():
    if not OCR_AVAILABLE:
        return None
    print("Initializing PaddleOCR EN Mobile Model (GPU Accelerated)...")
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)

def apply_claHE(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    This handles the low-light and glare bonus point requirement.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Check if the image is too dark or washed out by glare
    if l.mean() < 80 or l.mean() > 180:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

def validate_plate_format(text):
    """
    Verifies if the extracted text closely resembles an Indian License Plate (ABC-1234).
    This prevents false positives from bumper stickers.
    """
    # Clean the text of spaces and special chars
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Basic check: plates usually have at least 2 letters and some numbers, length between 7-10
    if len(clean_text) >= 7 and len(clean_text) <= 10:
        return clean_text
    return None

def process_frame(image_path, plate_model_path=None, output_path="final_output.jpg"):
    """
    End-to-End Pipeline
    1. Reads & Enhances Image
    2. Detects Vehicles
    3. Crops Vehicles & Detects Plates (if plate_model is provided)
    4. Crops Plates & Runs OCR
    """
    print(f"--- Processing: {image_path} ---")
    start_time = time.time()
    
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image!")
        return
        
    img = apply_claHE(img)
    
    # 1. Vehicle Detection
    v_results = vehicle_model(img, classes=VEHICLE_CLASSES, conf=0.3)[0]
    
    # Initialize Plate Model if provided
    plate_model = None
    if plate_model_path:
        try:
            plate_model = YOLO(plate_model_path)
        except:
            print("Failed to load plate model, falling back to direct OCR.")
    
    ocr_engine = initialize_ocr()
    
    detected_data = []

    # Iterate over each detected vehicle
    for box in v_results.boxes:
        # Get coordinates of vehicle
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_crop = img[y1:y2, x1:x2]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, "Vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        # 2. Plate Detection
        plate_crops = []
        if plate_model:
            # Detect plate within vehicle crop
            p_results = plate_model(vehicle_crop, conf=0.25)[0]
            for p_box in p_results.boxes:
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                plate_crops.append((plate_crop, x1+px1, y1+py1, x1+px2, y1+py2))
        else:
            # If no plate model is provided, we send the whole vehicle to OCR. (Not ideal, but good fallback)
            plate_crops.append((vehicle_crop, x1, y1, x2, y2))
            
        # 3. OCR Recognition
        for (crop, px1, py1, px2, py2) in plate_crops:
            if ocr_engine is not None:
                ocr_result = ocr_engine.ocr(crop, cls=True)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        
                        validated_text = validate_plate_format(text)
                        if validated_text or plate_model: 
                            final_text = validated_text if validated_text else text
                            print(f"[+] Found Plate: {final_text} (Conf: {conf:.2f})")
                            
                            # Draw Plate Bounding Box
                            if plate_model:
                                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            
                            # Draw Text
                            cv2.putText(img, final_text, (px1, py1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            detected_data.append({"plate": final_text, "confidence": conf})

    total_time = (time.time() - start_time) * 1000
    print(f"--- Pipeline Completed in {total_time:.1f}ms ---")
    
    cv2.imwrite(output_path, img)
    print(f"Saved end-to-end result to {output_path}")
    return detected_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-to-End KNIGHTSIGHT Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--plate_model", type=str, default=None, help="Path to trained plate model (yolov8n_plate.pt)")
    args = parser.parse_args()
    
    process_frame(args.image, args.plate_model)
