import cv2
import time
from ultralytics import YOLO
import sys

# Load the pretrained YOLOv8n model
# It will be downloaded automatically the first time this runs if not present.
try:
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# COCO Dataset IDs for vehicles
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def detect_vehicles(image_path, output_path="output.jpg"):
    """
    Detects vehicles in the given image and saves a copy with bounding boxes.
    """
    print(f"Processing image: {image_path}")
    start_time = time.time()
    
    # Run inference only looking for vehicle classes
    results = model(image_path, classes=VEHICLE_CLASSES, conf=0.3)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Render the results on the image
    result = results[0]
    
    # Count how many vehicles were found
    detections = len(result.boxes)
    print(f"Detected {detections} vehicles in {inference_time:.1f}ms")
    
    # result.plot() returns a BGR numpy array of the image with annotations
    annotated_frame = result.plot()
    
    # Save the output image
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved annotated image to {output_path}")
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle Detection Module")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image")
    args = parser.parse_args()
    
    # We will need to ensure the user actually passes a valid image path.
    detect_vehicles(args.image, args.output)
