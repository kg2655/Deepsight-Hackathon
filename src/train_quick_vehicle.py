import os
import urllib.request
import zipfile
from ultralytics import YOLO

def main():
    print("=" * 50)
    print("🚗 Quick Vehicle Model Training (COCO128)")
    print("=" * 50)
    
    # 1. Download small dataset
    if not os.path.exists("dataset/coco128"):
        print("Downloading dataset (~20MB)...")
        os.makedirs("dataset", exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
            "dataset/coco128.zip"
        )
        print("Extracting...")
        with zipfile.ZipFile("dataset/coco128.zip", 'r') as zip_ref:
            zip_ref.extractall("dataset/")
        print("Dataset ready!")
    
    # 2. Train model
    print("\n🚀 Starting training...")
    model = YOLO("yolo11n.pt")
    
    # Train only on vehicle classes: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
    model.train(
        data="coco128.yaml",
        classes=[2, 3, 5, 7],
        epochs=10,
        imgsz=640,
        batch=16,
        device="cpu",
        name="vehicle_detector_v2",
        project="runs/detect"
    )
    
    print("\n✅ Training complete!")
    print("Your new vehicle model is at: runs/detect/vehicle_detector_v2/weights/best.pt")

if __name__ == "__main__":
    main()
