from ultralytics import YOLO
import argparse
import os

def train_plate_model(data_yaml_path, epochs=30, batch_size=16):
    """
    Trains an Advanced YOLO11n model specifically for license plate detection.
    """
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found. Please ensure your downloaded dataset is extracted and the path is correct.")
        return
        
    print(f"Initializing YOLO11n for SOTA plate detection training on {data_yaml_path}...")
    
    # Load a fresh YOLO11n model (weights)
    # This will automatically download yolo11n.pt if not present.
    model = YOLO("yolo11n.pt")
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        name="plate_detector_yolo11",
        project="runs/detect",
        device=0 # Enforce GPU Usage
    )
    
    print("\n✅ Training complete!")
    print("Your advanced YOLO11 plate detection model is located at: runs/detect/plate_detector_yolo11/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOTA Plate Detector (YOLO11)")
    parser.add_argument("--data", type=str, required=True, help="Path to your dataset's data.yaml file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    train_plate_model(args.data, args.epochs, args.batch)
