from ultralytics import YOLO
import argparse
import os

def train_plate_model(data_yaml_path, epochs=30, batch_size=16):
    """
    Trains a YOLOv8n model specifically for license plate detection.
    
    Args:
        data_yaml_path (str): Path to the dataset.yaml file (YOLO format)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
    """
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found. Please ensure your downloaded dataset is extracted and the path is correct.")
        return
        
    print(f"Initializing YOLOv8n for plate detection training on {data_yaml_path}...")
    
    # Load a fresh YOLOv8n model (weights)
    model = YOLO("yolov8n.pt")
    
    # Train the model
    # Note: On Colab, change device to 0 (for GPU). For CPU it defaults to 'cpu'.
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        name="plate_detector_run",
        project="runs/detect"
    )
    
    print("\nTraining complete!")
    print("Your trained plate detection model is located at: runs/detect/plate_detector_run/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plate Detector YOLOv8 Modal")
    parser.add_argument("--data", type=str, required=True, help="Path to your dataset's data.yaml file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    train_plate_model(args.data, args.epochs, args.batch)
