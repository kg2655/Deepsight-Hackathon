import os
import json
import glob
from PIL import Image

def convert_dataset(data_dir):
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    json_files = glob.glob(os.path.join(labels_dir, "*.json"))
    print(f"Found {len(json_files)} label files to convert...")
    
    for json_path in json_files:
        basename = os.path.basename(json_path)
        name_only = os.path.splitext(basename)[0]
        
        # Get image dimensions
        img_path = os.path.join(images_dir, name_only + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, name_only + ".png")
            
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for {json_path}")
            continue
            
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue
            
        # Parse JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        txt_path = os.path.join(labels_dir, name_only + ".txt")
        with open(txt_path, 'w') as f:
            for item in data:
                # Calculate normalized YOLO format
                x = item['x']
                y = item['y']
                w = item['width']
                h = item['height']
                
                # YOLO format: class_id x_center y_center width height
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                w_norm = w / float(img_w)
                h_norm = h / float(img_h)
                
                # Force class 0 since it's a single class detector (license plate)
                class_id = 0 
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
        # Optional: Remove the JSON file to keep the directory clean, or keep it.
        # os.remove(json_path)
        
    print("Conversion complete!")

    # Generate data.yaml
    yaml_content = f"""train: {images_dir}
val: {images_dir}  # Using train set for validation just to ensure script runs

nc: 1
names: ['license_plate']
"""
    yaml_path = os.path.join(os.path.dirname(data_dir), "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Generated {yaml_path}")

if __name__ == "__main__":
    convert_dataset(r"C:\Users\Kannu Goyal\Downloads\deepsight\train set")
