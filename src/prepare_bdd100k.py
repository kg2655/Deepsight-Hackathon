"""
BDD100K → YOLO format converter + training launcher
Run this once BDD100K is downloaded.

Expected BDD100K folder structure:
bdd100k/
  images/
    100k/
      train/   ← 70K images
      val/     ← 10K images
  labels/
    det_20/
      det_train.json
      det_val.json
"""

import json, os, shutil
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
BDD_ROOT   = r"C:\Users\Kannu Goyal\Downloads\bdd100k"   # ← update if different
OUT_ROOT   = r"C:\Users\Kannu Goyal\OneDrive\Desktop\deepsight\dataset\bdd_vehicle"
MAX_TRAIN  = 15000   # use 15K images — enough for good accuracy, fast training
MAX_VAL    = 3000

# BDD100K class → YOLO class ID mapping (vehicle classes only)
BDD_TO_YOLO = {
    "car":        0,
    "truck":      1,
    "bus":        2,
    "motorcycle": 3,
    "bicycle":    4,   # optional — remove if you don't want bicycles
}
# ──────────────────────────────────────────────────────────────────────────────


def convert_split(json_path, img_src_dir, out_img_dir, out_lbl_dir, max_imgs):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"\nLoading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    count = 0
    skipped = 0

    for item in tqdm(data[:max_imgs * 2]):  # iterate more to find valid ones
        if count >= max_imgs:
            break

        fname = item["name"]
        img_path = os.path.join(img_src_dir, fname)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        labels = item.get("labels", [])
        yolo_lines = []

        for lbl in labels:
            cat = lbl.get("category", "")
            if cat not in BDD_TO_YOLO:
                continue

            cls_id = BDD_TO_YOLO[cat]
            box2d = lbl.get("box2d")
            if not box2d:
                continue

            # BDD100K gives absolute pixel coords → convert to YOLO normalized
            # Image size for BDD100K is always 1280×720
            W, H = 1280, 720
            x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            w  = (x2 - x1) / W
            h  = (y2 - y1) / H

            # Clamp to [0,1]
            cx, cy, w, h = [max(0, min(1, v)) for v in [cx, cy, w, h]]

            if w > 0.001 and h > 0.001:
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            continue  # skip images with no vehicle labels

        # Copy image
        shutil.copy2(img_path, os.path.join(out_img_dir, fname))

        # Write label
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        with open(os.path.join(out_lbl_dir, lbl_name), "w") as f:
            f.write("\n".join(yolo_lines))

        count += 1

    print(f"✅ Done: {count} images saved, {skipped} skipped (not found)")
    return count


def create_yaml():
    yaml_content = f"""# BDD100K Vehicle Detection Dataset
path: {OUT_ROOT}
train: images/train
val: images/val

nc: {len(BDD_TO_YOLO)}
names: {list(BDD_TO_YOLO.keys())}
"""
    yaml_path = os.path.join(OUT_ROOT, "bdd_vehicle.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ YAML saved: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    print("=" * 55)
    print("BDD100K → YOLO Converter")
    print("=" * 55)

    train_json = os.path.join(BDD_ROOT, "labels", "det_20", "det_train.json")
    val_json   = os.path.join(BDD_ROOT, "labels", "det_20", "det_val.json")
    train_imgs = os.path.join(BDD_ROOT, "images", "100k", "train")
    val_imgs   = os.path.join(BDD_ROOT, "images", "100k", "val")

    # Check paths exist
    for p in [train_json, val_json, train_imgs, val_imgs]:
        if not os.path.exists(p):
            print(f"❌ NOT FOUND: {p}")
            print("Update BDD_ROOT at the top of this script!")
            exit(1)

    convert_split(train_json, train_imgs,
                  f"{OUT_ROOT}/images/train", f"{OUT_ROOT}/labels/train", MAX_TRAIN)
    convert_split(val_json, val_imgs,
                  f"{OUT_ROOT}/images/val",   f"{OUT_ROOT}/labels/val",   MAX_VAL)

    yaml_path = create_yaml()

    print("\n" + "=" * 55)
    print("✅ Dataset ready! Starting training...")
    print("=" * 55)

    # Auto-start training
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(
        data=yaml_path,
        epochs=25,
        imgsz=640,
        batch=16,
        device=0,
        name="vehicle_bdd100k",
        patience=8,
        save=True,
        project="runs/detect",
        exist_ok=True,
    )
    print("\n✅ Training complete! Check runs/detect/vehicle_bdd100k/weights/best.pt")
