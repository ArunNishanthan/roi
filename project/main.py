import os
import shutil
import random
from pathlib import Path
import yaml
import subprocess
import cv2
from ultralytics import YOLO

IMG_DIR = "D:\\cheque\\project\\images"
LBL_DIR = "D:\\cheque\\project\\labels"
TRAIN_IMG_DIR = "D:\\cheque\\project\\images/train"
VAL_IMG_DIR = "D:\\cheque\\project\\images/val"
TRAIN_LBL_DIR = "D:\\cheque\\project\\labels/train"
VAL_LBL_DIR = "D:\\cheque\\project\\labels/val"
CLASSES_FILE = "D:\\cheque\\project\\classes.txt"
DATA_YAML = "D:\\cheque\\project\\cheque.yaml"
SPLIT_RATIO = 0.8
YOLO_MODEL = "yolov8n.pt"  # You can change to yolov8m.pt, yolov8l.pt, etc.
EPOCHS = 100
IMGSZ = 640

def detect():
    MODEL_PATH = 'D:\\cheque\\runs\detect\\train\\weights\\best.pt'   # Path to trained YOLOv8 model
    SOURCE_PATH = 'D:\\cheque\\project\\test\\'                           # Folder or image path
    OUTPUT_PATH = 'D:\\cheque\\project\\output\\'                            # Output folder for annotated images
    IMG_SIZE = 640                                     # Optional: image resize size

    # ==== Load Class Names ====
    with open(CLASSES_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # ==== Create Output Folder ====
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ==== Load YOLOv8 Model ====
    model = YOLO(MODEL_PATH)

    # ==== Detect Images ====
    def detect_and_draw(image_path, model, class_names):
        image = cv2.imread(image_path)
        if image is None:
            print(f"[!] Could not read {image_path}")
            return

        # Run YOLOv8 prediction
        results = model.predict(source=image, imgsz=IMG_SIZE, conf=0.25, verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                label = f"{class_names[cls]} {conf:.2f}"
                color = (0, 255, 0)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save annotated image
        filename = os.path.basename(image_path)
        output_file = os.path.join(OUTPUT_PATH, filename)
        cv2.imwrite(output_file, image)
        print(f"[✓] Saved: {output_file}")

    # ==== Run Detection ====
    if os.path.isdir(SOURCE_PATH):
        for filename in os.listdir(SOURCE_PATH):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                detect_and_draw(os.path.join(SOURCE_PATH, filename), model, class_names)
    else:
        detect_and_draw(SOURCE_PATH, model, class_names)


def main():
    detect()
    # train()


# def train():
#     for folder in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
#         Path(folder).mkdir(parents=True, exist_ok=True)

#     # Step 2: Read and split images
#     images = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]
#     random.shuffle(images)
#     split = int(len(images) * SPLIT_RATIO)

#     for i, img_file in enumerate(images):
#         name, ext = os.path.splitext(img_file)
#         label_file = f"{name}.txt"
#         src_img = os.path.join(IMG_DIR, img_file)
#         src_lbl = os.path.join(LBL_DIR, label_file)

#         if not os.path.exists(src_lbl):
#             print(f"Warning: label missing for image {img_file}")
#             continue

#         if i < split:
#             shutil.copy(src_img, os.path.join(TRAIN_IMG_DIR, img_file))
#             shutil.copy(src_lbl, os.path.join(TRAIN_LBL_DIR, label_file))
#         else:
#             shutil.copy(src_img, os.path.join(VAL_IMG_DIR, img_file))
#             shutil.copy(src_lbl, os.path.join(VAL_LBL_DIR, label_file))

# # Step 3: Read classes.txt and write YAML
#     with open(CLASSES_FILE, "r") as f:
#         class_names = [line.strip() for line in f.readlines()]

#     data_yaml = {
#         "train": str(Path(TRAIN_IMG_DIR).resolve()),
#         "val": str(Path(VAL_IMG_DIR).resolve()),
#         "nc": len(class_names),
#         "names": class_names
#     }

#     with open(DATA_YAML, "w") as f:
#         yaml.dump(data_yaml, f)

#     print(f"✅ Created {DATA_YAML} with {len(class_names)} classes.")

#     # Step 4: Train YOLOv8
#     print("🚀 Starting training...")
#     train_cmd = [
#         "yolo", "task=detect", "mode=train",
#         f"model={YOLO_MODEL}",
#         f"data={DATA_YAML}",
#         f"epochs={EPOCHS}",
#         f"imgsz={IMGSZ}"
#     ]

#     # Run training
#     subprocess.run(train_cmd)


if __name__ == "__main__":
    main()
