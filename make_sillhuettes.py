from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import os
import torch

input_root = "input\frames"
output_root = "output\masks"
os.makedirs(output_root, exist_ok=True)

yolo_model = YOLO("yolov8s.pt")
sam_checkpoint = "sam_weights\sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)

selected_classes = [0, 1, 2, 3, 5, 7]

for folder in sorted(os.listdir(input_root)):
    input_folder = os.path.join(input_root, folder)
    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(output_root, folder)
    os.makedirs(output_folder, exist_ok=True)

    for img_name in sorted(os.listdir(input_folder)):
        if not img_name.lower().endswith(('.jpg', '.png')):
            continue

        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = yolo_model(image_rgb)
        boxes = [
            r.boxes.xyxy[i].cpu().numpy().astype(int)
            for r in results
            for i, cls in enumerate(r.boxes.cls)
            if int(cls) in selected_classes
        ]

        predictor.set_image(image_rgb)
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for x0, y0, x1, y1 in boxes:
            masks, _, _ = predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
            full_mask = np.maximum(full_mask, masks[0].astype(np.uint8) * 255)

        cv2.imwrite(os.path.join(output_folder, img_name), full_mask)