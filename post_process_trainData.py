import os
import re
import shutil
from PIL import Image
import numpy as np

def extract_step_number(path):
    filename = os.path.basename(path)
    match = re.search(r"step(\d+)", filename)
    return int(match.group(1)) if match else -1

source_dir = 'raw/synth/unity/data'
destination_dir = 'cleaned/synth/data'
pattern = re.compile(r"step(\d+)\.camera_(\d+)\.semantic segmentation(?:_\d+)?\.png")

def contains_blue(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    return np.any((arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 255))

def is_fully_black(image_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    return np.sum(arr) == 0

os.makedirs(destination_dir, exist_ok=True)
camera_files = {}

for filename in sorted(os.listdir(source_dir)):
    if not filename.lower().endswith(".png"):
        continue
    match = pattern.match(filename)
    if not match:
        continue
    step_num, camera_id = match.groups()
    camera_folder = os.path.join(destination_dir, f"camera_{camera_id}")
    os.makedirs(camera_folder, exist_ok=True)
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(camera_folder, filename)
    shutil.copy2(src_path, dst_path)
    camera_files.setdefault(camera_id, []).append(dst_path)

for camera_id, files in camera_files.items():
    files = sorted(files, key=extract_step_number)
    sequence_idx = 0
    current_sequence = []
    camera_folder = os.path.join(destination_dir, f"camera_{camera_id}")

    for file_path in files:
        if contains_blue(file_path) or is_fully_black(file_path):
            if current_sequence:
                sequence_folder = os.path.join(camera_folder, f"sequence_{sequence_idx:03}")
                os.makedirs(sequence_folder, exist_ok=True)
                for img in current_sequence:
                    shutil.move(img, os.path.join(sequence_folder, os.path.basename(img)))
                sequence_idx += 1
                current_sequence = []
            os.remove(file_path)
        else:
            current_sequence.append(file_path)

    if current_sequence:
        sequence_folder = os.path.join(camera_folder, f"sequence_{sequence_idx:03}")
        os.makedirs(sequence_folder, exist_ok=True)
        for img in current_sequence:
            shutil.move(img, os.path.join(sequence_folder, os.path.basename(img)))

