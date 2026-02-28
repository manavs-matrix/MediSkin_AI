import os
from PIL import Image
import shutil
import random
from PIL import Image
import numpy as np

# Data verification 

path = "dataset/raw"

print("Dataset verification:\n")

for folder in os.listdir(path):
    folder_path = os.path.join(path,folder)
    if os.path.isdir(folder_path):
        images = os.listdir(folder_path)
        print(f"{folder}:{len(images)}images")

# check image format and resolution

path = "dataset/raw"
print("\nImage format and resolution check:\n")

image_sizes = {}
formats = {}

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path)
                size = img.size
                fmt = img.format

                image_sizes[size] = image_sizes.get(size, 0) + 1
                formats[fmt] = formats.get(fmt, 0) + 1
            except:
                print("Corrupted image:", img_path)

print("Image Sizes Distribution:")
for k, v in image_sizes.items():
    print(k, ":", v)

print("\nImage Formats Distribution:")
for k, v in formats.items():
    print(k, ":", v)

# Check corrupted images

print("\nChecking for corrupted images...\n")

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path)
                img.verify()
            except:
                print("❌ Corrupted image found. Deleting:", img_path)
                os.remove(img_path)

print("✅ Corrupted image check completed.")

# Data Spliting

raw_path = "dataset/raw"
processed_path = "dataset/processed"

train_ratio = 0.7
val_ratio = 0.15

print("\nStarting dataset split...\n")

for class_name in os.listdir(raw_path):
    class_path = os.path.join(raw_path, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        split_folder = os.path.join(processed_path, split, class_name)
        os.makedirs(split_folder, exist_ok=True)

        for img in splits[split]:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_folder, img)
            if not os.path.exists(dst):   # prevent duplicate copy
                shutil.copy(src, dst)

    print(f"✅ {class_name} split completed.")

print("\n🎉 Dataset split completed successfully.")

# Image Preprocessing (Resize + Normalize)

input_dir = "dataset/processed"
output_dir = "dataset/preprocessed"
IMG_SIZE = 224

print("\nStarting image preprocessing...\n")

for split in ["train", "val", "test"]:
    split_input_path = os.path.join(input_dir, split)
    split_output_path = os.path.join(output_dir, split)
    os.makedirs(split_output_path, exist_ok=True)

    for class_name in os.listdir(split_input_path):
        class_input_path = os.path.join(split_input_path, class_name)
        class_output_path = os.path.join(split_output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for img_name in os.listdir(class_input_path):
            img_path = os.path.join(class_input_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))

                img_array = np.array(img) / 255.0   # normalize

                save_path = os.path.join(class_output_path, img_name)
                Image.fromarray((img_array * 255).astype(np.uint8)).save(save_path)

            except Exception as e:
                print("❌ Failed processing:", img_path)

        print(f"✅ {split}/{class_name} preprocessing done.")

print("\n🎉 Image preprocessing completed successfully.")

