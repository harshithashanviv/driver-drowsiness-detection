import os
import cv2

# Absolute dataset path (recommended)
dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")

valid_extensions = (".jpg", ".jpeg", ".png")

total = 0
removed = 0

print("Starting Advanced Data Cleaning...\n")

for split in ["train", "test"]:
    split_path = os.path.join(dataset_path, split)
    
    if not os.path.exists(split_path):
        print(f"{split_path} not found")
        continue
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            total += 1
            
            # Remove invalid extension
            if not file.lower().endswith(valid_extensions):
                os.remove(file_path)
                removed += 1
                print(f"Removed invalid file: {file_path}")
                continue
            
            # Check corrupted image
            img = cv2.imread(file_path)
            
            if img is None:
                os.remove(file_path)
                removed += 1
                print(f"Removed corrupted image: {file_path}")
                continue
            
            # Check very small images (bad quality)
            h, w = img.shape[:2]
            
            if h < 20 or w < 20:
                os.remove(file_path)
                removed += 1
                print(f"Removed too small image: {file_path}")

print("\nCleaning Completed")
print(f"Total checked: {total}")
print(f"Removed: {removed}")
print(f"Remaining: {total - removed}")