import os
import random
import shutil


def split_dataset(original_dataset_dir, output_base_dir,targets, split_ratio=0.8, seed=42):
    random.seed(seed)
    folders = [name for name in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, name))]

    for target in folders:
        # Create output folders
        train_dir = os.path.join(output_base_dir, 'train', target)
        test_dir = os.path.join(output_base_dir, 'test', target)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # List all images
        all_images = [f for f in os.listdir(f"{original_dataset_dir}{target}") if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        random.shuffle(all_images)

        # Split
        split_index = int(len(all_images) * split_ratio)
        train_images = all_images[:split_index]
        test_images = all_images[split_index:]

        # Copy files
        for img_name in train_images:
            src_path = os.path.join(f"{original_dataset_dir}{target}", img_name)
            dst_path = os.path.join(train_dir, img_name)
            shutil.copy2(src_path, dst_path)

        for img_name in test_images:
            src_path = os.path.join(f"{original_dataset_dir}{target}", img_name)
            dst_path = os.path.join(test_dir, img_name)
            shutil.copy2(src_path, dst_path)

    print(f"Dataset split complete! {len(train_images)} training images, {len(test_images)} testing images.")


base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{base_path}/car_dataset/"
targets = [
    "Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley",
    "BMW", "Bugatti", "Buick", "Cadillac", "Chevrolet", "Chrysler","Citroen","Daewoo","Dodge","Ferrari",""
]
# Example usage:
original_dataset = f'{dataset_path}'
output_dataset = f'{base_path}/car_dataset2/'
split_dataset(original_dataset, output_dataset,targets)
