import re
from collections import defaultdict
import os
import torch
from torchvision import transforms
from PIL import Image
import math

from PIL import Image
from torchvision import transforms
import math

class DynamicPadding:
    def __init__(self, target_size, fill=255, padding_mode="constant"):
        """
        Initialize the dynamic padding class:
        Args:
            target_size: Tuple (width, height) or single int for square
            fill: Padding fill value (255 for white)
            padding_mode: Padding mode ('constant', 'edge', etc)
        """
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        print("TARGET SIZE:")
        print(self.target_size)
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Add dynamic padding to the image:
        Args:
            img: PIL Image to pad
        Returns:
            Padded image as a PIL Image
        """
        print(f"before processing: {img.size}")
        width, height = img.size
        target_width, target_height = self.target_size

        scale_factor = min(target_width / width, target_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # resize to maintain aspect ratio
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        width, height = img.size
        print(f"after width, height: {width}, {height}")

        # calculate the padding sizes
        left = max((target_width - width) // 2, 0)
        top = max((target_height - height) // 2, 0)
        right = max(target_width - (width + left), 0)
        bottom = max(target_height - (height + top), 0)

        # add padding
        img = transforms.functional.pad(img, (left, top, right, bottom), fill=self.fill, padding_mode=self.padding_mode)
        print("AFTER TRANSFORM:")
        print(img)
        return img

os.chdir("C:/vvcc/archive-dump-Sept-2024")
cwd = os.getcwd()

input_dir = "C:/vvcc/archive-dump-Sept-2024/compiled-photos/_master_set_photos/"
output_dir = "C:/vvcc/archive-dump-Sept-2024/processed_photos/"

os.makedirs(output_dir, exist_ok=True)

# # regex pattern to match SKUs with variable lengths (3-5 digits)
sku_pattern = r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$"

# resize the images first and then apply data augmentation
transform = transforms.Compose([
    DynamicPadding(target_size=224, fill=255),  # dims will be 224x224
    transforms.ToTensor()                       
])

augment_transform = transforms.Compose([
    DynamicPadding(target_size=224, fill=255),
    transforms.RandomRotation(degrees=2),  # rotate between -2 to +2 degrees
    transforms.ToTensor()
])

grouped_images = defaultdict(list) # defaultdict initializes a dictionary's keys with default values

for filename in os.listdir(input_dir):
# for key, file_paths in grouped_images.items():
#     if file_paths:
#         first_file = file_paths[0]
#         print(f"Key: {key}, Image Path: {first_file}")
#         break
     if filename.endswith(".jpg"):
        match = re.match(sku_pattern, filename)
        if match:
            sku = match.group(1)  # extract the sku from the filename
            file_path = os.path.join(input_dir, filename)
            grouped_images[sku].append(file_path)
            output_path = os.path.join(output_dir, filename)

            try:
                img = Image.open(file_path)
                padded_image = transform(img)
                padded_image_pil = transforms.ToPILImage()(padded_image)
                # padded_image_pil.save(output_path) # uncomment these if you want to save the images unrotated & with padding
                # print(f"Processed and saved: {output_path}")

                for angle, suffix in [(-2, "_rotated_left"), (2, "_rotated_right")]:
                    augmented_image = augment_transform(img)
                    augmented_output_path = os.path.join(output_dir, f"{sku}{suffix}.jpg")
                    augmented_image_pil = transforms.ToPILImage()(augmented_image)
                    augmented_image_pil.save(augmented_output_path)

                print(f"Processed and saved: {output_path} and augmentations.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
