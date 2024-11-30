
import os
import re
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
image_dir = "C:/vvcc/archive-dump-Sept-2024/processed_photos/"

sku_pattern = r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$"

catalog_csv = pd.read_csv(catalog)
dresses = catalog_csv[catalog_csv['itemtype'].str.contains('dress', case=False, na=False)]
dress_skus = dresses['sku'].tolist()
dress_skus = [str(sku) for sku in dress_skus]

dresses = dresses.dropna(subset=['era'])
dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
print(dresses['era'].unique())

unique_eras = sorted(dresses['era'].unique())
era_to_index = {era: idx for idx, era in enumerate(unique_eras)}
index_to_era = {idx: era for era, idx in era_to_index.items()}
dresses['era'] = dresses['era'].map(era_to_index)
print("Mapping Era To Index:", era_to_index)
print(dresses['era'].unique())

image_mapping = {}
sku_image_count = {}

for filename in os.listdir(image_dir):
    # get the SKU from the filename
    sku_match = re.match(r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$", filename)
    if sku_match: # for every file in image_dir
        sku = sku_match.group(1) # capture the sku
        sku = sku.lstrip('v')
        sku = str(sku)

        if sku in dress_skus:
            if sku not in image_mapping:
                image_mapping[sku] = []
                sku_image_count[sku] = 0
            image_mapping[sku].append(os.path.join(image_dir, filename))
            sku_image_count[sku] += 1

total_images = sum(sku_image_count.values())
total_skus = len(sku_image_count)
average_images_per_sku = total_images / total_skus if total_skus > 0 else 0

print(f"Total SKUs: {total_skus}")
print(f"Total Images: {total_images}")
print(f"Average Images per SKU: {average_images_per_sku:.2f}")

# all images mapped, now build CNN
transform = transforms.Compose([
    transforms.ToTensor(),         # convert PIL image to PyTorch tensor, scaling happens automatically in this step
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize for RGB channels (these values are standard for pre-trained if using later)
                         std=[0.229, 0.224, 0.225])
])

image_tensors = []
labels = []

# print(f"image mapping: {image_mapping.items()}")

for sku, image_paths in image_mapping.items():
    for image_path in image_paths:
        try:
            numeric_sku = int(sku) # needs to be int to compare to dresses['sku']
            label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
            if label_row.empty: # there are about 70 images that don't have a value in labels, filter them out
                print(f"No label found for SKU: {sku}")
                continue
            label = dresses.loc[dresses['sku'] == numeric_sku, 'era'].values[0]
            numeric_label = int(label)
            labels.append(numeric_label)
            
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img)
            image_tensors.append(img_tensor)
            # print(f"Processed image: {image_path} for SKU: {sku}")
        except Exception as e:
            continue
            # print(f"Error processing {image_path}: {e}")

# stack images into a tensor and convert labels to Tensor
image_tensors = torch.stack(image_tensors)
labels = torch.tensor(labels, dtype=torch.long)

print(f"Image tensor shape: {image_tensors.shape}")
print(f"Labels shape: {labels.shape}")

torch.save(image_tensors, "image_tensors.pt")
torch.save(labels, "labels.pt")
print("image_tensors and labels pickled & saved")