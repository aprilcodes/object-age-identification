

import os
import re
import pandas as pd
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, batched_images, batched_labels, skus, transform=None):
        self.batched_images = batched_images
        self.batched_labels = batched_labels
        self.skus = skus
        self.transform = transform

    def __len__(self):
        return len(self.batched_images)

    def __getitem__(self, idx):
        images = self.batched_images[idx]
        label = self.batched_labels[idx]
        sku = self.skus[idx]

        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
        return images, label, sku


catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
image_dir_test = "C:/vvcc/archive-dump-Sept-2024/processed_photos_test/"
image_dir_train = "C:/vvcc/archive-dump-Sept-2024/processed_photos_train/"

sku_pattern = r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$"

catalog_csv = pd.read_csv(catalog)
dresses = catalog_csv[catalog_csv['itemtype'].str.contains('dress', case=False, na=False)]
dress_skus = dresses['sku'].tolist()
dress_skus = [str(sku) for sku in dress_skus]

dresses = dresses.dropna(subset=['era'])
dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
print(dresses['era'].unique())
print(dresses['era'].value_counts())

unique_eras = sorted(dresses['era'].unique())
era_to_index = {era: idx for idx, era in enumerate(unique_eras)}
index_to_era = {idx: era for era, idx in era_to_index.items()}
dresses['era'] = dresses['era'].map(era_to_index)
print("Mapping Era To Index:", era_to_index)
print(dresses['era'].unique())

image_mapping_test = {}
image_mapping_train = {}

sku_image_count_test = {}
sku_image_count_train = {}

for filename in os.listdir(image_dir_test):
    sku_match = re.match(r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$", filename)
    if sku_match: # for every file in image_dir
        sku = sku_match.group(1) # capture the sku
        sku = sku.lstrip('v')
        sku = str(sku)

        if sku in dress_skus:
            if sku not in image_mapping_test:
                image_mapping_test[sku] = []
                sku_image_count_test[sku] = 0
            image_mapping_test[sku].append(os.path.join(image_dir_test, filename))
            sku_image_count_test[sku] += 1

for filename in os.listdir(image_dir_train):
    # get the SKU from the filename
    sku_match = re.match(r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$", filename)
    if sku_match: # for every file in image_dir
        sku = sku_match.group(1) # capture the sku
        sku = sku.lstrip('v')
        sku = str(sku)

        if sku in dress_skus:
            if sku not in image_mapping_train:
                image_mapping_train[sku] = []
                sku_image_count_train[sku] = 0
            image_mapping_train[sku].append(os.path.join(image_dir_train, filename))
            sku_image_count_train[sku] += 1

# all images mapped, now build CNN
transform = transforms.Compose([
    transforms.ToTensor(),         # convert PIL image to PyTorch tensor, scaling happens automatically in this step
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize for RGB channels (these values are standard for pre-trained if using later)
                         std=[0.229, 0.224, 0.225])
])

image_tensors_train = []
image_tensors_test = []

labels_train = []
labels_test = []

skus_train = []
skus_test = []

batched_image_tensors_train = []
batched_labels_train = []
batched_skus_train = []

######### for each in image_tensors_train
for sku, image_paths in image_mapping_train.items():
    numeric_sku = int(sku)  # Convert SKU to int once per group of images
    try:
        # Retrieve the label for the SKU
        label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
        if label_row.empty:
            print(f"No label found for SKU: {sku}")
            continue
        numeric_label = int(label_row.values[0])
        # Process and aggregate all images for the SKU
        sku_images = []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img)
            sku_images.append(img_tensor)  # Collect all images for this SKU

        # Append the aggregated batch, label, and SKU
        batched_image_tensors_train.append(torch.stack(sku_images))  # Stack all images into a tensor
        batched_labels_train.append(numeric_label)  # One label for the SKU
        batched_skus_train.append(numeric_sku)  # The SKU itself

    except Exception as e:
        print(f"Error processing SKU {sku}: {e}")
        continue

print("check train set:")
print(f"skus train: {len(batched_skus_train)}")
print(f"unique skus: {len(set(batched_skus_train))}")
print(f"how many image tensors: {len(batched_image_tensors_train)}")
print(f"how many labels: {len(batched_labels_train)}")

total_image_paths = sum(len(image_paths) for image_paths in image_mapping_train.values())
print(f"Total image paths in image_mapping_train: {total_image_paths}")

batched_image_tensors_test = []
batched_labels_test = []
batched_skus_test = []

for sku, image_paths in image_mapping_test.items():
    numeric_sku = int(sku)
    try:
        label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
        if label_row.empty:
            print(f"No label found for SKU: {sku}")
            continue
        numeric_label = int(label_row.values[0])

        sku_images = []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img)
            sku_images.append(img_tensor)

        batched_image_tensors_test.append(torch.stack(sku_images))  # Stack all images into a tensor
        batched_labels_test.append(numeric_label)  # One label for the SKU
        batched_skus_test.append(numeric_sku)  # The SKU itself

    except Exception as e:
        print(f"Error processing SKU {sku}: {e}")
        continue

print("check test set:")
print(len(batched_skus_test))
print(len(batched_image_tensors_test))

total_image_paths_test = sum(len(image_paths) for image_paths in image_mapping_test.values())
print(f"Total image paths in image_mapping_test: {total_image_paths_test}")

sku_to_images_train = defaultdict(list)
sku_to_label_train = {}

for image, label, sku in zip(batched_image_tensors_train, batched_labels_train, batched_skus_train):
    sku_to_images_train[sku].append(image)
    sku_to_label_train[sku] = label

print("check length of sku_to_images_train:")
print(len(sku_to_images_train))

skus_list_train = list(sku_to_images_train.keys())
batched_images_train = [torch.stack(sku_to_images_train[sku]) for sku in skus_list_train]  # Stack images for each SKU
batched_labels_train = [sku_to_label_train[sku] for sku in skus_list_train]
train_dataset = CustomDataset(batched_images_train, batched_labels_train, skus_list_train)

sku_to_images_test = defaultdict(list)
sku_to_label_test = {}

for image, label, sku in zip(batched_image_tensors_test, batched_labels_test, batched_skus_test):
    sku_to_images_test[sku].append(image)
    sku_to_label_test[sku] = label

print("check length of sku_to_images_test:")
print(len(sku_to_images_test))

skus_list_test = list(sku_to_images_test.keys())
batched_images_test = [torch.stack(sku_to_images_test[sku]) for sku in skus_list_test]  # Stack images for each SKU
batched_labels_test = [sku_to_label_test[sku] for sku in skus_list_test]

test_dataset = CustomDataset(batched_images_test, batched_labels_test, skus_list_test)

print(f"Number of samples in train_dataset: {len(train_dataset)}")
print(f"Number of samples in test_dataset: {len(test_dataset)}")

torch.save(train_dataset, "train_dataset_DRESSES_ONLY.pt")
torch.save(test_dataset, "test_dataset_DRESSES_ONLY.pt") # must be loaded with dataloader in next file

# also saved without using the class CustomDataset

torch.save((batched_images_train, batched_labels_train, skus_list_train), "train_dataset_DRESSES_ONLY_to_class.pt")
torch.save((batched_images_test, batched_labels_test, skus_list_test), "test_dataset_DRESSES_ONLY_to_class.pt")

print("train & test set of image_tensors, labels, and skus all pickled & saved")

# image_tensors_train = torch.stack(image_tensors_train)
# labels_train = torch.tensor(labels_train, dtype=torch.long)

# image_tensors_test = torch.stack(image_tensors_test)
# labels_test = torch.tensor(labels_test, dtype=torch.long)

# print(f"Image tensor train shape: {image_tensors_train.shape}")
# print(f"Labels train shape: {labels_train.shape}")

# print(f"Image tensor train shape: {image_tensors_test.shape}")
# print(f"Labels train shape: {labels_test.shape}")

# print(f"Unique SKUs in skus_train: {len(set(skus_train))}")
# print(f"Unique SKUs in skus_test: {len(set(skus_test))}")

# torch.save(image_tensors_train, "image_tensors_train.pt") # all images including augmentations, not just dresses
# torch.save(image_tensors_test, "image_tensors_test.pt") # all images, not just dresses
# torch.save(labels_train, "labels_train.pt") # all labels
# torch.save(labels_test, "labels_test.pt") # all labels
# torch.save(skus_train, "skus_train.pt") # only dresses
# torch.save(skus_test, "skus_test.pt") # only dresses

# print("train set of image_tensors, labels, and skus all pickled & saved")