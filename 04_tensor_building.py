# transforms photos to tensors
# two sets of photos: the ones with augmentation are saved as train
# those without augmentation are saved as test

import os
import re
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, skus, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.skus = skus
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        label = self.labels[idx]
        sku = self.skus[idx]
        # image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, sku

catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
# image_dir = "C:/vvcc/archive-dump-Sept-2024/processed_photos/"
image_dir_test = "C:/vvcc/archive-dump-Sept-2024/processed_photos_test/"
image_dir_train = "C:/vvcc/archive-dump-Sept-2024/processed_photos_train/"

sku_pattern = r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$"

catalog_csv = pd.read_csv(catalog)
all_skus = catalog_csv['sku'].tolist()
all_skus = [str(sku) for sku in all_skus]

# catalog_csv['sku'] has some erroneous big values, remove them
sorted_catalog = catalog_csv.sort_values(by='sku', ascending=False)
catalog_csv = catalog_csv[catalog_csv['sku'] <= 56000]
# print(catalog_csv['sku'].max())

catalog_csv = catalog_csv.dropna(subset=['era'])
# print(f"Rows beforehand: {len(catalog_csv)}")
catalog_csv['era'] = catalog_csv['era'].astype(str) # once I added these 4 lines, no matches
alphanumeric_mask = catalog_csv['era'].str.match(r'^[a-zA-Z0-9.\s]+$', na=False)
catalog_csv = catalog_csv[alphanumeric_mask]
# print(f"after mask: {len(catalog_csv)}")
catalog_csv['era'] = catalog_csv['era'].astype(float)
catalog_csv.loc[:, 'era'] = catalog_csv['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
# print(f"Remaining rows: {len(catalog_csv)}")
catalog_csv['era'] = catalog_csv['era'].astype(int)
# print(catalog_csv['era'].dtype)
# print(catalog_csv['era'].unique())

unique_eras = sorted(catalog_csv['era'].unique())
era_to_index = {era: idx for idx, era in enumerate(unique_eras)}
index_to_era = {idx: era for era, idx in era_to_index.items()}
catalog_csv['era'] = catalog_csv['era'].map(era_to_index)
print("Mapping Era To Index:", era_to_index)
# print(catalog_csv['era'].unique())

image_mapping_test = {}
image_mapping_train = {}

# sku_image_count_test = {}
# sku_image_count_train = {}

for filename in os.listdir(image_dir_test):
    # get the SKU from the filename
    sku_match = re.match(r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$", filename)
    if sku_match: # for every file in image_dir
        sku = sku_match.group(1) # capture the sku
        sku = sku.lstrip('v')
        sku = str(sku)

        if sku in all_skus:
            if sku not in image_mapping_test:
                image_mapping_test[sku] = []
                # sku_image_count_test[sku] = 0
            image_mapping_test[sku].append(os.path.join(image_dir_test, filename))
            # sku_image_count_test[sku] += 1

for filename in os.listdir(image_dir_train):
    # get the SKU from the filename
    sku_match = re.match(r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$", filename)
    if sku_match: # for every file in image_dir
        sku = sku_match.group(1) # capture the sku
        sku = sku.lstrip('v')
        sku = str(sku)

        if sku in all_skus:
            if sku not in image_mapping_train:
                image_mapping_train[sku] = []
                # sku_image_count_train[sku] = 0
            image_mapping_train[sku].append(os.path.join(image_dir_train, filename))
            # sku_image_count_train[sku] += 1

# total_images = sum(sku_image_count_train.values())
# total_skus = len(sku_image_count_train)
# average_images_per_sku = total_images / total_skus if total_skus > 0 else 0

# print(f"Total SKUs: {total_skus}") # 28903
# print(f"Total Images: {total_images}")
# print(f"Average Images per SKU: {average_images_per_sku:.2f}")

# all images mapped, now build tensors
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

# print(f"image mapping: {image_mapping.items()}")

catalog_csv['sku'] = catalog_csv['sku'].astype(str)
catalog_csv['sku'] = catalog_csv['sku'].str.strip()
catalog_csv['sku'] = catalog_csv['sku'].astype(int)
sku = int(str(sku).strip())  # Ensure the loop's SKU is also cleaned
# print(catalog_csv['sku'].dtype)

######### for each in image_tensors_train
for sku, image_paths in image_mapping_train.items():
    for image_path in image_paths:
        try:
            numeric_sku = int(sku) # needs to be int to compare to catalog_csv['sku']
            # label_row = catalog_csv.loc[catalog_csv['sku'] == numeric_sku, 'era']
            # if label_row.empty: # there are about 70 images that don't have a value in labels, filter them out
            #     print(f"No label found for SKU: {sku}")
            #     continue
            label = catalog_csv.loc[catalog_csv['sku'] == numeric_sku, 'era'].values[0]
            numeric_label = int(label)
            labels_train.append(numeric_label)
            # if sku not in skus_train:
            #     skus_train.append(sku)
            skus_train = list(image_mapping_train.keys())
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img)
            image_tensors_train.append(img_tensor)
            # print(f"Processed image: {image_path} for SKU: {sku}")
        except Exception as e:
            continue
            # print(f"Error processing {image_path}: {e}")

print("check train set:")
print(f"skus train: {len(skus_train)}")
print(f"unique skus: {len(set(skus_train))}")
print(f"how many image tensors: {len(image_tensors_train)}")
print(f"how many labels: {len(labels_train)}")

total_image_paths = sum(len(image_paths) for image_paths in image_mapping_train.values())
print(f"Total image paths in image_mapping_train: {total_image_paths}")



for sku, image_paths in image_mapping_test.items():
    for image_path in image_paths:
        try:
            numeric_sku = int(sku) # needs to be int to compare to catalog_csv['sku']
            # label_row = catalog_csv.loc[catalog_csv['sku'] == numeric_sku, 'era']
            # if label_row.empty: # there are about 70 images that don't have a value in labels, filter them out
            #     print(f"No label found for SKU: {sku}")
            #     continue
            label = catalog_csv.loc[catalog_csv['sku'] == numeric_sku, 'era'].values[0]
            numeric_label = int(label)
            labels_test.append(numeric_label)
            #if sku not in skus_test:
            #    skus_test.append(sku)
            skus_test = list(image_mapping_test.keys())
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img)
            image_tensors_test.append(img_tensor)
            # print(f"Processed image: {image_path} for SKU: {sku}")
        except Exception as e:
            continue

print("check test set:")
print(len(skus_test))
print(len(image_tensors_test))

total_image_paths_test = sum(len(image_paths) for image_paths in image_mapping_test.values())
print(f"Total image paths in image_mapping_test: {total_image_paths_test}")

train_dataset = CustomDataset(image_tensors_train, labels_train, skus_train)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

# stack images into a tensor and convert labels to Tensor
# image_tensors_train = torch.stack(image_tensors_train)
# labels_train = torch.tensor(labels_train, dtype=torch.long)

test_dataset = CustomDataset(image_tensors_test, labels_test, skus_test)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# image_tensors_test = torch.stack(image_tensors_test)
# labels_test = torch.tensor(labels_test, dtype=torch.long)

# print(f"Image tensor train shape: {image_tensors_train.shape}")
# print(f"Labels train shape: {labels_train.shape}")

# print(f"Image tensor train shape: {image_tensors_test.shape}")
# print(f"Labels train shape: {labels_test.shape}")

# print(f"Unique SKUs in skus_train: {len(set(skus_train))}")
# print(f"Unique SKUs in skus_test: {len(set(skus_test))}")

print(f"Number of samples in train_dataset: {len(train_dataset)}")
print(f"Number of samples in test_dataset: {len(test_dataset)}")

torch.save(train_dataset, "train_dataset_ALL.pt")
torch.save(test_dataset, "test_dataset_ALL.pt") # must be loaded with dataloader in next file
# torch.save(image_tensors_train, "image_tensors_train_ALL.pt")
# torch.save(image_tensors_test, "image_tensors_test_ALL.pt")
# torch.save(labels_train, "labels_train_ALL.pt")
# torch.save(labels_test, "labels_test_ALL.pt")
# torch.save(skus_train, "skus_train_ALL.pt")
# torch.save(skus_test, "skus_test_ALL.pt")

print("train & test set of image_tensors, labels, and skus all pickled & saved")