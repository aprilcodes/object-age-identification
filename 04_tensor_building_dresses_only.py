
# performs custom version of SMOTE with >1 image per label
# saves a set of tensors for train (with synthetic SKUs and images)
# saves a set of tensors for test (no synthetic SKUs) 

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm

class LazyCustomDataset(Dataset):
    def __init__(self, image_mapping, labels_mapping, transform=None):
        """
        Initializes a dataset that dynamically loads and processes images.
        
        Args:
            image_mapping (dict): Mapping of SKUs to lists of image file paths.
            labels_mapping (dict): Mapping of SKUs to their corresponding labels.
            transform (callable, optional): Transformation function for preprocessing images.
        """
        self.image_mapping = image_mapping
        self.labels_mapping = labels_mapping
        self.transform = transform
        self.skus = list(image_mapping.keys()) 

    def __len__(self):
        return len(self.skus)  # Number of unique SKUs

    def __getitem__(self, idx):
        """
        Dynamically loads and processes images for a given SKU.

        Args:
            idx (int): Index of the SKU in the dataset.

        Returns:
            torch.Tensor: Batched tensor of images for the SKU.
            int: Label corresponding to the SKU.
            str: SKU identifier.
        """
        sku = self.skus[idx]  # Get the SKU at the specified index
        image_paths = self.image_mapping[sku]  # List of image paths for the SKU
        label = self.labels_mapping[sku]  # Label for the SKU

        sku_images = []  # List to store processed images
        for image_path in image_paths:
            with Image.open(image_path).convert("RGB") as img:
                if self.transform:
                    img = self.transform(img)  # Apply transformations
                sku_images.append(img)

        # Stack all images for the SKU into a single tensor
        batched_images = torch.stack(sku_images)

        return batched_images, label, sku


def main():
    catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
    image_dir_test = "C:/vvcc/archive-dump-Sept-2024/processed_photos_test/"
    image_dir_train = "C:/vvcc/archive-dump-Sept-2024/processed_photos_train/"

    sku_pattern = r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$"

    catalog_csv = pd.read_csv(catalog)
    dresses = catalog_csv[catalog_csv['itemtype'].str.contains('dress', case=False, na=False)]
    dresses = dresses.dropna(subset=['era'])
    dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
    dresses = dresses[~dresses['era'].isin([1850, 1890, 1900, 1910, 1920, 1930, 1940, 1980, 1990, 2000])] # remove severely underrepresented classes
    dress_skus = dresses['sku'].tolist()
    dress_skus = [str(sku) for sku in dress_skus]
    print(dresses['era'].value_counts())

    image_mapping_test = {}
    image_mapping_train = {}

    sku_image_count_train = {}
    sku_to_label_train = {}

    for filename in os.listdir(image_dir_test):
        sku_match = re.match(r"^(v\d{3,5})(?:[a-zA-Z0-9]{3})?\.jpg$", filename)
        if sku_match: # for every file in image_dir
            sku = sku_match.group(1) # capture the sku
            sku = sku.lstrip('v')
            sku = str(sku)

            if sku in dress_skus:
                if sku not in image_mapping_test:
                    image_mapping_test[sku] = []
                image_mapping_test[sku].append(os.path.join(image_dir_test, filename))

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
    print("original length of image_mapping_train: ", len(image_mapping_train))


    # no longer being used: prep the dataset for SMOTE
    X_train = []  # Placeholder for image paths (instead of flattened image data)
    y_train = []  # Labels (eras)
    sku_train = []  # SKUs corresponding to each image

    sku_to_images_train = defaultdict(list)

    image_mapping_train = {int(k): v for k, v in image_mapping_train.items()}
    # print(f"All key types after image_mapping_train conversion: {set(type(k) for k in image_mapping_train.keys())}")

    dresses['sku'] = dresses['sku'].astype(int)

    # flatten image paths and associate with labels and SKUs
    for sku, image_paths in image_mapping_train.items():
        numeric_sku = int(sku)
        label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
        if label_row.empty:
            print(f"First set: No label found for SKU: {sku}, numeric_sku: {numeric_sku}")
            continue
        numeric_label = int(label_row.values[0]) # is it class 0, 1, or 2
        sku_to_label_train[sku] = numeric_label
        for image_path in image_paths:
            sku_to_images_train[sku].append(image_path)
            X_train.append(image_path)
            y_train.append(numeric_label)
            sku_train.append(numeric_sku)

    # Group SKUs by their class
    skus_by_class = defaultdict(list)
    for sku, label in sku_to_label_train.items():
        skus_by_class[label].append(sku)

    # Check current distribution of SKUs per class
    sku_class_counts = {label: len(skus) for label, skus in skus_by_class.items()}
    print(f"Current SKU distribution per class: {sku_class_counts}")
    max_skus_per_class = max(sku_class_counts.values())
    skus_to_add = {label: max_skus_per_class - count for label, count in sku_class_counts.items()}
    print(f"Number of SKUs to add per class: {skus_to_add}")

    print(f"Number of SKUs in sku_to_images_train: {len(sku_to_images_train)}")
    # print(f"Example entry of sku_to_label_train: {list(sku_to_label_train.items())[:5]}")

    print(f"Number of SKUs in sku_to_label_train: {len(sku_to_label_train)}")
    missing_skus = [sku for sku in sku_to_label_train if sku not in sku_to_images_train]
    if missing_skus:
        print(f"SKUs with labels but no images: {missing_skus}")
    else:
        print("All labeled SKUs have corresponding images.")

    print(f"Keys in sku_to_label_train: {list(sku_to_label_train.keys())[:10]}")  # Show a sample
    print(f"Missing SKUs in sku_to_label_train: {[sku for sku in image_mapping_train.keys() if sku not in sku_to_label_train]}")


    print(f"Before custom SMOTE, class distribution (# of IMAGES per class): {Counter(y_train)}")

    # SMOTE function can't handle strings, must encode strings as int
    # file_path_encoder = LabelEncoder()
    # X_train_encoded = file_path_encoder.fit_transform(X_train).reshape(-1, 1)

    # Apply SMOTE
    # smote = SMOTE(random_state=42)
    # X_train_resampled_encoded, y_train_resampled = smote.fit_resample(np.array(X_train_encoded), np.array(y_train))

    # X_train_resampled = file_path_encoder.inverse_transform(X_train_resampled_encoded.flatten())

    # Generate unique integer SKUs for synthetic data starting from 56001
    start_synthetic_sku = 56001
    max_skus = max(sku_class_counts.values())
    skus_to_create = {label: max_skus - count for label, count in sku_class_counts.items()}
    total_skus_to_create = sum(skus_to_create[label] for label in skus_to_create if skus_to_create[label] > 0)

    new_synthetic_skus = list(range(start_synthetic_sku, start_synthetic_sku + total_skus_to_create))

    synthetic_skus_by_class = {}
    current_index = 0
    for label, count in skus_to_create.items():
        if count > 0:
            synthetic_skus_by_class[label] = new_synthetic_skus[current_index:current_index + count]
            current_index += count

    # create image sets per SKU, allowing them to vary based on # of images
    suffix_list = ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8", "dt9", "dt10", "dt11", "dt12"] 

    synthetic_to_original_mapping = {}

    # augment existing images for synthetic SKUs
    for synthetic_sku in new_synthetic_skus:
        if synthetic_sku not in image_mapping_train:
            image_mapping_train[synthetic_sku] = []
            sku_image_count_train[synthetic_sku] = 0

        if synthetic_sku in new_synthetic_skus:  # Only process synthetic SKUs
            original_sku = random.choice(sku_train)  # Pick a random original SKU from the same dataset
            synthetic_to_original_mapping[synthetic_sku] = original_sku
            num_images_original_sku = len(image_mapping_train[original_sku])
            
        synthetic_label = sku_to_label_train[original_sku]
        sku_to_label_train[synthetic_sku] = synthetic_label

        for i in range(num_images_original_sku):
            suffix = suffix_list[i % len(suffix_list)]  
            filename = f"v{synthetic_sku}{suffix}.jpg"  # e.g. v56001dt1.jpg
            synthetic_file_path = os.path.join(image_dir_train, filename)

            # Ensure only synthetic files are appended
            file_sku_match = re.match(r"^v(\d+)", os.path.basename(synthetic_file_path))
            if file_sku_match:  # Check if the filename matches the expected pattern
                extracted_sku = int(file_sku_match.group(1))  # Convert extracted SKU to integer
                if extracted_sku == synthetic_sku:  # Ensure the extracted SKU matches the synthetic SKU
                    image_mapping_train[synthetic_sku].append(synthetic_file_path)
                    sku_image_count_train[synthetic_sku] += 1
                else:
                    print(f"Mismatch: Extracted SKU {extracted_sku}, Expected SKU {synthetic_sku}, File: {filename}")
            else:
                print(f"Filename did not match pattern: {filename}")

    print(f"SKU type in image_mapping_train: {type(list(image_mapping_train.keys())[0])}")
    print(f"First key in image_mapping_train: {list(image_mapping_train.keys())[0]}")
    print(f"Example filename being checked: {synthetic_file_path}")
    
    # debugging why sku count doesn't increase after adding synthetic skus
    print(f"Length of image_mapping_train BEFORE synthetic additions: {len(image_mapping_train)}")
    original_sku_count = sum(1 for sku in image_mapping_train if int(sku) <= 56000)
    print(f"Original SKUs: {original_sku_count}")
    synthetic_sku_count = sum(1 for sku in image_mapping_train if int(sku) > 56000)
    print(f"Synthetic SKUs: {synthetic_sku_count}")

    # augment existing images for synthetic SKUs
    #for synthetic_sku, image_paths in image_mapping_train.items():
    for synthetic_sku in new_synthetic_skus: 
        #if synthetic_sku in new_synthetic_skus: 
        original_sku = synthetic_to_original_mapping[synthetic_sku]

        # place synthetic skus into the image set for training
        if synthetic_sku not in image_mapping_train:
            image_mapping_train[synthetic_sku] = []

        #for original_image_path, synthetic_file_path in zip(image_mapping_train[original_sku], image_mapping_train[synthetic_sku]):
        for original_image_path in image_mapping_train[original_sku]:
            synthetic_file_path = f"synthetic_{synthetic_sku}_{len(image_mapping_train[synthetic_sku])}.jpg"
            with Image.open(original_image_path).convert("RGB") as img:            
                # apply random transformations
                img = F.rotate(img, angle=random.uniform(-5, 5))
                img = F.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2)) 
                img = F.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2)) 
                img.save(synthetic_file_path)
                image_mapping_train[synthetic_sku].append(synthetic_file_path)
    
    print(f"Length of image_mapping_train AFTER synthetic additions: {len(image_mapping_train)}")
    original_sku_count = sum(1 for sku in image_mapping_train if int(sku) <= 56000)
    print(f"Original SKUs: {original_sku_count}")
    synthetic_sku_count = sum(1 for sku in image_mapping_train if int(sku) > 56000)
    print(f"Synthetic SKUs: {synthetic_sku_count}")


    # send a copy of the new synthetic labels to the dataset
    synthetic_skus_labels = pd.DataFrame({
        'sku': new_synthetic_skus,
        'era': [dresses.loc[dresses['sku'] == original_sku, 'era'].values[0] for original_sku in synthetic_to_original_mapping.values()]
        # 'era': y_train_resampled[len(y_train):]  # labels for synthetic SKUs
    })

    dresses = pd.concat([dresses, synthetic_skus_labels], ignore_index=True)
    print(f"Updated dresses with synthetic SKUs. Total SKUs: {len(dresses)}")
    dresses.to_csv("C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned_plus_synthetic.csv")

    print(f"Debug: SKU {sku} (type: {type(sku)})")
    sku_image_count_train = {int(k): v for k, v in sku_image_count_train.items()}
    image_mapping_train = {int(k): v for k, v in image_mapping_train.items()}

    # all images mapped, now make tensors
    transform = transforms.Compose([
        transforms.ToTensor(),         # convert PIL image to PyTorch tensor, scaling happens automatically in this step
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalize for RGB channels (these values are standard for pre-trained if using later)
                            std=[0.229, 0.224, 0.225])
    ])

    total_image_paths = sum(len(image_paths) for image_paths in image_mapping_train.values())
    print(f"Total image paths in image_mapping_train: {total_image_paths}")

    total_image_paths_test = sum(len(image_paths) for image_paths in image_mapping_test.values())
    print(f"Total image paths in image_mapping_test: {total_image_paths_test}")

    print("check length of sku_to_images_train:")
    print(len(sku_to_images_train))

    shuffled_skus = list(image_mapping_train.keys())
    random.shuffle(shuffled_skus)

    missing_keys = [sku for sku in image_mapping_train.keys() if sku not in sku_to_label_train]
    if missing_keys:
        print(f"Missing keys in sku_to_label_train: {missing_keys}")
    else:
        print("All SKUs in image_mapping_train have labels in sku_to_label_train.")

    shuffled_image_mapping_train = {sku: image_mapping_train[sku] for sku in shuffled_skus}
    print(f"Sample of shuffled_image_mapping_train: {list(shuffled_image_mapping_train.items())[:5]}")
    shuffled_sku_to_label_train = {sku: sku_to_label_train[sku] for sku in shuffled_skus}

    print(f"FINAL number of SKUs in image_mapping_train: {len(image_mapping_train)}")
    print(f"FINAL number of SKUs in sku_to_label_train: {len(sku_to_label_train)}")

    # make targets a number series, not the actual eras
    label_to_index = {label: idx for idx, label in enumerate(sorted(dresses['era'].unique()))}

    # Re-encode labels
    shuffled_sku_to_label_train = {
        sku: label_to_index[label] for sku, label in shuffled_sku_to_label_train.items()
    }

    print("Unique values in shuffled_sku_to_label_train (after encoding):", set(shuffled_sku_to_label_train))

    train_dataset = LazyCustomDataset(shuffled_image_mapping_train, shuffled_sku_to_label_train, transform)

    sku_to_label_test = {}

    for sku in image_mapping_test.keys():
        numeric_sku = int(sku)  # Convert SKU to numeric format
        label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
        if label_row.empty:
            print(f"No label found for test SKU: {sku}")
            continue
        sku_to_label_test[sku] = int(label_row.values[0])

    print("length of image mapping train", len(image_mapping_train))
    print("length of image mapping test", len(image_mapping_test))

    print("length of sku to label test", len(sku_to_label_test))

    sku_to_label_test = {
        sku: label_to_index[label] for sku, label in sku_to_label_test.items()
    }

    test_dataset = LazyCustomDataset(image_mapping_test, sku_to_label_test, transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    train_tensors = []  
    train_labels = []   
    train_skus = []     

    print("Building train tensors...")
    for batched_images, label, sku in train_loader:
        train_tensors.append(batched_images)  # each SKU's images as a tensor
        train_labels.append(label)           # label for the SKU
        train_skus.append(sku) 

    print(f"Number of samples in train_dataset: {len(train_dataset)}")
    print(f"Number of samples in test_dataset: {len(test_dataset)}")

    torch.save((train_tensors, train_labels, train_skus), "train_dataset_DRESSES_ONLY.pt")
    print("Train dataset saved.")

    test_tensors = [] 
    test_labels = []   
    test_skus = []     

    print("Building test tensors...")
    for batched_images, label, sku in test_loader:
        test_tensors.append(batched_images)  
        test_labels.append(label)           
        test_skus.append(sku)               

    # Save the test tensors
    torch.save((test_tensors, test_labels, test_skus), "test_dataset_DRESSES_ONLY.pt")
    print("Test dataset saved.")
    print("train & test set of image_tensors, labels, and skus all pickled & saved")

if __name__ == '__main__':
    main()