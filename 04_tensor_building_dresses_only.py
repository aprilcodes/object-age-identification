

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
from imblearn.over_sampling import SMOTE
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
        self.skus = list(image_mapping.keys())  # List of SKUs

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


    # SMOTE HERE: prep the dataset for SMOTE
    X_train = []  # Placeholder for image paths (instead of flattened image data)
    y_train = []  # Labels (eras)
    sku_train = []  # SKUs corresponding to each image

    sku_to_images_train = defaultdict(list)

    image_mapping_train = {int(k): v for k, v in image_mapping_train.items()}
    print(f"All key types after image_mapping_train conversion: {set(type(k) for k in image_mapping_train.keys())}")

    print(f"dresses['sku'] dtype: {dresses['sku'].dtype}")
    dresses['sku'] = dresses['sku'].astype(int)
    print(f"dresses['sku'] dtype: {dresses['sku'].dtype}")

    # flatten image paths and associate with labels and SKUs
    for sku, image_paths in image_mapping_train.items():
        numeric_sku = int(sku)
        label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
        if label_row.empty:
            print(f"First set: No label found for SKU: {sku}, numeric_sku: {numeric_sku}")
            continue
        numeric_label = int(label_row.values[0])
        sku_to_label_train[sku] = numeric_label
        for image_path in image_paths:
            sku_to_images_train[sku].append(image_path)
            X_train.append(image_path)
            y_train.append(numeric_label)
            sku_train.append(numeric_sku)

    print(f"Number of SKUs in sku_to_images_train: {len(sku_to_images_train)}")
    print(f"Example entry: {list(sku_to_images_train.items())[:5]}")

    print(f"Number of SKUs in sku_to_label_train: {len(sku_to_label_train)}")
    missing_skus = [sku for sku in sku_to_label_train if sku not in sku_to_images_train]
    if missing_skus:
        print(f"SKUs with labels but no images: {missing_skus}")
    else:
        print("All labeled SKUs have corresponding images.")

    print(f"Keys in sku_to_label_train: {list(sku_to_label_train.keys())[:10]}")  # Show a sample
    print(f"Missing SKUs in sku_to_label_train: {[sku for sku in image_mapping_train.keys() if sku not in sku_to_label_train]}")


    print(f"Before SMOTE, class distribution: {Counter(y_train)}")

    # SMOTE function can't handle strings, must encode strings as int
    file_path_encoder = LabelEncoder()
    X_train_encoded = file_path_encoder.fit_transform(X_train).reshape(-1, 1)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled_encoded, y_train_resampled = smote.fit_resample(np.array(X_train_encoded), np.array(y_train))

    X_train_resampled = file_path_encoder.inverse_transform(X_train_resampled_encoded.flatten())

    # Generate unique integer SKUs for synthetic data starting from 56001
    start_synthetic_sku = 56001
    new_synthetic_skus = list(range(start_synthetic_sku, start_synthetic_sku + (len(y_train_resampled) - len(y_train))))

    # create image sets per SKU, allowing them to vary based on # of images
    suffix_list = ["dt1", "dt2", "dt3", "dt4", "dt5", "dt6", "dt7", "dt8", "dt9", "dt10", "dt11", "dt12"] 

    # send a copy of the new synthetic labels to the dataset
    synthetic_skus_labels = pd.DataFrame({
        'sku': new_synthetic_skus,
        'era': y_train_resampled[len(y_train):]  # labels for synthetic SKUs
    })

    for synthetic_sku in new_synthetic_skus:
        if synthetic_sku not in image_mapping_train:
            image_mapping_train[synthetic_sku] = []
            sku_image_count_train[synthetic_sku] = 0

        # synthetic_label = dresses.loc[dresses['sku'] == synthetic_sku, 'era']
        # if synthetic_label.empty:
        #     print(f"No label found for synthetic SKU: {synthetic_sku}")
        #     continue  # Skip if no label is found
        # synthetic_label = int(synthetic_label.values[0])
        synthetic_label = y_train_resampled[len(y_train):][new_synthetic_skus.index(synthetic_sku)]
        # print("synthetic label")
        # Update sku_to_label_train
        sku_to_label_train[synthetic_sku] = synthetic_label

        # create filenames for synthetic images
        for i, suffix in enumerate(suffix_list):
            filename = f"v{synthetic_sku}{suffix}.jpg"  # e.g. v56001dt1.jpg
            synthetic_file_path = os.path.join(image_dir_train, filename)

            # Ensure only synthetic files are appended
            file_sku_match = re.match(r"^v(\d+)", os.path.basename(synthetic_file_path))
            # if file_sku_match and int(file_sku_match.group(1)) == synthetic_sku:
            #     image_mapping_train[synthetic_sku].append(synthetic_file_path)
            #     sku_image_count_train[synthetic_sku] += 1
            if file_sku_match:  # Check if the filename matches the expected pattern
                # print("file_sku_match is true")
                extracted_sku = int(file_sku_match.group(1))  # Convert extracted SKU to integer
                if extracted_sku == synthetic_sku:  # Ensure the extracted SKU matches the synthetic SKU
                    # print("appending synthetic sku to image_mapping_train")
                    image_mapping_train[synthetic_sku].append(synthetic_file_path)
                    sku_image_count_train[synthetic_sku] += 1
                else:
                    print(f"Mismatch: Extracted SKU {extracted_sku}, Expected SKU {synthetic_sku}, File: {filename}")
            else:
                print(f"Filename did not match pattern: {filename}")

            # else:    
            #     print(f"Mismatched file {filename} in SKU {sku} and synthetic sku {synthetic_sku}")

    # for sku, file_paths in image_mapping_train.items():
    #     for file_path in file_paths:
    #         file_sku_match = re.match(r"^v(\d+)", os.path.basename(file_path))
    #         if file_sku_match:
    #             file_sku = int(file_sku_match.group(1))
    #             if file_sku != sku:
    #                 print(f"Mismatched file {file_path} in SKU {sku}")

    print(f"SKU type in image_mapping_train: {type(list(image_mapping_train.keys())[0])}")
    print(f"First key in image_mapping_train: {list(image_mapping_train.keys())[0]}")
    print(f"Example filename being checked: {synthetic_file_path}")

    # augment existing images for synthetic SKUs
    for synthetic_sku, image_paths in image_mapping_train.items():
        if synthetic_sku in new_synthetic_skus:  # Only process synthetic SKUs
            original_sku = random.choice(sku_train)  # Pick a random original SKU from the same dataset
            for image_path in image_paths:
                original_image_path = random.choice(image_mapping_train[original_sku])  # random original image
                with Image.open(original_image_path).convert("RGB") as img:            
                    # apply random transformations
                    img = F.rotate(img, angle=random.uniform(-5, 5))
                    img = F.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2)) 
                    img = F.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2)) 
                    img.save(image_path)

    dresses = pd.concat([dresses, synthetic_skus_labels], ignore_index=True)
    print(f"Updated dresses with synthetic SKUs. Total SKUs: {len(dresses)}")
    dresses.to_csv("C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned_plus_synthetic.csv")

    # Combine existing SKUs with synthetic SKUs
    sku_train_resampled = sku_train + new_synthetic_skus

    # Convert back to lists for easier handling
    X_train_resampled = X_train_resampled.flatten().tolist()
    y_train_resampled = y_train_resampled.tolist()

    print(f"After SMOTE, class distribution: {Counter(y_train_resampled)}")

    print(f"Debug: SKU {sku} (type: {type(sku)})")
    sku_image_count_train = {int(k): v for k, v in sku_image_count_train.items()}
    image_mapping_train = {int(k): v for k, v in image_mapping_train.items()}

    # Update `image_mapping_train` to include synthetic samples
    for i, (image_path, label, sku) in enumerate(zip(X_train_resampled, y_train_resampled, sku_train_resampled)):
        sku = int(sku)
        if sku not in image_mapping_train:
            image_mapping_train[sku] = []
            sku_image_count_train[sku] = 0
        image_mapping_train[sku].append(image_path)
        sku_image_count_train[sku] += 1

    print(f"New training set size: {len(X_train_resampled)} samples")


    # all images mapped, now make tensors
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

    # batched_image_tensors_train = []
    # batched_labels_train = []
    # batched_skus_train = []

    ######### for each in image_tensors_train
    # for sku, image_paths in image_mapping_train.items():
    #     numeric_sku = int(sku)  # Convert SKU to int once per group of images
    #     try:
    #         # Retrieve the label for the SKU
    #         label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
    #         if label_row.empty:
    #             print(f"Second set: No label found for SKU: {sku}")
    #             continue
    #         numeric_label = int(label_row.values[0])
    #         # Process and aggregate all images for the SKU
    #         sku_images = []
    #         for image_path in image_paths:
    #             img = Image.open(image_path).convert("RGB")
    #             img_tensor = transform(img)
    #             sku_images.append(img_tensor)  # Collect all images for this SKU

    #         # Append the aggregated batch, label, and SKU
    #         batched_image_tensors_train.append(torch.stack(sku_images))  # Stack all images into a tensor
    #         batched_labels_train.append(numeric_label)  # One label for the SKU
    #         batched_skus_train.append(numeric_sku)  # The SKU itself

    #     except Exception as e:
    #         print(f"Error processing SKU {sku}: {e}")
    #         continue

    print("check train set:")
    # print(f"skus train: {len(batched_skus_train)}")
    # print(f"unique skus: {len(set(batched_skus_train))}")
    # print(f"how many image tensors: {len(batched_image_tensors_train)}")
    # print(f"how many labels: {len(batched_labels_train)}")

    total_image_paths = sum(len(image_paths) for image_paths in image_mapping_train.values())
    print(f"Total image paths in image_mapping_train: {total_image_paths}")

    # batched_image_tensors_test = []
    # batched_labels_test = []
    # batched_skus_test = []

    # for sku, image_paths in image_mapping_test.items():
    #     numeric_sku = int(sku)
    #     try:
    #         label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
    #         if label_row.empty:
    #             print(f"Third set: No label found for SKU: {sku}")
    #             continue
    #         numeric_label = int(label_row.values[0])

    #         sku_images = []
    #         for image_path in image_paths:
    #             img = Image.open(image_path).convert("RGB")
    #             img_tensor = transform(img)
    #             sku_images.append(img_tensor)

    #         batched_image_tensors_test.append(torch.stack(sku_images))  # Stack all images into a tensor
    #         batched_labels_test.append(numeric_label)  # One label for the SKU
    #         batched_skus_test.append(numeric_sku)  # The SKU itself

    #     except Exception as e:
    #         print(f"Error processing SKU {sku}: {e}")
    #         continue

    # print("check test set:")
    # print(len(batched_skus_test))
    # print(len(batched_image_tensors_test))

    total_image_paths_test = sum(len(image_paths) for image_paths in image_mapping_test.values())
    print(f"Total image paths in image_mapping_test: {total_image_paths_test}")

    # sku_to_images_train = defaultdict(list)

    # for image, label, sku in zip(batched_image_tensors_train, batched_labels_train, batched_skus_train):
    #     sku_to_images_train[sku].append(image)
    #     sku_to_label_train[sku] = label

    print("check length of sku_to_images_train:")
    print(len(sku_to_images_train))

    # skus_list_train = list(sku_to_images_train.keys())
    # batched_images_train = [torch.stack(sku_to_images_train[sku]) for sku in skus_list_train]  # Stack images for each SKU
    # batched_labels_train = [sku_to_label_train[sku] for sku in skus_list_train]

    shuffled_skus = list(image_mapping_train.keys())
    random.shuffle(shuffled_skus)
    # rebuild image_mapping_train based on the shuffled SKUs

    # for sku in shuffled_skus:
    #     try:
    #         _ = image_mapping_train[sku]  # Access the key to test it
    #     except KeyError:
    #         print(f"KeyError for SKU: {sku}")
    #     else:
    #         continue

    # for sku, paths in image_mapping_train.items():
    #     for path in paths:
    #         if not path.split('/')[-1].startswith(f'v{sku}'):
    #             print(f"Mismatched file {path} in SKU {sku}")

    # missing_keys = [sku for sku in new_synthetic_skus if sku not in sku_to_label_train]
    # if missing_keys:
    #     print(f"Missing synthetic SKUs in sku_to_label_train: {missing_keys}")
    # else:
    #     print("All synthetic SKUs have corresponding labels in sku_to_label_train.")
    missing_keys = [sku for sku in image_mapping_train.keys() if sku not in sku_to_label_train]
    if missing_keys:
        print(f"Missing keys in sku_to_label_train: {missing_keys}")
    else:
        print("All SKUs in image_mapping_train have labels in sku_to_label_train.")



    shuffled_image_mapping_train = {sku: image_mapping_train[sku] for sku in shuffled_skus}
    print(f"Sample of shuffled_image_mapping_train: {list(shuffled_image_mapping_train.items())[:5]}")
    shuffled_sku_to_label_train = {sku: sku_to_label_train[sku] for sku in shuffled_skus}

    # train_dataset = CustomDataset(batched_images_train, batched_labels_train, skus_list_train)
    train_dataset = LazyCustomDataset(shuffled_image_mapping_train, shuffled_sku_to_label_train, transform)
    # train_dataset = LazyCustomDataset(image_mapping_train, sku_to_label_train, transform)

    # sku_to_images_test = defaultdict(list)
    sku_to_label_test = {}

    for sku in image_mapping_test.keys():
        numeric_sku = int(sku)  # Convert SKU to numeric format
        label_row = dresses.loc[dresses['sku'] == numeric_sku, 'era']
        if label_row.empty:
            print(f"No label found for test SKU: {sku}")
            continue
        sku_to_label_test[sku] = int(label_row.values[0])

    # for image, label, sku in zip(batched_image_tensors_test, batched_labels_test, batched_skus_test):
    #     sku_to_images_test[sku].append(image)
    #     sku_to_label_test[sku] = label

    # print("check length of sku_to_images_test:")
    # print(len(sku_to_images_test))

    # skus_list_test = list(sku_to_images_test.keys())
    # batched_images_test = [torch.stack(sku_to_images_test[sku]) for sku in skus_list_test]  # Stack images for each SKU
    # batched_labels_test = [sku_to_label_test[sku] for sku in skus_list_test]

    # test_dataset = CustomDataset(batched_images_test, batched_labels_test, skus_list_test)
    test_dataset = LazyCustomDataset(image_mapping_test, sku_to_label_test, transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    train_tensors = []  # To hold tensors for all SKUs
    train_labels = []   # To hold labels for all SKUs
    train_skus = []     # To hold SKUs for all SKUs

    # Process the train_loader
    print("Building train tensors...")
    for batched_images, label, sku in train_loader:
        train_tensors.append(batched_images)  # Each SKU's images as a tensor
        train_labels.append(label)           # Label for the SKU
        train_skus.append(sku) 

    print(f"Number of samples in train_dataset: {len(train_dataset)}")
    print(f"Number of samples in test_dataset: {len(test_dataset)}")

    torch.save((train_tensors, train_labels, train_skus), "train_dataset_DRESSES_ONLY.pt")
    print("Train dataset saved.")

    test_tensors = []  # To hold tensors for all SKUs in the test set
    test_labels = []   # To hold labels for all SKUs in the test set
    test_skus = []     # To hold SKUs for all SKUs in the test set

    print("Building test tensors...")
    for batched_images, label, sku in test_loader:
        test_tensors.append(batched_images)  # Each SKU's images as a tensor
        test_labels.append(label)           # Label for the SKU
        test_skus.append(sku)               # SKU identifier

    # Save the test tensors
    torch.save((test_tensors, test_labels, test_skus), "test_dataset_DRESSES_ONLY.pt")
    print("Test dataset saved.")
    # also saved without using the class CustomDataset

    #torch.save((batched_images_train, batched_labels_train, skus_list_train), "train_dataset_DRESSES_ONLY_to_class.pt")
    #torch.save((batched_images_test, batched_labels_test, skus_list_test), "test_dataset_DRESSES_ONLY_to_class.pt")

    print("train & test set of image_tensors, labels, and skus all pickled & saved")

if __name__ == '__main__':
    main()