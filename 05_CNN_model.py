# trains on a subset of dresses only

# ideas from Matt post-presentation:
# develop an ensemble method with one model each processing
# 1) silhouette masked only,
# 2) 32 x 32 (or similar) squares of all images "brute force" fed into the CNN,
# 3) garment label (if available) only
# then join all three at the end of the modeling to classify

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    
def custom_collate_fn(batch): # needed to allow for varying numbers of images per SKU
    # Separate images, labels, and SKUs from the batch
    batched_images, batched_labels, batched_skus = zip(*batch)
    # Ensure images are a list of tensors with variable batch size
    batched_images = list(batched_images)
    batched_labels = torch.tensor(batched_labels, dtype=torch.long)
    batched_skus = list(batched_skus)

    return batched_images, batched_labels, batched_skus

def main():
    catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
    dresses = pd.read_csv(catalog)
    dresses = dresses[dresses['itemtype'].str.contains('dress', case=False, na=False)]
    dresses = dresses.dropna(subset=['era'])
    dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
    dresses = dresses[~dresses['era'].isin([1850, 1890, 1900, 1910, 1920, 1930, 1940, 1980, 1990, 2000])]
    all_skus = dresses['sku']

    class_counts = dresses['era'].value_counts()
    class_counts_sorted = class_counts.sort_index() # not used

    batched_images_train, batched_labels_train, skus_train = torch.load("train_dataset_DRESSES_ONLY.pt")
    batched_images_test, batched_labels_test, skus_test = torch.load("test_dataset_DRESSES_ONLY.pt")

    train_dataset = CustomDataset(batched_images_train, batched_labels_train, skus_train)
    test_dataset = CustomDataset(batched_images_test, batched_labels_test, skus_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    train_skus, test_skus = train_test_split(all_skus, test_size=0.2, random_state=42, stratify=dresses["era"])

    # a bit redundant but I'll need to subset both train/test and then reuse DataLoader, I know this is going to compound runtime :\

    images_train_subset, labels_train_subset, skus_train_all = [], [], []
    images_test_subset, labels_test_subset, skus_test_all = [], [], []

    for images, label, sku in train_dataset:
        images_train_subset.append(images)
        labels_train_subset.append(label)
        skus_train_all.append(sku)

    for images, label, sku in test_dataset:
        images_test_subset.append(images)
        labels_test_subset.append(label)
        skus_test_all.append(sku)

    filtered_train_images = []
    filtered_train_labels = []
    filtered_train_skus = []

    skus_train_all = [sku.item() if isinstance(sku, torch.Tensor) else sku for sku in skus_train_all]
    skus_test_all = [int(sku[0]) if isinstance(sku, tuple) else int(sku) for sku in skus_test_all]

    print(f"Length of test_skus: {len(test_skus)}")
    print(f"Length of skus_test_all: {len(skus_test_all)}")
    print(f"First 10 test_skus: {test_skus[:10]}")
    print(f"First 10 skus_test_all: {skus_test_all[:10]}")

    print(f"First 10 test_skus (as integers): {test_skus[:10].tolist()}")
    print(f"First 10 skus_test_all (after normalization): {skus_test_all[:10]}")

    intersection = set(skus_test_all) & set(test_skus)
    print(f"Number of matching SKUs between skus_test_all and test_skus: {len(intersection)}")


    for images, label, sku in zip(images_train_subset, labels_train_subset, skus_train_all):
        if sku in train_skus:
            filtered_train_images.append(images)
            filtered_train_labels.append(label)
            filtered_train_skus.append(sku)

    # subset test data
    filtered_test_images = []
    filtered_test_labels = []
    filtered_test_skus = []

    for images, label, sku in zip(images_test_subset, labels_test_subset, skus_test_all):
        if sku in test_skus:
            filtered_test_images.append(images)
            filtered_test_labels.append(label)
            filtered_test_skus.append(sku)
            

    train_dataset_filtered = CustomDataset(filtered_train_images, filtered_train_labels, filtered_train_skus)
    test_dataset_filtered = CustomDataset(filtered_test_images, filtered_test_labels, filtered_test_skus)

    print(f"Length of train_dataset_filtered: {len(train_dataset_filtered)}")


    train_loader = DataLoader(train_dataset_filtered, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset_filtered, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    print("TRAIN SKUS LENGTH:")
    print(len(train_skus))

    print("TEST SKUS LENGTH:")
    print(len(test_skus))

    import torch.nn as nn

    class BasicCNN(nn.Module):
        def __init__(self, num_classes):
            super(BasicCNN, self).__init__()

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

            self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
            self.bn6 = nn.BatchNorm2d(1024)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv7 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
            self.bn7 = nn.BatchNorm2d(1024)
            self.conv8 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
            self.bn8 = nn.BatchNorm2d(2048)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2) 

            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(2048, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout(p=0.3)

            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.activation(self.bn1(self.conv1(x)))
            x = self.activation(self.bn2(self.conv2(x)))
            x = self.pool(x)

            x = self.activation(self.bn3(self.conv3(x)))
            x = self.activation(self.bn4(self.conv4(x)))
            x = self.pool(x)

            x = self.activation(self.bn5(self.conv5(x)))
            x = self.activation(self.bn6(self.conv6(x)))
            x = self.pool(x)

            x = self.activation(self.bn7(self.conv7(x)))
            x = self.activation(self.bn8(self.conv8(x)))
            x = self.pool(x)

            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)

            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.activation(self.fc2(x))
            x = self.fc3(x)
            return x

    train_skus = [int(sku) for sku in train_skus]
    test_skus = [int(sku) for sku in test_skus]

    num_classes = len(dresses['era'].unique())

    model = BasicCNN(num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # was 0.0001, 0.001, 0.01 (0.1 was very bad)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    num_epochs = 20

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train()
        total_loss = 0
        total_correct = 0
        all_targets, all_predictions = [], []

        for batched_images, label, sku in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            aggregated_logits_list = []

            for images in batched_images:
                flattened_images = images.view(-1, *images.shape[2:])
                model_output = model(flattened_images)
                aggregated_logits = torch.mean(model_output, dim=0)
                aggregated_logits_list.append(aggregated_logits)

            aggregated_logits_batch = torch.stack(aggregated_logits_list)
            loss = criterion(aggregated_logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(aggregated_logits_batch, dim=1)
            total_correct += (predictions == label).sum().item()
            all_targets.append(label.item())
            all_predictions.append(predictions.item())

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        print(f"train_accuracy is total_correct {total_correct} / len(train_dataset) {len(train_loader.dataset)}")
        train_f1 = f1_score(all_targets, all_predictions, average="weighted")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_loss, total_correct = 0, 0
        all_targets, all_predictions = [], []

        with torch.no_grad():
            for batched_images, label, sku in tqdm(test_loader, desc="Testing"):
                for images, label, sku in zip(batched_images, label, sku):
                    flattened_images = images.view(-1, *images.shape[2:])  # Shape becomes [batch_size * num_images, channels, height, width]
                    model_output = model(flattened_images)
                    aggregated_logits = torch.mean(model_output, dim=0)
                    loss = criterion(aggregated_logits.unsqueeze(0), label.unsqueeze(0))

                    total_loss += loss.item()

                    prediction = torch.argmax(aggregated_logits)
                    total_correct += (prediction == label).sum().item()
                    all_targets.append(label.item())
                    all_predictions.append(prediction.item())

        test_loss = total_loss / len(test_loader)
        test_accuracy = total_correct / len(test_loader.dataset)
        print(f"test_accuracy is total_correct {total_correct} / len(test_dataset) {len(test_dataset)}")
        scheduler.step(test_loss) # gauges how to adapt lr

        print(f"all_targets: {set(all_targets)}")
        print(f"all_predictions: {set(all_predictions)}")

        print(f"all_targets: {set(all_targets)}")
        print(f"all_predictions: {set(all_predictions)}")
        print(f"Valid labels: {set(range(num_classes))}")

        # Filter out invalid predictions or targets (optional)
        valid_labels = set(range(num_classes))
        print(f"valid labels: {valid_labels}")
        all_targets = [target for target in all_targets if target in valid_labels]
        all_predictions = [pred for pred in all_predictions if pred in valid_labels]

        test_f1 = f1_score(all_targets, all_predictions, average="weighted")
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}"
        )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over 10 Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over 10 Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()