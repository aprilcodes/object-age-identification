# trains on a subset of dresses only

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
# catalog_dresses_subset = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned_dresses.csv"
# image_dir = "C:/vvcc/archive-dump-Sept-2024/processed_photos/"
# original_image_dir = "C:/vvcc/archive-dump-Sept-2024/compiled-photos/_master_set_photos"

# image_dir_test = "C:/vvcc/archive-dump-Sept-2024/processed_photos_test/"
# image_dir_train = "C:/vvcc/archive-dump-Sept-2024/processed_photos_train/"

catalog_csv = pd.read_csv(catalog)

# get only the eras listed for dresses, no Nans
dresses = pd.read_csv(catalog)
dresses = dresses[dresses['itemtype'].str.contains('dress', case=False, na=False)]
dresses = dresses.dropna(subset=['era'])
dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
all_skus = dresses['sku']
# print(f"dresses' eras: {dresses['era'].unique()}")

# can't bring in both sets of files, too big and overwhelms memory
image_tensors_train = torch.load("image_tensors_train.pt", weights_only=True)
image_tensors_test = torch.load("image_tensors_test.pt", weights_only=True)
labels_train = torch.load("labels_train.pt", weights_only=True)
labels_test = torch.load("labels_test.pt", weights_only=True)
skus_train = torch.load("skus_train.pt", weights_only=True)
skus_test = torch.load("skus_test.pt", weights_only=True)

# train_skus = [sku for sku in labels_train] 
# test_skus = [sku for sku in labels_test]
# train_skus = dresses.loc[dresses['era'].isin(labels_train), sku_column].tolist()
# test_skus = dresses.loc[dresses['era'].isin(labels_test), sku_column].tolist()
# all_skus = dresses.loc[dresses['era'].isin(labels_train), sku_column].tolist()
train_skus, test_skus = train_test_split(all_skus, test_size=0.2, random_state=42)

print("TRAIN SKUS LENGTH:")
print(len(train_skus))
# print(train_skus[0:10])

print("TEST SKUS LENGTH:")
print(len(test_skus))
# print(test_skus[0:10])

# Remap the labels using the era_to_index dictionary
# mapped_labels = [era_to_index[int(label.item())] for label in labels]

# # Verify remapped labels (optional)
# print(f"Mapped labels (sample): {mapped_labels[:10]}")

# # Save the remapped labels back to the file
# torch.save(torch.tensor(mapped_labels, dtype=torch.long), "labels.pt")
# print("Remapped labels saved to 'labels.pt'")

# print(image_tensors.shape)

class BasicCNN(nn.Module):
    # def __init__(self, num_classes):
    #     super(BasicCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    #     self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    #     self.dropout = nn.Dropout(p=0.3) # 30% probability
    #     self.fc1 = nn.Linear(32 * 56 * 56, 128) 
    #     self.fc2 = nn.Linear(128, num_classes)

    # def forward(self, x):
    #     x = self.pool(nn.ReLU()(self.conv1(x)))
    #     x = self.pool(nn.ReLU()(self.conv2(x)))
    #     x = x.view(x.size(0), -1) 
    #     x = nn.ReLU()(self.fc1(x))
    #     x = self.dropout(x)  
    #     x = self.fc2(x)
    #     return x

    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout for regularization

    def forward(self, x):
        # First Convolutional Block
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Second Convolutional Block
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Adaptive Pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully Connected Layers
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

labels_train = [int(label) for label in labels_train]
labels_test = [int(label) for label in labels_test]
skus_train = [int(sku) for sku in skus_train]
skus_test = [int(sku) for sku in skus_test]
train_skus = [int(sku) for sku in train_skus]
test_skus = [int(sku) for sku in test_skus]

# separate out tensors & labels to train & test with no overlap between the two, based on sku:
train_dict = {
    sku: (tensor, label)
    for tensor, label, sku in zip(image_tensors_train, labels_train, skus_train)
    if sku in train_skus
}

test_dict = {
    sku: (tensor, label)
    for tensor, label, sku in zip(image_tensors_test, labels_test, skus_test)
    if sku in test_skus
}

print(f"length of train_dict: {len(train_dict)}")
print(f"length of test_dict: {len(test_dict)}")

filtered_train_tensors = []
filtered_train_labels = []
filtered_test_tensors = []
filtered_test_labels = []

for train_sku in train_skus:
    # for img_tensor, label in zip(image_tensors_train, labels_train):
    #     if label == train_sku:
    if train_sku in train_dict:
        img_tensor, label = train_dict[train_sku]
        filtered_train_tensors.append(img_tensor)
        filtered_train_labels.append(label)

for test_sku in test_skus:
    # for img_tensor, label in zip(image_tensors_test, labels_test):
    #     if label == test_sku:
    if test_sku in test_dict:
        tensor, label = test_dict[test_sku]
        filtered_test_tensors.append(img_tensor)
        filtered_test_labels.append(label)

# for img_tensor, label in zip(image_tensors_test, labels_test):
#     if label in test_skus:
#         filtered_test_tensors.append(img_tensor)
#         filtered_test_labels.append(label)
#     elif label in train_skus:
#         filtered_train_tensors.append(img_tensor)
#         filtered_train_labels.append(label)
#     print(f"Filtered train tensors: {len(filtered_train_tensors)}")
#     print(f"Filtered test tensors: {len(filtered_test_tensors)}")

print(f"Filtered train tensors: {len(filtered_train_tensors)}")
print(f"Filtered train labels: {len(filtered_train_labels)}")
print(f"Filtered test tensors: {len(filtered_test_tensors)}")
print(f"Filtered test labels: {len(filtered_test_labels)}")

# Ensure no SKU overlap between train and test
train_skus_set = set(train_skus)
test_skus_set = set(test_skus)
overlap = train_skus_set.intersection(test_skus_set)
assert not overlap, f"SKUs overlap between train and test: {overlap}"

# prepare DataLoader
train_dataset = TensorDataset(
    torch.stack(filtered_train_tensors), 
    torch.tensor(filtered_train_labels, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.stack(filtered_test_tensors), 
    torch.tensor(filtered_test_labels, dtype=torch.long)
)

# train_dataset = TensorDataset(image_tensors_train, labels_train) 
# test_dataset = TensorDataset(image_tensors_test, labels_test) 

# train_size = int(0.8 * len(train_dataset))
# test_size = len(test_dataset) - train_size

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # ensure all labels are same type and not nan
# dresses = dresses.dropna(subset=['era'])
# dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)

num_classes = len(dresses['era'].unique())
# print(f"num_classes: {num_classes}")
# print(dresses['era'].unique())

# initialize model, loss, and optimizer
model = BasicCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# lr scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

num_epochs = 20

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
    model.train()
    total_loss = 0
    total_correct = 0
    all_targets, all_predictions = [], []

    for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == targets).sum().item()
        # print(f"total_correct: {total_correct}")
        # print(f"predictions: {predictions}")
        all_targets.extend(targets.tolist())
        all_predictions.extend(predictions.tolist())

    # Metrics for training
    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / len(train_dataset)
    print(f"train_accuracy is total_correct {total_correct} / len(train_dataset) {len(train_dataset)}")
    train_f1 = f1_score(all_targets, all_predictions, average="weighted")
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation loop
    model.eval()
    total_loss, total_correct = 0, 0
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == targets).sum().item()
            all_targets.extend(targets.tolist())
            all_predictions.extend(predictions.tolist())

    test_loss = total_loss / len(test_loader)
    test_accuracy = total_correct / len(test_dataset)
    print(f"test_accuracy is total_correct {total_correct} / len(test_dataset) {len(test_dataset)}")
    scheduler.step(test_loss) # gauges how to adapt lr

    test_f1 = f1_score(all_targets, all_predictions, average="weighted")
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch {epoch + 1}/{num_epochs}: "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}"
    )

# Plotting results
plt.figure(figsize=(12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over 20 Epochs")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over 20 Epochs")
plt.legend()

plt.tight_layout()
plt.show()