# trains on a subset of dresses only

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
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

catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"

catalog_csv = pd.read_csv(catalog)

# get only the eras listed for dresses, no Nans
dresses = pd.read_csv(catalog)
dresses = dresses[dresses['itemtype'].str.contains('dress', case=False, na=False)]
dresses = dresses.dropna(subset=['era'])
dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
all_skus = dresses['sku']
class_counts = dresses['era'].value_counts()
class_counts_sorted = class_counts.sort_index()
print(f"class_counts_sorted: {class_counts_sorted}")
class_weights = 1.0 / torch.tensor(class_counts_sorted.values, dtype=torch.float)

batched_images_train, batched_labels_train, skus_train = torch.load("train_dataset_DRESSES_ONLY_to_class.pt")
batched_images_test, batched_labels_test, skus_test = torch.load("test_dataset_DRESSES_ONLY_to_class.pt")

# Recreate the CustomDataset objects
train_dataset = CustomDataset(batched_images_train, batched_labels_train, skus_train)
test_dataset = CustomDataset(batched_images_test, batched_labels_test, skus_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

train_skus, test_skus = train_test_split(all_skus, test_size=0.2, random_state=42)

# a bit redundant but I'll need to subset both train/test and then reuse DataLoader:

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

for images, label, sku in zip(images_train_subset, labels_train_subset, skus_train_all):
    if sku in train_skus:
        filtered_train_images.append(images)
        filtered_train_labels.append(label)
        filtered_train_skus.append(sku)

# Subset test data
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

train_loader = DataLoader(train_dataset_filtered, batch_size=32, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset_filtered, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

# Check the length of DataLoaders
print(f"Number of train batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")

print("TRAIN SKUS LENGTH:")
print(len(train_skus))
# print(train_skus[0:10])

print("TEST SKUS LENGTH:")
print(len(test_skus))

class BasicCNN(nn.Module):

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

train_skus = [int(sku) for sku in train_skus]
test_skus = [int(sku) for sku in test_skus]

num_classes = len(dresses['era'].unique())

# initialize model, loss, and optimizer
# model = BasicCNN(num_classes)
model = vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)
model.classifier[5] = nn.Dropout(p=0.5)  # dropout for VGG at final layer

for name, param in model.features.named_parameters():
    if "28" in name or "29" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False # this is where layers get frozen

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000001, weight_decay=1e-4) # changed for vgg model, weight_decay is L2 regularization

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

    # predictions_by_sku_train = {} # tracks each image's prediction per sku

    for batched_images, label, sku in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        print(f"Batch size (SKUs): {len(batched_images)}")
        
        for images, label, sku in zip(batched_images, label, sku):
            flattened_images = images.view(-1, *images.shape[2:])  # Shape becomes [batch_size * num_images, channels, height, width]
            model_output = model(flattened_images)
            aggregated_logits = torch.mean(model_output, dim=0)

            loss = criterion(aggregated_logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            # predictions_by_sku_train[sku] = {
            # "images": images,
            # "predictions": torch.argmax(aggregated_logits).item(),
            # "label": label.item()
            # }

            total_loss += loss.item()
            predictions = torch.argmax(aggregated_logits)
            total_correct += (predictions == label).sum().item()
            all_targets.append(label.item())
            all_predictions.append(predictions.item())

    # Metrics for training
    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / len(train_loader.dataset)
    print(f"train_accuracy is total_correct {total_correct} / len(train_dataset) {len(train_loader.dataset)}")
    train_f1 = f1_score(all_targets, all_predictions, average="weighted")
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation loop
    model.eval()
    total_loss, total_correct = 0, 0
    all_targets, all_predictions = [], []

    # predictions_by_sku_test = {}
    true_positive_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}  # Tracks correct predictions per class
    total_predictions_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}  # Tracks total predictions made per class

    with torch.no_grad():
        for batched_images, label, sku in tqdm(test_loader, desc="Testing"):
            for images, label, sku in zip(batched_images, label, sku):
                flattened_images = images.view(-1, *images.shape[2:])
                model_output = model(flattened_images)
                aggregated_logits = torch.mean(model_output, dim=0)

                loss = criterion(aggregated_logits.unsqueeze(0), label.unsqueeze(0))
                total_loss += loss.item()

                prediction = torch.argmax(aggregated_logits)
                total_correct += (prediction == label).sum().item()
                all_targets.append(label.item())
                all_predictions.append(prediction.item())

                    # Update per-class stats
                if prediction.item() in true_positive_counts:
                    total_predictions_counts[prediction.item()] += 1
                    if prediction.item() == label.item():
                        true_positive_counts[prediction.item()] += 1

    # Calculate per-class accuracy
    for cls in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        total_preds = total_predictions_counts[cls]
        correct_preds = true_positive_counts[cls]
        class_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
        print(f"Class {cls} Accuracy: {class_accuracy:.4f} ({correct_preds}/{total_preds} correct)")

    test_loss = total_loss / len(test_loader)
    test_accuracy = total_correct / len(test_loader.dataset)
    print(f"test_accuracy is total_correct {total_correct} / len(test_dataset) {len(test_dataset)}")
    scheduler.step(test_loss) # gauges how to adapt lr

    # if len(test_losses) > 5 and test_loss > min(test_losses[-5:]):        
    #     print("Early stopping triggered: 5 epochs with no improvement in test loss")
    #     break

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

# confusion matrix
cm = confusion_matrix(all_targets, all_predictions, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix Over 20 Epochs")
plt.show()

# precision, recall, f1
report = classification_report(all_targets, all_predictions, labels=list(range(num_classes)), target_names=[str(cls) for cls in range(num_classes)])
print(report)

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