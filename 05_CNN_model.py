import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned.csv"
# catalog_dresses_subset = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned_dresses.csv"
image_dir = "C:/vvcc/archive-dump-Sept-2024/processed_photos/"

catalog_csv = pd.read_csv(catalog)

# get only the eras listed for dresses, no Nans
dresses = pd.read_csv(catalog)
dresses = dresses[dresses['itemtype'].str.contains('dress', case=False, na=False)]
dresses = dresses.dropna(subset=['era'])
dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
# print(f"dresses' eras: {dresses['era'].unique()}")

image_tensors = torch.load("image_tensors.pt", weights_only=True)
labels = torch.load("labels.pt", weights_only=True)

# Remap the labels using the era_to_index dictionary
# mapped_labels = [era_to_index[int(label.item())] for label in labels]

# # Verify remapped labels (optional)
# print(f"Mapped labels (sample): {mapped_labels[:10]}")

# # Save the remapped labels back to the file
# torch.save(torch.tensor(mapped_labels, dtype=torch.long), "labels.pt")
# print("Remapped labels saved to 'labels.pt'")

# print(image_tensors.shape)

class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# prepare DataLoader
dataset = TensorDataset(image_tensors, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # ensure all labels are same type and not nan
# dresses = dresses.dropna(subset=['era'])
# dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)

num_classes = len(dresses['era'].unique())
# print(f"num_classes: {num_classes}")
# print(dresses['era'].unique())

# initialize model, loss, and optimizer
model = BasicCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
    model.train()
    total_loss = 0
    total_correct = 0
    all_targets, all_predictions = [], []

    for images, targets in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
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

    # Metrics for testing
    test_loss = total_loss / len(test_loader)
    test_accuracy = total_correct / len(test_dataset)
    print(f"test_accuracy is total_correct {total_correct} / len(test_dataset) {len(test_dataset)}")
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