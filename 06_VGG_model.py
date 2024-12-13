# trains on a subset of dresses only: VGG pre-trained model

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
    batched_images, batched_labels, batched_skus = zip(*batch)
    batched_images = list(batched_images)
    batched_labels = torch.tensor(batched_labels, dtype=torch.long)
    batched_skus = list(batched_skus)

    return batched_images, batched_labels, batched_skus

def main():
    catalog = "C:/vvcc/archive-dump-Sept-2024/compiled-catalogs/complete_catalog_cleaned_plus_synthetic.csv"

    # get only the eras listed for dresses, no Nans
    dresses = pd.read_csv(catalog)
    dresses = dresses[dresses['itemtype'].str.contains('dress', case=False, na=False)]
    dresses = dresses.dropna(subset=['era'])
    dresses = dresses[~dresses['era'].isin([1850, 1890, 1900, 1910, 1920, 1930, 1940, 1980, 1990, 2000])]
    dresses.loc[:, 'era'] = dresses['era'].apply(lambda x: int(float(x)) if pd.notna(x) else x)
    all_skus = dresses['sku']
    class_counts = dresses['era'].value_counts()
    class_counts_sorted = class_counts.sort_index()
    print(f"class_counts_sorted: {class_counts_sorted}")

    batched_images_train, batched_labels_train, skus_from_dataset = torch.load("train_dataset_DRESSES_ONLY.pt")
    batched_images_test, batched_labels_test, skus_test = torch.load("test_dataset_DRESSES_ONLY.pt")

    # get test skus only (no synthetic skus)
    skus_test_with_labels = list(zip(skus_test, batched_labels_test))
    all_test_skus = [sku for sku, label in skus_test_with_labels]
    all_test_labels = [label for sku, label in skus_test_with_labels]

    # get test_skus only
    test_skus, _, _, _ = train_test_split(
        all_test_skus, all_test_labels, test_size=0.1, stratify=all_test_labels, random_state=42
    )

    # subset skus_from_dataset to include only non-test skus (this will include synthetic skus)
    print("length of skus_from_dataset before: ", len(skus_from_dataset))
    skus_from_dataset = [sku for sku in skus_from_dataset if sku not in set(test_skus)]

    filtered_train_indices = [i for i, sku in enumerate(skus_from_dataset) if sku in skus_from_dataset]
    filtered_train_images = [batched_images_train[i] for i in filtered_train_indices]
    filtered_train_labels = [batched_labels_train[i] for i in filtered_train_indices]
    filtered_train_skus = [skus_from_dataset[i] for i in filtered_train_indices]

    print("length of skus_from_dataset after: ", len(skus_from_dataset))

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(15),          
        transforms.RandomResizedCrop(224),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    train_images, val_images, train_labels, val_labels, train_skus, val_skus = train_test_split(
        filtered_train_images, filtered_train_labels, filtered_train_skus, test_size=0.222,
        random_state=42, stratify=filtered_train_labels
    )

    filtered_test_indices = [i for i, sku in enumerate(skus_test) if sku in test_skus]
    filtered_test_images = [batched_images_test[i] for i in filtered_test_indices]
    filtered_test_labels = [batched_labels_test[i] for i in filtered_test_indices]
    filtered_test_skus = [skus_test[i] for i in filtered_test_indices]

    train_dataset_filtered = CustomDataset(train_images, train_labels, train_skus, transform=train_transforms)
    val_dataset_filtered = CustomDataset(val_images, val_labels, val_skus)
    test_dataset_filtered = CustomDataset(filtered_test_images, filtered_test_labels, filtered_test_skus)

    train_loader = DataLoader(train_dataset_filtered, batch_size=32, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset_filtered, batch_size=32, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset_filtered, batch_size=32, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    print("TRAIN SKUS LENGTH:")
    print(len(train_skus))
    print("VAL SKUS LENGTH:")
    print(len(val_skus))
    print("TEST SKUS LENGTH:")
    print(len(test_skus))

    for sku in val_skus:
        print(sku)

    train_skus = [int(sku) for sku in train_skus]
    val_skus = [int(sku.item()) for sku in val_skus]
    test_skus = [int(sku[0]) for sku in test_skus]

    num_classes = len(dresses['era'].unique())

    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.classifier[5] = nn.Dropout(p=0.5) 

    for name, param in model.features.named_parameters():
        if int(name.split(".")[0]) < 27:
            param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    num_epochs = 20

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train()
        total_loss = 0
        total_correct = 0
        all_targets, all_predictions = [], []

        for batched_images, label, sku in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            for images, label, sku in zip(batched_images, label, sku):
                flattened_images = images.view(-1, *images.shape[2:])  # Shape becomes [batch_size * num_images, channels, height, width]
                model_output = model(flattened_images)
                aggregated_logits = torch.mean(model_output, dim=0)

                loss = criterion(aggregated_logits.unsqueeze(0), label.unsqueeze(0))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = torch.argmax(aggregated_logits)
                total_correct += (predictions == label).sum().item()
                all_targets.append(label.item())
                all_predictions.append(predictions.item())

        # training metrics
        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        print(f"train_accuracy is total_correct {total_correct} / len(train_dataset) {len(train_loader.dataset)}")
        train_f1 = f1_score(all_targets, all_predictions, average="weighted")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # validation loop
        model.eval()
        total_loss, total_correct = 0, 0
        all_targets, all_predictions = [], []

        with torch.no_grad():
            for batched_images, label, sku in tqdm(val_loader, desc="Validation"):
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

        val_loss = total_loss / len(val_loader)
        val_accuracy = total_correct / len(val_loader.dataset)
        val_f1 = f1_score(all_targets, all_predictions, average="weighted")
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")

        total_loss, total_correct = 0, 0
        all_targets, all_predictions = [], []
        true_positive_counts = {0: 0, 1: 0, 2: 0}  
        total_predictions_counts = {0: 0, 1: 0, 2: 0}

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

                    if prediction.item() in true_positive_counts:
                        total_predictions_counts[prediction.item()] += 1
                        if prediction.item() == label.item():
                            true_positive_counts[prediction.item()] += 1

        for cls in [0, 1, 2]:
            total_preds = total_predictions_counts[cls]
            correct_preds = true_positive_counts[cls]
            class_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
            print(f"Class {cls} Accuracy: {class_accuracy:.4f} ({correct_preds}/{total_preds} correct)")

        test_loss = total_loss / len(test_loader)
        test_accuracy = total_correct / len(test_loader.dataset)
        test_f1 = f1_score(all_targets, all_predictions, average="weighted")
        print(f"test_accuracy is total_correct {total_correct} / len(test_skus) {len(test_skus)}")
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        scheduler.step(val_loss) # gauges how to adapt lr

        print(f"all_targets: {set(all_targets)}")
        print(f"all_predictions: {set(all_predictions)}")

        print(f"all_targets: {set(all_targets)}")
        print(f"all_predictions: {set(all_predictions)}")
        print(f"Valid labels: {set(range(num_classes))}")

        # filter out invalid predictions or targets (optional)
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

if __name__ == '__main__':
    main()