import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Add tqdm for progress bar
from data.data_preprocessing.breast_cancer_data import BreastCancerDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
img_dir = "/home/seanhuang/MIAI/data/dataset/img/"
mask_dir = "/home/seanhuang/MIAI/data/dataset/masks/"

# Load dataset
dataset = BreastCancerDataset(img_dir, mask_dir, transform=None)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# Define ResNet-based classifier with Dropout
class BreastCancerClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BreastCancerClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout to reduce overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
model = BreastCancerClassifier().to(device)

# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduce learning rate

# Lists for storing training and validation loss
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training function with tqdm progress bar
def train_classifier(model, train_loader, val_loader, num_epochs=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")  # Add tqdm progress bar

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, val_loader)

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "/home/seanhuang/MIAI/model/breast_cancer/best_model.pth")
            print("✅ Model saved!")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    return total_loss / len(val_loader), val_acc

# Train model
train_classifier(model, train_loader, val_loader, num_epochs=20)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid()
plt.savefig("/home/seanhuang/MIAI/model/breast_cancer/loss_curve.png")
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy Curve")
plt.legend()
plt.grid()
plt.savefig("/home/seanhuang/MIAI/model/breast_cancer/accuracy_curve.png")
plt.show()
