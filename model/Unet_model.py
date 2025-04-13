import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data.data_preprocessing.unet_data import BreastCancerSegmentationDataset  

# ========================== 1. Data Loading ==========================
img_dir = "/home/seanhuang/MIAI/data/dataset/img/"
mask_dir = "/home/seanhuang/MIAI/data/dataset/masks/"

dataset = BreastCancerSegmentationDataset(img_dir, mask_dir)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# ========================== 2. U-Net Model Definition ==========================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.output_layer(d1)

# ========================== 3. Training & Validation ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

# Create result folders
os.makedirs("results/sigmoid_distribution", exist_ok=True)
os.makedirs("results/predictions", exist_ok=True)

# Save visualization
def save_comparison(image, true_mask, pred_mask, filename):
    mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    image_np = image.squeeze().cpu().numpy()

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(mask_color, contours, -1, (255, 0, 0), 2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image_np, cmap="gray")
    ax[0].set_title("Original Image")

    ax[1].imshow(true_mask.squeeze().cpu().numpy(), cmap="gray")
    ax[1].set_title("Ground Truth Mask")

    ax[2].imshow(mask_color)
    ax[2].set_title("Predicted Mask with Contours")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Validation function
def validate_unet(model, val_loader, epoch):
    model.eval()
    total_loss = 0
    sigmoid_outputs = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            sigmoid_out = torch.sigmoid(logits).cpu()
            sigmoid_outputs.extend(sigmoid_out.flatten().numpy())

            pred_masks = (sigmoid_out > 0.5).float()

            for i in range(len(images)):
                save_comparison(
                    images[i], masks[i], pred_masks[i], 
                    f"results/predictions/val_epoch{epoch}_{i}.png"
                )

    plt.figure(figsize=(8, 5))
    sns.histplot(sigmoid_outputs, bins=50, kde=True)
    plt.title(f"Sigmoid Output Distribution (Epoch {epoch})")
    plt.xlabel("Sigmoid Output Value")
    plt.ylabel("Frequency")
    plt.savefig(f"results/sigmoid_distribution/epoch_{epoch}.png")
    plt.close()

    return total_loss / len(val_loader)

# Start training
def train_unet(model, train_loader, val_loader, num_epochs=50):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = validate_unet(model, val_loader, epoch)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/unet_best.pth")
            print("✅ Model saved!")

train_unet(model, train_loader, val_loader, num_epochs=200)
