import cv2
import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 数据增强
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

def load_image_and_mask(image_name, dataset_dir="dataset", img_size=(224, 224)):
    """
    读取病变超声影像及其掩码，并进行预处理
    :param image_name: 图像文件名 (如 "lesion_1.png")
    :param dataset_dir: 数据集根目录
    :param img_size: 目标尺寸
    :return: 预处理后的 PyTorch Tensor 图像和掩码
    """
    img_path = os.path.join(dataset_dir, "img", image_name)
    mask_path = os.path.join(dataset_dir, "masks", image_name.replace(".png", "_mask.png"))

    # 确保文件存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f" 图像文件 {img_path} 不存在！")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f" 掩码文件 {mask_path} 不存在！")

    # 读取灰度图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=-1)  # 增加通道维度

    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size)
    mask = np.expand_dims(mask, axis=-1)  # 增加通道维度

    # 数据增强
    augmented = transform(image=img, mask=mask)

    return augmented["image"], augmented["mask"]

