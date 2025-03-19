import os
import torch
from data.ingestion.image_preprocessor import load_image_and_mask

def load_all_images(dataset_dir="dataset"):
    """
    读取所有病变超声影像及其掩码，并转换为 PyTorch Tensor
    :param dataset_dir: 数据集根目录
    :return: images_tensor (N, 1, 224, 224), masks_tensor (N, 1, 224, 224)
    """
    img_dir = os.path.join(dataset_dir, "img")
    mask_dir = os.path.join(dataset_dir, "masks")

    # 获取所有 .png 文件
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    # 用于存储所有图像和掩码
    images = []
    masks = []

    # 遍历所有图片
    for image_name in image_files:
        try:
            img_tensor, mask_tensor = load_image_and_mask(image_name, dataset_dir)
            images.append(img_tensor)
            masks.append(mask_tensor)
            print(f" 处理完成: {image_name}")
        except FileNotFoundError as e:
            print(f" 错误: {e}")

    # 转换为 PyTorch Tensor
    if len(images) > 0:
        images_tensor = torch.stack(images)  # 形状: (N, 1, 224, 224)
        masks_tensor = torch.stack(masks)    # 形状: (N, 1, 224, 224)
        print(f" 数据加载完成！图像张量形状: {images_tensor.shape}, 掩码张量形状: {masks_tensor.shape}")
        return images_tensor, masks_tensor
    else:
        print(" 没有加载任何图像，请检查数据集是否存在！")
        return None, None

# 运行测试
if __name__ == "__main__":
    images_tensor, masks_tensor = load_all_images()
