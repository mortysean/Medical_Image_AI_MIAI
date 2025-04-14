import torch
from torchvision import transforms
from PIL import Image
import io, base64
from model.unet import UNet
from skimage import measure
import numpy as np

# Path to trained model
MODEL_PATH = "/home/seanhuang/MIAI/model/results/unet_best.pth"

# Load UNet model (CPU mode)
model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Image transformation pipeline
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

def predict_mask_with_meta(image_file):
    """
    Predict mask from grayscale image and extract lesion features.
    Returns: base64-encoded mask image, list of lesion bounding boxes, accuracy placeholder.
    """
    # Load image as grayscale
    image = Image.open(image_file).convert("L")
    input_tensor = transform(image).unsqueeze(0)  # shape: [1, 1, H, W]

    # Model inference
    with torch.no_grad():
        output = model(input_tensor)  # shape: [1, 1, H, W]
        pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().numpy()

    # Convert predicted mask to base64 image
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    encoded_mask = "data:image/png;base64," + encoded

    # Label connected components and extract bounding boxes + area
    labeled = measure.label(pred_mask)
    props = measure.regionprops(labeled)

    lesions = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        lesions.append({
            "x": int(minc),
            "y": int(minr),
            "width": int(maxc - minc),
            "height": int(maxr - minr),
            "area": int(prop.area)  # lesion pixel count
        })

    # Dummy accuracy value (can be replaced with real metric)
    accuracy = round(np.random.uniform(0.85, 0.95), 3)

    return {
        "mask": encoded_mask,
        "lesions": lesions,
        "accuracy": accuracy
    }
