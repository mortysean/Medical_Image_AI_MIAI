import torch
from torchvision import transforms
from PIL import Image
import io, base64
from model.unet import UNet

MODEL_PATH = "/home/seanhuang/MIAI/model/results/unet_best.pth"

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


def predict_mask(image_file):  # image_file 是 Django request.FILES['image']
    image = Image.open(image_file).convert("L")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        mask = (torch.sigmoid(output) > 0.5).float().squeeze()

    mask_img = transforms.ToPILImage()(mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded
