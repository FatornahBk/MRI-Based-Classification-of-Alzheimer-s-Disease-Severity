# prediction.py
from typing import List, Tuple
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

def pred_class(
    model: torch.nn.Module,
    image: Image.Image,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
):
    # เลือกอุปกรณ์อัตโนมัติ (รองรับ MPS บน Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    tfm = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        # ใช้ autocast เฉพาะบน GPU/MPS เพื่อความเร็ว
        if device.type in ("cuda", "mps"):
            with torch.autocast(device_type=device.type):
                logits = model(x)
        else:
            logits = model(x)

        probs = F.softmax(logits, dim=1)[0].to("cpu")

    top_idx = int(torch.argmax(probs))
    top_name = class_names[top_idx]
    return top_idx, top_name, probs.numpy().tolist()