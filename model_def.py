import torch
from timm import create_model

def build_model(num_classes: int, device: str = "cpu"):
    # ไม่ใส่ argument แปลก ๆ ที่บางโมเดลไม่รองรับ (กัน error 'unexpected keyword')
    model = create_model("inception_v3", pretrained=False, num_classes=num_classes)
    model.to(device)
    return model