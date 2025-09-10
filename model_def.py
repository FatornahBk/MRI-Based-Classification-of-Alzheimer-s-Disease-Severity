import torch
from torchvision.models import inception_v3

def build_model(num_classes: int, device: str = "cpu"):
    # ปิด aux_logits เพื่อให้ forward คืน logits เดียว (สะดวกตอน inference)
    model = inception_v3(weights=None, aux_logits=False)
    # แก้หัวให้เท่ากับจำนวนคลาส
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.to(device)
    model.eval()
    return model