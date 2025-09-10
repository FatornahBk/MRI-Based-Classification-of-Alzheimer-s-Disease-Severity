import torch
from torchvision.models import inception_v3

def build_model(num_classes: int, device: str = "cpu"):
    model = inception_v3(weights=None, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.to(device)
    return model