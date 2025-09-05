import torch

ckpt = torch.load("efficientnet_b7_checkpoint_fold1.pt", map_location="cpu")
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
torch.save(state_dict, "efficientnet_b7_state_dict.pt")