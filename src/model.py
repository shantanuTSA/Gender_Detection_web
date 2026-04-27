import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path, device):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model