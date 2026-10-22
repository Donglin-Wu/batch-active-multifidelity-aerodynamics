import torch


def save_models(models, path):
    torch.save(models, path)


def load_models(path):
    return torch.load(path, map_location='cpu', weights_only=False)
