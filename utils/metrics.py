import torch

def calculate_accuracy(predicted: torch.Tensor, actual: torch.Tensor) -> int:
    return torch.eq(predicted.argmax(dim=1), actual).sum().item() / len(predicted)