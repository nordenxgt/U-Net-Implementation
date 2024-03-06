from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import CustomDataset

def dataloader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(root_dir="{DIR}", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = CustomDataset(root_dir="{DIR}", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    return train_dataloader, test_dataloader