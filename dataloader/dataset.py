import os
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.file_list = []

        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            files = os.listdir(class_path)
            self.file_list.extend([(cls, os.path.join(class_path, file)) for file in files])

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        class_name, img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB").resize((256, 256))

        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label