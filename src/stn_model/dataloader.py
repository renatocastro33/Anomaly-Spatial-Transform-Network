import os
import torch
from PIL import Image


class MVTEC:
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [mvtec_class  for mvtec_class in os.listdir(root_dir) if not '.' in mvtec_class]
        print(self.classes)
        
    def __len__(self):
        return sum(len(os.listdir(os.path.join(self.root_dir, cls, "train", "good"))) for cls in self.classes)

    def __getitem__(self, index):
        class_idx = 0
        for cls in self.classes:
            num_samples = len(os.listdir(os.path.join(self.root_dir, cls, "train", "good")))
            if index < num_samples:
                class_idx = cls
                break
            index -= num_samples

        img_name = os.listdir(os.path.join(self.root_dir, class_idx, "train", "good"))[index]
        img_path = os.path.join(self.root_dir, class_idx, "train", "good", img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, class_idx