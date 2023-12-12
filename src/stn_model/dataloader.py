import cv2
import os
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as TF
import random

import torch
import torchvision.transforms as transforms
from natsort import natsorted

from .transformations import *


class MVTEC(Dataset):
    def __init__(self, root_dir, transform=None,device='cuda'):
      self.device = device
      self.root_dir = root_dir
      self.transform = transform
      self.classes = [mvtec_class for mvtec_class in os.listdir(root_dir) if not '.' in mvtec_class]
      self.data = []  # Collect all the data upon initialization
      
      self.classes = natsorted(self.classes)

      self.class_to_idx = {c: i for i, c in enumerate(list(set(self.classes)))}

      for clase in self.classes:
          class_path = os.path.join(root_dir, clase, "train", "good")
          images = os.listdir(class_path)

          images = natsorted(images)

          for img_name in images:
            img_path = os.path.join(class_path, img_name)
            self.data.append((img_path, clase))

      self.rotation = RandomRotation(degrees=45)
      self.translation = RandomTranslate(translate=(0.2, 0.2))
      self.zoom = RandomZoom(zoom_range=0.2)
      self.grid_distort = GridDistortion(alpha=50)
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, clase = self.data[index]
        #img = Image.open(img_path).convert("RGB")        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        img.to(self.device)
        if random.random() < 0.5:
          transformed_img, transforms = self.apply_random_transforms(img) 
        else:
          transformed_img, transforms = img, [0,[0,0]]
        transformed_img.to(self.device)
        img.to(self.device)

        return transforms, img, transformed_img
    
    def apply_random_transforms(self, img):
        img, angle = self.rotation(img)
        img, trans = self.translation(img)
        img, zoom_factor = self.zoom(img)
        img = self.grid_distort(img)  
        return img, [angle, trans]