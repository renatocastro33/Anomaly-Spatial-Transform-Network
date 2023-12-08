import os
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as TF

import torch
import torchvision.transforms as transforms


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        rotated_img = transforms.functional.rotate(img, angle)
        
        return rotated_img, angle
    
    
class RandomTranslate:
    def __init__(self, translate): 
        self.translate = translate

    def __call__(self, img):
        horiz_trans = random.uniform(-self.translate[0], self.translate[0]) 
        vert_trans = random.uniform(-self.translate[1], self.translate[1])
        
        translated_img = transforms.functional.affine(img, angle=0, translate=[horiz_trans, vert_trans], scale=1.0, shear=0)
        
        return translated_img, [horiz_trans, vert_trans]



class MVTEC(Dataset):
    def __init__(self, root_dir, transform=None):
      self.root_dir = root_dir
      self.transform = transform
      self.classes = [mvtec_class for mvtec_class in os.listdir(root_dir) if not '.' in mvtec_class]
      self.data = []  # Collect all the data upon initialization
      
      sorted(self.classes)
      self.class_to_idx = {c: i for i, c in enumerate(list(set(self.classes)))}

      for cls in self.classes:
          class_path = os.path.join(root_dir, cls, "train", "good")
          images = os.listdir(class_path)
          for img_name in images:
              img_path = os.path.join(class_path, img_name)
              self.data.append((img_path, cls))

      self.rotation = RandomRotation(degrees=45)
      self.translation = RandomTranslate(translate=(0.2, 0.2))
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, cls = self.data[index]
        img = Image.open(img_path).convert("RGB")        
        # aplicar transformación aleatoria
        if self.transform:
            img = self.transform(img)

        transformed_img, transforms = self.apply_random_transforms(img) 
        
        return transforms, img, transformed_img
    
    def apply_random_transforms(self, img):
        rotation_img, angle = self.rotation(img)
        translation_img, trans = self.translation(rotation_img)
            
        return translation_img, [angle, trans]

"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [mvtec_class for mvtec_class in os.listdir(root_dir) if not '.' in mvtec_class]
        self.data = []  # Collect all the data upon initialization
        
        sorted(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(list(set(self.classes)))}

        for cls in self.classes:
            class_path = os.path.join(root_dir, cls, "train", "good")
            images = os.listdir(class_path)
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, cls))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, cls = self.data[index]
        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            print(f"Error leyendo {img_path}, imagen dañada")
            return None,None

        if self.transform:
            image = self.transform(image)

        image = image.to("cuda")
        class_idx = torch.tensor(self.class_to_idx[cls]).to("cuda")
        return image, class_idx
"""