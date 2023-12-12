import random
import torchvision.transforms.functional as TF
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


import random

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



class RandomZoom(object):
    def __init__(self, zoom_range):
        self.zoom_range = zoom_range
        
    def __call__(self, img):
        zoom_factor = random.uniform(1-self.zoom_range, 1+self.zoom_range)
        height, width = img.shape[-2:]
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        
        resized_img = transforms.functional.resize(img, (new_height, new_width))
        
        if zoom_factor > 1:
            # alejamiento
            y1 = random.randint(0, new_height - height)
            x1 = random.randint(0, new_width - width)
            crop_img = resized_img[:, y1:y1 + height, x1:x1 + width]
        else:
            # acercamiento
            new_img = torch.zeros((3, height, width))
            y1 = random.randint(0, height - new_height)
            x1 = random.randint(0, width - new_width)
            new_img[:, y1:y1 + new_height, x1:x1 + new_width] = resized_img
            
            crop_img = new_img
            
        return crop_img, zoom_factor

from torchvision.transforms import v2



class GridDistortion:
    def __init__(self,alpha=50):
        self.alpha = alpha    
    def __call__(self, img):
        alpha = random.uniform(self.alpha-20,self.alpha+20)
        self.elastic_transform = v2.ElasticTransform(alpha=alpha)
        img = self.elastic_transform(img)
        return img