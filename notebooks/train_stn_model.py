import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import sys
import os
import numpy as np
sys.path.insert(1, '../')


from src.padim_model import train as padim_train
from src.stn_model import train as stn_train
from src import utils
from src.stn_model.gpu import configurar_cuda_visible


CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

import argparse


def get_default_args():

  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', default='../data', type=str, help='directorio de los datos')
  parser.add_argument('--batch_size', default=128, type=int, help='tamaño de batch') 
  parser.add_argument('--learning_rate', default=0.0005, type=float, help='tasa de aprendizaje')
  parser.add_argument('--num_epochs', default=1, type=int, help='número de épocas')
  parser.add_argument('--loss_name', default='mse', type=str, help='función de pérdida') 
  parser.add_argument('--model_name', default='../results/stn_model_01_new.pt', type=str, help='path para guardar el modelo')
  parser.add_argument('--sweep', default=0, type=int, help='path para guardar el modelo')
  parser.add_argument('--device', default='0', type=str, help='path para guardar el modelo')
  parser.add_argument('--ssim_weight', default=0.1, type=float, help='path para guardar el modelo')
  parser.add_argument('--l1_weight', default=1, type=float, help='path para guardar el modelo')

  return parser

def train(args):



  #args = parser.parse_args()

  print(args)
  
  utils.set_seed()

  stn_model,train_loader = stn_train.start(data_dir =args.data_dir,
                        batch_size = args.batch_size,
                        learning_rate = args.learning_rate,num_epochs = args.num_epochs,loss_name=args.loss_name,args=args)

  stn_train.save_model(stn_model,model_name=args.model_name)


  from src.stn_model.model import SpatialTransformerNetwork
  import torch
  from torch import Tensor, nn

  stn_model = SpatialTransformerNetwork()
  #stn_model = nn.DataParallel(stn_model)
  stn_model.to("cuda")

  stn_model_loaded = stn_train.load_model(stn_model,model_name='../results/stn_model_01_new.pt')
  stn_model_loaded.to("cuda")
  stn_model_loaded.eval()



  import torch
  import numpy as np
  from torchvision import transforms
  import os
  from PIL import Image
  import torch
  from torch.utils.data import Dataset, DataLoader



  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),  # Resize and crop the image to 224x224
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

  ])

  import glob
  import matplotlib.pyplot as plt

  img_paths = glob.glob("../data/bottle/train/good/*.png")
  #img_paths = glob.glob("data/bottle/test/broken_large/*.png")

  for i,img_path_1 in enumerate(img_paths[:5]):
    image_1 = Image.open(img_path_1).convert("RGB")

    image_1 = train_transform(image_1)
    imgs = [image_1]
    inputs = torch.stack(imgs).cuda()
    with torch.no_grad():
      stn_predicted = stn_model_loaded(inputs)

    # Convert tensors to numpy arrays for visualization
    orig_image = inputs[0].cpu().numpy().transpose((1, 2, 0))
    stn_predicted_image = stn_predicted[0].cpu().numpy().transpose((1, 2, 0))

    

    # Undo normalization to display images correctly
    orig_image = (orig_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    #orig_image = np.clip(orig_image, 0, 1)

    stn_predicted_image = (stn_predicted_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    #stn_predicted_image = np.clip(stn_predicted_image, 0, 1)

    # Plot the images side by side
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(stn_predicted_image)
    axes[1].set_title('STN Predicted Image')
    axes[1].axis('off')

    axes[2].imshow(orig_image-stn_predicted_image)
    axes[2].set_title('DIFF STN Predicted Image')
    axes[2].axis('off')
    
    orig_name = f"comparison_{i}.png"
    fig.savefig(orig_name)
    print(f"Figure saved to: {orig_name}")
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    
    print("args.device:",args.device)
    #@configurar_cuda_visible([int(value) for value in args.device.split(",")])
    def main_process(): 
        train(args)
    main_process()