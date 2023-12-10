import sys
import os
import numpy as np
sys.path.insert(1, '../')


from src.padim_model import train as padim_train
from src.stn_model import train as stn_train
from src import utils




from src.stn_model.model import SpatialTransformerNetwork
import torch
from torch import Tensor, nn


import argparse


def get_default_args():

  parser = argparse.ArgumentParser()

  parser.add_argument('--data_path', default='../data', type=str, help='directorio de los datos')
  
  parser.add_argument('--experiment_name', default='exp-padim', type=str, help='función de pérdida') 
  parser.add_argument('--batch_size', default=128, type=int, help='tamaño de batch') 
  parser.add_argument('--learning_rate', default=0.0005, type=float, help='tasa de aprendizaje')
  parser.add_argument('--path_results', default='../results/padim_results', type=str, help='número de épocas')
  parser.add_argument('--arch', default='resnet18', type=str, help='función de pérdida') 
  parser.add_argument('--use_stn', default=0, type=int, help='path para guardar el modelo')
  parser.add_argument('--use_stn_mask', default=0, type=int, help='path para guardar el modelo')
  parser.add_argument('--path_model_stn', default='../results/models/v1_mse/checkpoint_best_model.pth', type=str, help='path para guardar el modelo')
  parser.add_argument('--device', default='0', type=str, help='path para guardar el modelo')

  return parser



def train(args):
    #args = parser.parse_args()

    print(args)

    utils.set_seed()
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    #CLASS_NAMES = ['screw']

    stn_model = None
    if args.use_stn == 1:
        
        stn_model = SpatialTransformerNetwork()
        #stn_model = nn.DataParallel(stn_model)
        checkpoint = torch.load(args.path_model_stn,map_location='cuda:0')
        stn_model.load_state_dict(checkpoint["model_state_dict"])
        stn_model.to("cuda")
        stn_model.eval()

    fig_pixel_rocauc,fig_img_rocauc,total_roc_auc,total_pixel_roc_auc,fig = padim_train.start(CLASS_NAMES,
                    stn_model,data_path = args.data_path,arch = args.arch,path_results=args.path_results,
                    batch_size=args.batch_size,experiment_name=args.experiment_name,use_stn = args.use_stn,args=args)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    print("args.device:",args.device)
    def main_process(): 
        train(args)
    main_process()
    
    #python train_padim.py --arch resnet18 --use_stn 0 --path_model_stn ../results/models/v1_mse/checkpoint_best_model.pth
    #! python train_padim.py --arch resnet18 --use_stn_mask 0 --use_stn 0 --path_model_stn ../results/models/v1_mse/checkpoint_best_model_ssim_weight_08.pth
    #! python train_padim.py --arch resnet18 --use_stn_mask 1 --use_stn 1 --path_model_stn ../results/models/v1_mse/checkpoint_best_model_ssim_weight_08.pth
    #!python train_padim.py --arch wide_resnet50_2 --use_stn_mask 0 --use_stn 0 --path_model_stn ../results/models/v1_mse/checkpoint_best_model_ssim_weight_08.pth
    #!python train_padim.py --arch wide_resnet50_2 --use_stn_mask 1 --use_stn 1 --path_model_stn ../results/models/v1_mse/checkpoint_best_model_ssim_weight_08.pth