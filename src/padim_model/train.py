import torch

from .model import *
from.dataloader import MVTecDataset

from tqdm import tqdm

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries

from PIL import Image

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
import wandb

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')




def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def start(CLASS_NAMES,stn_model,data_path = '../data/mvtec_anomaly_detection',
          arch = 'resnet18',path_results='../results/mvtec_result',batch_size=128,experiment_name='exp-padim',
          use_stn = 1,args=None):
    
    
    PROJECT_WANDB = "neurips_23_padim"
    ENTITY = "ml_projects"

    config = wandb.config
    wandb.watch_called = False
    
    run = wandb.init(project=PROJECT_WANDB, 
                    entity=ENTITY,
                    config=args, 
                    name=experiment_name, 
                    job_type="model-training",
                    tags=["paper"])
    
    model,t_d,d =  base_model(device,arch=arch)

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)   

    os.makedirs(os.path.join(path_results, 'temp_%s' % arch), exist_ok=True)
    os.makedirs(os.path.join(path_results, 'temp_%s' % arch,str(run.id)), exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    
    CLASS_NAMES = sorted(CLASS_NAMES)

    for class_name in CLASS_NAMES:

        train_dataset = MVTecDataset(CLASS_NAMES,data_path , class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataset  = MVTecDataset(CLASS_NAMES,data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(path_results, 'temp_%s' % arch,str(run.id), 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            print("Training..")
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    #if use_stn==1:
                    #    x = stn_model(x.to(device))
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            gt_list.extend(y.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                if use_stn==1 and args.use_stn_mask == 1:
                    x,mask = stn_model(x.to(device), mask.to(device))
                _ = model(x.to(device))
                
            test_imgs.extend(x.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append({"class_name":class_name,
                              "image_ROCAUC":img_roc_auc})
        
        print(class_name,' test image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)

        gt_mask = np.array(gt_mask.flatten(), dtype=np.int8)
        #scores = np.array(scores.flatten(), dtype=np.float64)
        print(gt_mask.flatten())
        print(scores.flatten())
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

        total_pixel_roc_auc.append({"class_name":class_name,
                              "pixel_ROCAUC":per_pixel_rocauc})
        
        print(class_name,' test pixel ROCAUC: %.3f' % (per_pixel_rocauc))


        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = path_results + '/' + f'pictures_{arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
        
        wandb.log({(class_name+"_image_ROCAUC"):img_roc_auc,
                   (class_name+"_pixel_ROCAUC"):per_pixel_rocauc,

                   (class_name+"_plot"): wandb.Image(fig)

                   })

    df_image_rocauc = pd.DataFrame(total_roc_auc)
    df_pixel_rocauc = pd.DataFrame(total_pixel_roc_auc)
    
    df_total  = df_image_rocauc.merge(df_pixel_rocauc,on='class_name',how='left')
    
    
    wandb.log({"total_metrics":wandb.Table(dataframe=df_total)})
    
    run.finish()
    return fig_pixel_rocauc,fig_img_rocauc,total_roc_auc,total_pixel_roc_auc,fig