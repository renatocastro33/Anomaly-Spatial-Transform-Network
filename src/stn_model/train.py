import torch
import numpy as np
import torch.optim as optim
from torch import Tensor, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from .dataloader import MVTEC
from .model import SpatialTransformerNetwork

from itertools import islice
#from tqdm.notebook import trange, tqdm
from tqdm import tqdm

import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import wandb
import wandb
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize and crop the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    
])

def reverse_transformation(img,device):
    with torch.no_grad():
      img = [img]
      img = torch.stack(img).to(device)
      img = img[0].cpu().detach().numpy().transpose((1, 2, 0))
      img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    return img

# Definir una función para cargar y preprocesar imágenes
def load_and_preprocess_image(stn_model,img_path,device):
    image = Image.open(img_path).convert("RGB")
    image = train_transform(image)
    imgs = [image]
    inputs = torch.stack(imgs).to(device)
    with torch.no_grad():
        stn_predicted = stn_model(inputs)
    orig_image = inputs[0].cpu().numpy().transpose((1, 2, 0))
    stn_predicted_image = stn_predicted[0].cpu().numpy().transpose((1, 2, 0))
    orig_image = (orig_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    stn_predicted_image = (stn_predicted_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    return orig_image, stn_predicted_image, orig_image - stn_predicted_image


def save_model(model,model_name='../results/stn_model.pt'):
    torch.save(model.state_dict(), model_name)
    
def load_model(model,model_name='../results/stn_model.pt'):
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model

def start(data_dir ='../data/mvtec_anomaly_detection',batch_size = 32,learning_rate = 0.001,num_epochs = 10,
          experiment_name="exp-1",loss_name='mse',args=''):
    PROJECT_WANDB = "neurips_23"
    ENTITY = "ml_projects"





    # Log the parameters to wandb
    #wandb.config.update({
    #})

    config = wandb.config
    wandb.watch_called = False
    
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    mvtec_dataset = MVTEC(root_dir=data_dir, transform = train_transform,device=device)
    train_loader = DataLoader(mvtec_dataset, batch_size=batch_size, shuffle=True)
    # Define the proportion of data to use for validation
    validation_proportion = 0.3  # Adjust as needed
    
    # Calculate the number of samples for validation
    total_samples = len(train_loader.dataset)
    num_validation_samples = int(validation_proportion * total_samples)
    
    # Calculate the number of samples for training
    num_training_samples = total_samples - num_validation_samples
    
    # Split the train_loader into train and validation loaders
    training_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(range(num_training_samples)),
        num_workers=32,
    )
    
    validation_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(range(num_training_samples, total_samples)),
    )
    
    
    
    
    

    # Using GPU
    print("Using GPU optimization")
    stn_model = SpatialTransformerNetwork()
    #stn_model = nn.DataParallel(stn_model) 
    stn_model.to(device)

    if loss_name=='mse':
        criterion = nn.MSELoss()    
    elif loss_name== 'l1_mse':
        """
        class CombinedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
                self.l1 = nn.L1Loss()
            
            def forward(self, output, target):        
                return self.mse(output, target) + self.l1(output, target)
        """
        class CombinedLoss(nn.Module):
            def __init__(self, ssim_weight=0.1,l1_weight=1):
                super().__init__()
                self.mse = nn.MSELoss()
                self.l1 = nn.L1Loss()
                self.ssim = ssim  # Ajusta el tamaño de la ventana según tus necesidades
                self.ssim_weight = ssim_weight
                self.l1_weight = l1_weight

            def forward(self, output, target):
                mse_loss = self.mse(output, target)
                l1_loss = self.l1(output, target)
                ssim_loss = 1 - self.ssim(output, target)

                return mse_loss + self.l1_weight*l1_loss + self.ssim_weight * ssim_loss

        # Uso
        criterion = CombinedLoss(ssim_weight=args.ssim_weight,
                                 l1_weight=args.l1_weight) 

        print("*"*20)
        print("use CombinedLoss!")
        #criterion = CombinedLoss()
    else:    
        criterion = ssim
        
    optimizer = optim.Adam(stn_model.parameters(), lr=learning_rate)

    print("Training..")
    # Define el número total de batches
    total_batches = len(train_loader)
    
    run = wandb.init(project=PROJECT_WANDB, 
                     entity=ENTITY,
                     config=args, 
                     name=experiment_name, 
                     job_type="model-training",
                     tags=["paper"])
    
    best_ssim = -1
    
    for epoch in range(num_epochs):
        stn_model.train()
        running_loss = 0.0
        running_ssim_value = 0.0
        running_ms_ssim_value = 0.0

        train_images = []


        # Crea una barra de progreso con tqdm
        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for batch_idx, (transforms, img, transformed_img) in enumerate(train_loader):  
                
                img = img.to(device)
                transformed_img = transformed_img.to(device)
                    
                optimizer.zero_grad()

                transformed_recovered = stn_model(transformed_img)

                loss = criterion(transformed_recovered, img)

                if loss_name =='ssim':
                    loss = 1-loss
                loss.backward()
                optimizer.step()
                    
                running_loss += loss.item()

                if batch_idx==0:
                  train_images.append((img[0],transformed_img[0],transformed_recovered[0]))
                  train_images.append((img[1],transformed_img[1],transformed_recovered[1]))
                  train_images.append((img[2],transformed_img[2],transformed_recovered[2]))
                  print("img max :",img.max(),'img min :',img.min())
                  print("transformed_img max :",transformed_img.max(),'transformed_img min :',transformed_img.min())
                  print("transformed_recovered max :",transformed_recovered.max(),'transformed_recovered min :',transformed_recovered.min())

                # Convertir tensores a numpy
                
                #recovered = recovered.detach().cpu()
                #img = img.detach().cpu()
                
                # Calcula SSIM (entre 0 y 1)
                ssim_value = ssim(transformed_recovered, img, data_range=img.max() - img.min())
                # Calcula MS-SSIM (extiende SSIM para multiples escalas)
                ms_ssim_value = ms_ssim(transformed_recovered, img, data_range=img.max() - img.min())
                
                running_ssim_value+=ssim_value.item()
                running_ms_ssim_value+=ms_ssim_value.item()
                
                # Actualiza la barra de progreso
                pbar.set_postfix(loss=loss.item(), ssim=ssim_value.item(), ms_ssim=ms_ssim_value.item())
                pbar.update(1)

        # Calcula métricas finales de la época y muestra en la consola
        average_loss = running_loss / total_batches
        average_ssim = running_ssim_value / total_batches
        average_ms_ssim = running_ms_ssim_value / total_batches

        # Imágenes recuperada y original 
        recovered = stn_model(transformed_img)
        orig = img


        # Validation phase
        stn_model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0
        val_ssim_value = 0.0
        val_ms_ssim_value = 0.0

        val_images = []

        with torch.no_grad():
            # Crea una barra de progreso con tqdm para la validación
            with tqdm(total=len(validation_loader), desc="Validation", unit="batch") as pbar_val:
                for batch_idx, (transforms, img, transformed_img) in enumerate(validation_loader):
                    

                    img = img.to(device)
                    transformed_img = transformed_img.to(device)                    
                    transformed_recovered = stn_model(transformed_img)

                    loss = criterion(transformed_recovered, img)
                    
                    if loss_name =='ssim':
                        loss = 1-loss

                    if batch_idx==0:
                      val_images.append((img[0],transformed_img[0],transformed_recovered[0]))
                      val_images.append((img[1],transformed_img[1],transformed_recovered[1]))
                      val_images.append((img[2],transformed_img[2],transformed_recovered[2]))

                      print("img max :",img.max(),'img min :',img.min())
                      print("transformed_img max :",transformed_img.max(),'transformed_img min :',transformed_img.min())
                      print("transformed_recovered max :",transformed_recovered.max(),'transformed_recovered min :',transformed_recovered.min())

                    ssim_value    = ssim(transformed_recovered, img, data_range=img.max() - img.min())
                    ms_ssim_value = ms_ssim(transformed_recovered, img, data_range=img.max() - img.min())
                    
                    val_ssim_value += ssim_value.item()
                    val_ms_ssim_value += ms_ssim_value.item()
                    validation_loss += loss.item()

                    # Actualiza la barra de progreso de la validación
                    pbar_val.set_postfix(loss=loss.item(), ssim=ssim_value, ms_ssim=ms_ssim_value)
                    pbar_val.update(1)

            # Calcula métricas finales de validación y muestra en la consola
            average_val_loss = validation_loss / len(validation_loader)
            average_val_ssim = val_ssim_value / len(validation_loader)
            average_val_ms_ssim = val_ms_ssim_value / len(validation_loader)
            

        print(f"Epoch {epoch+1} Train Loss: {average_loss:.3f}, SSIM: {average_ssim:.3f}, MS-SSIM: {average_ms_ssim:.3f}")
        print(f"Epoch {epoch+1} Val   Loss: {average_val_loss:.3f}, SSIM: {average_val_ssim:.3f}, MS-SSIM: {average_val_ms_ssim:.3f}")
                    

        # Inicializa WandB
        #wandb.init(project='your_project_name', name='image_visualization')

        #training images
        fig1, axes1 = plt.subplots(6, 4, figsize=(20, 20))

        for i, lista in enumerate(train_images):

            img,transformed_img,transformed_recovered = lista

            # Plot the images side by side
            axes1[i, 0].imshow(reverse_transformation(img,device))
            axes1[i, 0].set_title('Train Original Image')
            axes1[i, 0].axis('off')

            axes1[i, 1].imshow(reverse_transformation(transformed_img,device))
            axes1[i, 1].set_title('Train Transformed Image')
            axes1[i, 1].axis('off')

            axes1[i, 2].imshow(reverse_transformation(transformed_recovered,device))
            axes1[i, 2].set_title('Train Recovered Transformed Image')
            axes1[i, 2].axis('off')

            axes1[i, 3].imshow(reverse_transformation(img,device)-reverse_transformation(transformed_recovered,device))
            axes1[i, 3].set_title('Train Diff Image - recovered')
            axes1[i, 3].axis('off')

        for i, lista in enumerate(val_images):

            img,transformed_img,transformed_recovered = lista

            # Plot the images side by side
            axes1[i+3, 0].imshow(reverse_transformation(img,device))
            axes1[i+3, 0].set_title('Val Original Image')
            axes1[i+3, 0].axis('off')

            axes1[i+3, 1].imshow(reverse_transformation(transformed_img,device))
            axes1[i+3, 1].set_title('Val Transformed Image')
            axes1[i+3, 1].axis('off')

            axes1[i+3, 2].imshow(reverse_transformation(transformed_recovered,device))
            axes1[i+3, 2].set_title('Val Recovered Transformed Image')
            axes1[i+3, 2].axis('off')

            axes1[i+3, 3].imshow(reverse_transformation(img,device)-reverse_transformation(transformed_recovered,device))
            axes1[i+3, 3].set_title('Train Diff Image - recovered')
            axes1[i+3, 3].axis('off')


        # Obtén las rutas de las imágenes
        img_paths_1 =  [(glob.glob("../data/bottle/test/broken_large/*.png")[10],'bottle_broken_large')]
        img_paths_1 += [(glob.glob("../data/cable/test/bent_wire/*.png")[10],'cable_bent_wire')]
        img_paths_1 += [(glob.glob("../data/capsule/test/crack/*.png")[10],'capsule_crack')]
        img_paths_1 += [(glob.glob("../data/carpet/test/hole/*.png")[10],'carpet_hole')]
        img_paths_1 += [(glob.glob("../data/screw/test/scratch_head/*.png")[10],'screw_scratch_head')]
        
        img_paths_2 =  [(glob.glob("../data/grid/test/broken/*.png")[10],'grid_broken')]
        img_paths_2 += [(glob.glob("../data/hazelnut/test/crack/*.png")[10],'hazelnut_crack')]
        img_paths_2 += [(glob.glob("../data/leather/test/glue/*.png")[10],'leather_glue')]
        img_paths_2 += [(glob.glob("../data/metal_nut/test/scratch/*.png")[10],'metal_nut_scratch')]
        img_paths_2 += [(glob.glob("../data/zipper/test/broken_teeth/*.png")[10],'zipper_broken_teeth')]

        # Crea una figura para colocar las imágenes
        fig, axes = plt.subplots(len(img_paths_1), 6, figsize=(30, 5 * len(img_paths_1)))

        for i, values in enumerate(img_paths_1):
            orig_image, stn_predicted_image, diff_image = load_and_preprocess_image(stn_model,img_paths_1[i][0],device)

            # Plot the images side by side
            axes[i, 0].imshow(orig_image)
            axes[i, 0].set_title('Original Image '+img_paths_1[i][1])
            axes[i, 0].axis('off')

            axes[i, 1].imshow(stn_predicted_image)
            axes[i, 1].set_title('STN Predicted Image '+img_paths_1[i][1])
            axes[i, 1].axis('off')

            axes[i, 2].imshow(diff_image)
            axes[i, 2].set_title('DIFF STN Predicted Image '+img_paths_1[i][1])
            axes[i, 2].axis('off')

            orig_image, stn_predicted_image, diff_image = load_and_preprocess_image(stn_model,img_paths_2[i][0],device)

            # Plot the images side by side
            axes[i, 3].imshow(orig_image)
            axes[i, 3].set_title('Original Image '+img_paths_2[i][1])
            axes[i, 3].axis('off')

            axes[i, 4].imshow(stn_predicted_image)
            axes[i, 4].set_title('STN Predicted Image '+img_paths_2[i][1])
            axes[i, 4].axis('off')

            axes[i, 5].imshow(diff_image)
            axes[i, 5].set_title('DIFF STN Predicted Image '+img_paths_2[i][1])
            axes[i, 5].axis('off')

        wandb.log({
                   "train_loss":average_loss,
                   "train_ssim":average_ssim,
                   "train_ms_ssim":average_ms_ssim,

                   "val_loss"  :average_val_loss,
                   "val_ssim":average_val_ssim,
                   "val_ms_ssim":average_val_ms_ssim,
                   "image_training": wandb.Image(fig1),
                   
                   "image_testing": wandb.Image(fig)})
        
        save_model(stn_model,model_name='../results/models/stn_model.pt')

        if average_val_ssim > best_ssim:
            best_ssim = average_val_ssim
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': stn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': average_loss,
                        'best_ssim':average_val_ssim
                    },"../results/models/checkpoint_best_model.pth")
                
            artifact = wandb.Artifact(f'best-model_{run.id}.pth', type='model')
            artifact.add_file("../results/models/checkpoint_best_model.pth")
            run.log_artifact(artifact)

    print("Finished Training")
    run.finish()
    
    return stn_model,train_loader


def visualize_stn(stn_model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stn_model.to(device)
    stn_model.eval()

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            print(inputs.size())

            # Get the STN-predicted image
            stn_predicted = stn_model(inputs)

            # Convert tensors to numpy arrays for visualization
            orig_image = inputs[0].cpu().numpy().transpose((1, 2, 0))
            stn_predicted_image = stn_predicted[0].cpu().numpy().transpose((1, 2, 0))

            # Undo normalization to display images correctly
            orig_image = (orig_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            orig_image = np.clip(orig_image, 0, 1)

            stn_predicted_image = (stn_predicted_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            stn_predicted_image = np.clip(stn_predicted_image, 0, 1)

            # Plot the images side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(stn_predicted_image)
            axes[1].set_title('STN Predicted Image')
            axes[1].axis('off')

            plt.show()
            print(stn_predicted.size())
            print(len(stn_predicted_image))
            break  # Show only the first image from the batch


