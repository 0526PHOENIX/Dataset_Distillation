"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import datetime
from typing import Literal
from tqdm import tqdm

import random

import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.optim import SGD
from torch.nn import Parameter
from Utils import *


"""
========================================================================================================================
Distillation
========================================================================================================================
"""
class Distillation():

    """
    ====================================================================================================================
    Critical Parameters
    ====================================================================================================================
    """
    def __init__(self,
                 epoch: list[int]           = [100, 25, 25],
                 batch: int                 = 16,
                 lr: float                  = 1e-3,
                 model: torch.nn.Module     = None,
                 device: torch.device       = torch.device('cpu'),
                 data: str                  = "",
                 result: str                = "",
                 weight: str                = "",
                 *args,
                 **kwargs) -> None:
        
        # Training Device: CPU(cpu) or GPU(cuda)
        self.device = device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print('\n' + 'Distilling on: ' + str(self.device))

        # Total Epoch & Batch Size
        self.epoch, self.teacher_epoch, self.student_epoch = epoch
        self.batch = batch

        # Learning Rate
        self.lr = lr

        # Model
        self.model = model.to(self.device)
        print('\n' + 'Distilling with Model: ' + type(self.model).__name__ + ' ' + str(self.model.width))

        # File Path
        self.data = data
        self.result = result
        self.weight = weight

        # Model, Optimizer, Data Loader
        self.initialization()

        return
    
    """
    ====================================================================================================================
    Optimizer, Data Loader
    ====================================================================================================================
    """
    def initialization(self) -> None:

        # Training Timestamp
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        print('\n' + 'Start From: ' + self.time)

        # Training Data Loader
        self.train_dl = DL(root = self.data, mode = 'Train', device = self.device, batch_size = self.batch)

        return
    
    """
    ====================================================================================================================
    Main Distillation Function
    ====================================================================================================================
    """
    def main(self) -> None:

        # Synthetic MR & CT
        syn_real1_g = Parameter(torch.tanh(torch.randn(2, 7, 256, 256, requires_grad = True).to(self.device)))
        syn_real2_g = Parameter(torch.tanh(torch.randn(2, 1, 256, 256, requires_grad = True).to(self.device)))

        # Raw Synthetic MR & CT
        raw_syn_real1_g = syn_real1_g.clone().detach()
        raw_syn_real2_g = syn_real2_g.clone().detach()

        # Model
        model = SwitchParams(self.model)
        model.train()

        # Learning Rate
        syn_lr = Parameter(torch.tensor(1e-3, requires_grad = True).to(self.device))

        # Optimizer: SGD
        opt_im = SGD([syn_real1_g, syn_real2_g], lr = self.lr, momentum = 0.5)
        opt_lr = SGD([syn_lr], lr = self.lr, momentum = 0.5)

        # Trajectory Path
        teacher_trajectory = sorted(os.listdir(self.weight), key = lambda x: int(x.rsplit('_', 1)[-1].split('.')[0]))

        # Save Image
        self.save_images(syn_real1_g, syn_real2_g, postfix = '')

        # Main Distillation
        for epoch_index in range(1, self.epoch + 1):

            # Starting & Ending Point
            start_epoch = random.randint(0, len(teacher_trajectory) - self.teacher_epoch - 1)
            final_epoch = start_epoch + self.teacher_epoch

            # Load Teacher's Parameter
            start_params = torch.load(os.path.join(self.weight, teacher_trajectory[start_epoch]))['model_state']
            final_params = torch.load(os.path.join(self.weight, teacher_trajectory[final_epoch]))['model_state']
            
            # Get Student's Parameter
            student_params = torch.cat([param.detach().clone().view(-1).to(self.device) for param in start_params.values()]).requires_grad_(True)

            # Simulate Training
            for student_epoch_index in range(1, self.student_epoch + 1):
                
                # Forward Pass wtih Flatten Parameter
                syn_fake2_g = model(syn_real1_g, flatten_param = student_params)

                # Pixel-Wise Loss
                loss = F.mse_loss(syn_fake2_g, syn_real2_g) + F.mse_loss(syn_real2_g, syn_fake2_g)

                # Compute Gradient
                grad = torch.autograd.grad(loss, student_params, create_graph = True)[0]

                # Gradient Descent
                student_params = student_params - syn_lr * grad

            # Flatten Teacher's Parameter
            start_params = torch.cat([param.data.view(-1).to(self.device) for param in start_params.values()])
            final_params = torch.cat([param.data.view(-1).to(self.device) for param in final_params.values()])

            # Initialize Loss Value
            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            # Compute Loss
            param_loss += F.mse_loss(student_params, final_params, reduction = "sum")
            param_dist += F.mse_loss(start_params, final_params, reduction = "sum")

            # Normalize
            param_loss /= model.param_total
            param_dist /= model.param_total
            param_loss /= param_dist

            # Fresh Optimizer's Gradient
            opt_im.zero_grad()
            opt_lr.zero_grad()

            # Gradient Descent
            param_loss.backward()
            opt_im.step()
            opt_lr.step()

            # Result Log
            print()
            print('=' * 110)
            print('Epoch' + '\t' + str(epoch_index))
            print('Loss' + '\t' + str(param_loss.item()))
            print('MR Difference' + '\t' + str((raw_syn_real1_g - syn_real1_g).abs().mean().item()))
            print('CT Difference' + '\t' + str((raw_syn_real2_g - syn_real2_g).abs().mean().item()))
            print('=' * 110)

            # Save Image
            self.save_images(syn_real1_g, syn_real2_g, postfix = '_')
        
            # Save Difference Map
            self.save_diff(raw_syn_real1_g - syn_real1_g, raw_syn_real2_g - syn_real2_g)

            # Clear Student Parameters
            for _ in student_params:
                del _

        # # Synthetic MR & CT
        # syn_real1_a = syn_real1_g.detach().cpu().numpy()
        # syn_real2_a = syn_real2_g.detach().cpu().numpy()

        # # Save Data
        # nib.save(nib.Nifti1Image(syn_real1_a, np.eye(4)), './Image/' + 'Syn_MR.nii')
        # nib.save(nib.Nifti1Image(syn_real2_a, np.eye(4)), './Image/' + 'Syn_CT.nii')

        return

    """
    ====================================================================================================================
    Save Image
    ====================================================================================================================
    """ 
    def save_images(self, syn_real1_g: Tensor, syn_real2_g: Tensor, postfix: str = '_') -> None:
        
        # Tensor to Array
        syn_real1_a = syn_real1_g.detach().cpu().numpy()
        syn_real2_a = syn_real2_g.detach().cpu().numpy()

        # Create the plot
        _, axs = plt.subplots(2, 2, figsize = (15, 15))

        # Remove Redundancy
        for ax in axs.flat:
            ax.axis('off')

        # Display Input MR Image
        ax = axs[0][0]
        ax.imshow(syn_real1_a[0, 3], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled MR 1')

        # Display Output CT Image
        ax = axs[0][1]
        ax.imshow(syn_real2_a[0, 0], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled CT 1')

        # Display Input MR Image
        ax = axs[1][0]
        ax.imshow(syn_real1_a[1, 3], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled MR 2')

        # Display Output CT Image
        ax = axs[1][1]
        ax.imshow(syn_real2_a[1, 0], cmap = 'gray', vmin = -1, vmax = 1)
        ax.set_title('Distilled CT 2')

        # Save Figure
        plt.tight_layout()
        plt.savefig('./Image/' + postfix + '.png', format = 'png', dpi = 300)
        plt.close()

        return

    """
    ====================================================================================================================
    Save Difference Map
    ====================================================================================================================
    """ 
    def save_diff(self, syn_diff1_g: Tensor, syn_diff2_g: Tensor, postfix: str = '_diff') -> None:

        # Tensor to Array
        syn_diff1_a = syn_diff1_g.detach().cpu().numpy()
        syn_diff2_a = syn_diff2_g.detach().cpu().numpy()

        # Create the plot
        _, axs = plt.subplots(2, 2, figsize = (15, 15))

        # Remove Redundancy
        for ax in axs.flat:
            ax.axis('off')

        # Display Input MR Image
        ax = axs[0][0]
        plot = ax.imshow(syn_diff1_a[0, 3], cmap = 'turbo')
        plt.colorbar(plot, ax = ax, cax = ax.inset_axes((1, 0, 0.05, 1.0)))
        ax.set_title('Distilled MR 1')

        # Display Output CT Image
        ax = axs[0][1]
        plot = ax.imshow(syn_diff2_a[0, 0], cmap = 'turbo')
        plt.colorbar(plot, ax = ax, cax = ax.inset_axes((1, 0, 0.05, 1.0)))
        ax.set_title('Distilled CT 1')

        # Display Input MR Image
        ax = axs[1][0]
        plot = ax.imshow(syn_diff1_a[1, 3], cmap = 'turbo')
        plt.colorbar(plot, ax = ax, cax = ax.inset_axes((1, 0, 0.05, 1.0)))
        ax.set_title('Distilled MR 2')

        # Display Output CT Image
        ax = axs[1][1]
        plot = ax.imshow(syn_diff2_a[1, 0], cmap = 'turbo')
        plt.colorbar(plot, ax = ax, cax = ax.inset_axes((1, 0, 0.05, 1.0)))
        ax.set_title('Distilled CT 2')

        # Save Figure
        plt.tight_layout()
        plt.savefig('./Image/' + postfix + '.png', format = 'png', dpi = 300)
        plt.close()

        return