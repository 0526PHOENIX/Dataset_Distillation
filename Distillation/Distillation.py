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
                 epoch: list[int]           = [400, 100, 100],
                 batch: int                 = 16,
                 lr: float                  = 1e-3,
                 model: torch.nn.Module     = None,
                 device: torch.device       = torch.device('cpu'),
                 data: str                  = "",
                 result: str                = "",
                 *args,
                 **kwargs) -> None:
        
        # Training Device: CPU(cpu) or GPU(cuda)
        self.device = device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print('\n' + 'Training on: ' + str(self.device))

        # Total Epoch & Batch Size
        self.epoch, self.teacher_epoch, self.student_epoch = epoch
        self.batch = batch

        # Learning Rate
        self.lr = lr

        # Model
        self.model = model.to(self.device)
        print('\n' + 'Training Model: ' + type(self.model).__name__ + ' ' + str(self.model.width))

        # File Path
        self.data = data
        self.result = result

        # Loss and Metrics
        self.get_loss = Loss(device = self.device)
        self.get_metrics = Metrics(device = self.device)

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

        # 
        syn_real1_g = Parameter(torch.randn(5, 7, 256, 256, requires_grad = True).to(self.device))
        syn_real2_g = Parameter(torch.randn(5, 1, 256, 256, requires_grad = True).to(self.device))

        self.save_images(syn_real1_g, syn_real2_g, postfix = '__')

        syn_lr = torch.tensor(0.01, requires_grad = True).to(self.device)

        # Optimizer: SGD
        opt_im = SGD([syn_real1_g, syn_real2_g], lr = self.lr, momentum = 0.5)
        opt_lr = SGD([syn_lr], lr = self.lr, momentum = 0.5)

        # 
        path = ""
        teacher_trajectory = sorted(os.listdir(path))

        # Main Distillation
        for epoch_index in range(1, self.epoch + 1):

            # 
            start_epoch = random.randint(0, len(teacher_trajectory) - self.teacher_epoch - 1)
            final_epoch = start_epoch + self.teacher_epoch

            # 
            start_params = torch.load(teacher_trajectory[start_epoch])['model_state']
            final_params = torch.load(teacher_trajectory[final_epoch])['model_state']
            
            # Model
            model = self.model
            model.train()
            
            # 
            student_params = [start_params.clone().requires_grad_()]

            # 
            for student_epoch_index in range(1, self.student_epoch + 1):

                syn_fake2_g = model(syn_real1_g)

                loss = self.get_loss.get_pix_loss(syn_fake2_g, syn_real2_g)

                grad = torch.autograd.grad(loss, student_params[-1], create_graph = True)[0]

                student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(self.device)
        param_dist = torch.tensor(0.0).to(self.device)

        param_loss += F.mse_loss(student_params[-1], final_params, reduction = "sum")
        param_dist += F.mse_loss(start_params, final_params, reduction = "sum")

        num_params = sum([np.prod(params.size()) for params in (model.parameters())])
        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        opt_im.zero_grad()
        opt_lr.zero_grad()

        grand_loss.backward()

        opt_im.step()
        opt_lr.step()

        return

    """
    ====================================================================================================================
    Save Image
    ====================================================================================================================
    """ 
    def save_images(self, syn_real1_g: Tensor, syn_real2_g: Tensor, postfix: str = '_') -> None:

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