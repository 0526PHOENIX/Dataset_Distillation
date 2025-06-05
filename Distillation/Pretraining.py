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
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from Utils import *


"""
========================================================================================================================
Global Constant
========================================================================================================================
"""
METRICS = 12
METRICS_LOSS        = 0
METRICS_LOSS_PIX    = 1
METRICS_LOSS_GDL    = 2
METRICS_LOSS_SIM    = 3
METRICS_LOSS_PER    = 4
METRICS_HEAD_MAE    = 5
METRICS_HEAD_PSNR   = 6
METRICS_HEAD_SSIM   = 7
METRICS_BONE_MAE    = 8
METRICS_BONE_PSNR   = 9
METRICS_BONE_SSIM   = 10
METRICS_BONE_DICE   = 11


"""
========================================================================================================================
Pretraining
========================================================================================================================
"""
class Pretraining():

    """
    ====================================================================================================================
    Critical Parameters
    ====================================================================================================================
    """
    def __init__(self,
                 epoch: int                 = 100,
                 batch: int                 = 16,
                 lr: float                  = 1e-3,
                 model: torch.nn.Module     = None,
                 device: torch.device       = torch.device('cpu'),
                 loss_lambda: list[float]   = [1, 1, 1, 1],
                 data: str                  = "",
                 result: str                = "",
                 weight: str                = "",
                 *args,
                 **kwargs) -> None:
        
        # Training Device: CPU(cpu) or GPU(cuda)
        self.device = device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print('\n' + 'Pretraining on: ' + str(self.device))

        # Total Epoch & Batch Size
        self.epoch = epoch
        self.batch = batch

        # Learning Rate
        self.lr = lr

        # Model
        self.model = model.to(self.device)
        print('\n' + 'Pretraining Model: ' + type(self.model).__name__)

        # Loss Function Weight + Change Rate
        self.lambda_pix, self.lambda_gdl, self.lambda_sim, self.lambda_per = loss_lambda

        # File Path
        self.data = data
        self.result = result
        self.weight = weight

        # Loss and Metrics
        self.get_loss = Loss(device = self.device)
        self.get_metrics = Metrics(device = self.device)

        # Optimizer, Data Loader
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

        # Optimizer: SGD
        self.opt = SGD(self.model.parameters(), lr = self.lr)

        # Training Data Loader and Sample Index
        self.train_dl = DL(root = self.data, mode = 'Train', device = self.device, batch_size = self.batch)
        self.train_index = np.random.randint(0, len(self.train_dl.dataset) - 1)

        # Metrics Filepath
        log_dir = os.path.join(self.result, 'Metrics', self.time)

        # Tensorboard Writer
        self.train_writer = SummaryWriter(log_dir)

        return

    """
    ====================================================================================================================
    Main Training Function
    ====================================================================================================================
    """
    def main(self) -> None:

        # Main Training
        for epoch_index in range(1, self.epoch + 1):
            
            """
            ============================================================================================================
            Training
            ============================================================================================================
            """
            # Get Training Metrics
            print('\n' + 'Training: ')
            metrics_train = self.training(epoch_index)

            # Save Training Metrics
            score = self.save_metrics(epoch_index, metrics_train)
            self.save_images(epoch_index)

            # Save Model
            self.save_model(epoch_index, score)

        # Close Tensorboard Writer
        self.train_writer.close()

        return
        
    """
    ====================================================================================================================
    Training Loop
    ====================================================================================================================
    """      
    def training(self, epoch_index: int) -> Tensor:

        # Buffer for Metrics
        metrics = torch.zeros(METRICS, len(self.train_dl), device = self.device)

        # Iterator
        l_bar = '{desc} {percentage:3.0f}% |'
        m_bar = '{bar:15}'
        r_bar = '| [{n_fmt:>3}/{total_fmt:>3}] [{elapsed}<{remaining}{postfix}]'
        progress = tqdm(enumerate(self.train_dl), total = len(self.train_dl), bar_format = l_bar + m_bar + r_bar)
        for batch_index, batch_tuple in progress:

            """
            ============================================================================================================
            Prepare Data
            ============================================================================================================
            """
            # real1: [-1, 1], real2: [-1, 1], hmask: [0 or 1], skull: [-1000, 3000]
            real1_g, real2_g, hmask_g, skull_g = batch_tuple

            # fake2: sCT
            fake2_g = self.model(real1_g)

            """
            ============================================================================================================
            Model
            ============================================================================================================
            """
            # Fresh Optimizer's Gradient
            self.opt.zero_grad()

            # Total Loss
            loss = torch.tensor(0.0, requires_grad = True).to(self.device)

            if self.lambda_pix > 0:
                # Pixelwise Loss
                loss_pix = self.get_loss.get_pix_loss(fake2_g, real2_g)
                loss = loss + self.lambda_pix * loss_pix

            if self.lambda_gdl > 0:
                # Gradient Difference Loss
                loss_gdl = self.get_loss.get_gdl_loss(fake2_g, real2_g)
                loss = loss + self.lambda_gdl * loss_gdl

            if self.lambda_sim > 0:
                # Similarity Loss
                loss_sim = self.get_loss.get_sim_loss(fake2_g, real2_g)
                loss = loss + self.lambda_sim * loss_sim

            if self.lambda_per > 0:
                # Perceptual Loss
                loss_per = self.get_loss.get_per_loss(fake2_g, real2_g)
                loss = loss + self.lambda_per * loss_per

            # Gradient Descent
            loss.backward()
            self.opt.step()

            """
            ============================================================================================================
            Metrics
            ============================================================================================================
            """
            # [-1000, 3000]
            real2_g = ((real2_g + 1) * 2000) - 1000
            fake2_g = ((fake2_g + 1) * 2000) - 1000

            # Head MAE, PSNR, SSIM
            head_mae, head_psnr, head_ssim = self.get_metrics.get_head(fake2_g, real2_g, hmask_g)

            # Bone MAE, PSNR, SSIM, DICE
            bone_mae, bone_psnr, bone_ssim, bone_dice = self.get_metrics.get_bone(fake2_g, skull_g)

            # Save Metrics
            metrics[METRICS_LOSS,      batch_index] = loss.item()
            metrics[METRICS_LOSS_PIX,  batch_index] = loss_pix.item() if 'loss_pix' in locals() else 0
            metrics[METRICS_LOSS_GDL,  batch_index] = loss_gdl.item() if 'loss_gdl' in locals() else 0
            metrics[METRICS_LOSS_SIM,  batch_index] = loss_sim.item() if 'loss_sim' in locals() else 0
            metrics[METRICS_LOSS_PER,  batch_index] = loss_per.item() if 'loss_per' in locals() else 0
            metrics[METRICS_HEAD_MAE,  batch_index] = head_mae
            metrics[METRICS_HEAD_PSNR, batch_index] = head_psnr
            metrics[METRICS_HEAD_SSIM, batch_index] = head_ssim
            metrics[METRICS_BONE_MAE,  batch_index] = bone_mae
            metrics[METRICS_BONE_PSNR, batch_index] = bone_psnr
            metrics[METRICS_BONE_SSIM, batch_index] = bone_ssim
            metrics[METRICS_BONE_DICE, batch_index] = bone_dice

            # Progress Bar Information
            progress.set_description_str('Epoch [ {:>4} / {:>4} ]'.format(epoch_index, self.epoch))
            progress.set_postfix_str('Head_MAE = {:<8} Bone_MAE = {:<8}'.format(head_mae, bone_mae))

            # Release Memory
            del fake2_g, loss
            del head_mae, head_psnr, head_ssim
            del bone_mae, bone_psnr, bone_ssim, bone_dice

        return metrics.to('cpu')
    
    """
    ====================================================================================================================
    Save Hyperparameter: Batch Size, Epoch, Learning Rate
    ====================================================================================================================
    """
    def save_hyper(self, epoch_index: int) -> None:

        path = os.path.join(self.result, 'Metrics', self.time, 'Hyper.txt')

        with open(path, 'w') as f:

            print('Time:',            self.time, file = f)
            print('Model:',           type(self.model).__name__, file = f)
            print('Epoch:',           epoch_index, file = f)
            print('Batch:',           self.batch, file = f)
            print('Learning Rate:',   self.lr, file = f)
            print('Pix Loss Lambda:', self.lambda_pix, file = f)
            print('GDL Loss Lambda:', self.lambda_gdl, file = f)
            print('Sim Loss Lambda:', self.lambda_sim, file = f)
            print('Per Loss Lambda:', self.lambda_per, file = f)

        return

    """
    ====================================================================================================================
    Save Metrics for Whole Epoch
    ====================================================================================================================
    """ 
    def save_metrics(self, epoch_index: int, metrics_t: Tensor) -> float:

        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.numpy().mean(axis = 1)

        # Create Dictionary
        metrics_dict = {}
        metrics_dict['Loss/Loss']           = metrics_a[METRICS_LOSS]
        metrics_dict['Loss/Loss_PIX']       = metrics_a[METRICS_LOSS_PIX]
        metrics_dict['Loss/Loss_GDL']       = metrics_a[METRICS_LOSS_GDL]
        metrics_dict['Loss/Loss_SIM']       = metrics_a[METRICS_LOSS_SIM]
        metrics_dict['Loss/Loss_PER']       = metrics_a[METRICS_LOSS_PER]
        metrics_dict['Metrics/Head_MAE']    = metrics_a[METRICS_HEAD_MAE]
        metrics_dict['Metrics/Head_PSNR']   = metrics_a[METRICS_HEAD_PSNR]
        metrics_dict['Metrics/Head_SSIM']   = metrics_a[METRICS_HEAD_SSIM]
        metrics_dict['Metrics/Bone_MAE']    = metrics_a[METRICS_BONE_MAE]
        metrics_dict['Metrics/Bone_PSNR']   = metrics_a[METRICS_BONE_PSNR]
        metrics_dict['Metrics/Bone_SSIM']   = metrics_a[METRICS_BONE_SSIM]
        metrics_dict['Metrics/Bone_DICE']   = metrics_a[METRICS_BONE_DICE]

        # Save Metrics
        for key, value in metrics_dict.items():
            self.train_writer.add_scalar(key, value, epoch_index)
        
        # Refresh Tensorboard Writer
        self.train_writer.flush()

        return metrics_dict['Metrics/Head_MAE']

    """
    ====================================================================================================================
    Save Image
    ====================================================================================================================
    """ 
    def save_images(self, epoch_index: int) -> None:

        with torch.no_grad():

            """
            ============================================================================================================
            Image: MR, rCT, sCT
            ============================================================================================================
            """ 
            # Model: Validation State
            self.model.eval()

            # real1: [-1, 1]; real2: [-1, 1]; hmask: [0 or 1]
            real1_t, real2_t, hmask_t, _ = self.train_dl.dataset[self.train_index]

            # fake2: sCT
            fake2_t = self.model(real1_t.to(self.device).unsqueeze(0)).squeeze(0).to('cpu')

            # Torch Tensor to Numpy Array
            real1_a, real2_a, fake2_a, hmask_a = (real1_t.numpy()[3 : 4], real2_t.numpy(), fake2_t.numpy(), hmask_t.numpy())

            # [0, 1]
            real1_a = (real1_a + 1) / 2
            real2_a = (real2_a + 1) / 2
            fake2_a = (fake2_a + 1) / 2

            # Remove Background
            fake2_a = np.where(hmask_a, fake2_a, 0)

            # Save Image
            self.train_writer.add_image('Train/MR',  real1_a, epoch_index, dataformats = 'CHW')
            self.train_writer.add_image('Train/rCT', real2_a, epoch_index, dataformats = 'CHW')
            self.train_writer.add_image('Train/sCT', fake2_a, epoch_index, dataformats = 'CHW')

            """
            ============================================================================================================
            Image: Difference Map
            ============================================================================================================
            """
            # Color Map
            colormap = LinearSegmentedColormap.from_list('colormap', [(1, 1, 1), (0, 0, 1), (1, 0, 0)])

            # Difference
            diff = np.abs(real2_a[0] - fake2_a[0]) * 4000

            # Difference Map + Colorbar
            fig = plt.figure(figsize = (5, 5))
            plt.imshow(diff, cmap = colormap, vmin = 0, vmax = 2000, aspect = 'equal')
            plt.colorbar()

            # Save Image
            self.train_writer.add_figure('Train/Diff', fig, epoch_index)

            # Refresh Tensorboard Writer
            self.train_writer.flush()

        return

    """
    ====================================================================================================================
    Save Model
    ====================================================================================================================
    """ 
    def save_model(self, epoch_index: int, score: float) -> None:

        # Time, Model, Optimizer, Loss Weight, Sample Index, Ending Epoch, Best Score
        state = {
                    'time':         self.time,
                    'model_state':  self.model.state_dict(),
                    'model_name':   type(self.model).__name__,
                    'opt_state':    self.opt.state_dict(),
                    'opt_name':     type(self.opt).__name__,
                    'epoch':        epoch_index,
                    'score':        score,
                }
    
        # Save Model
        torch.save(state, os.path.join(self.result, 'Weight', self.time + '_' + str(epoch_index) + '.pt'))

        return