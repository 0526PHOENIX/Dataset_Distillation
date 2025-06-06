"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter

from Distillation import *
from Model import *
from Utils import *


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    device = torch.device('cuda:2')

    model = Iso_Dilate_Shuffle(16).to(device = device)

    pretrain = True

    """
    ====================================================================================================================
    Button Activate Mode
    ====================================================================================================================
    """
    # File Path
    data = os.path.abspath(os.path.join(__file__, '..', '..', 'Data', 'Data_2D'))
    result = os.path.abspath(os.path.join(__file__, '..', 'Distillation', 'Result'))
    weight = os.path.abspath(os.path.join(__file__, '..', 'Distillation', 'Result', 'Weight'))

    # Pretraining or Distillation
    if pretrain:

        # Pretraining
        params = {
                    'epoch':        50,
                    'batch':        16,
                    'lr':           1e-3,
                    'model':        model,
                    'device':       device,
                    'data':         data,
                    'result':       result,
                 }

        pretraining = Pretraining(**params)
        pretraining.main()

    else:

        # Distillation
        params = {
                    'epoch':        [500, 1, 20],
                    'batch':        2,
                    'lr':           1e-5,
                    'model':        model,
                    'device':       device,
                    'data':         data,
                    'result':       result,
                    'weight':       weight,
                 }
                        
        distillation = Distillation(**params)
        distillation.main()