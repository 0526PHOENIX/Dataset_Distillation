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

    device = torch.device('cuda')

    model = Iso_Dilate_Shuffle(8).to(device = device)

    """
    ====================================================================================================================
    Button Activate Mode
    ====================================================================================================================
    """
    # File Path
    data = os.path.abspath(os.path.join(__file__, '..', '..', 'Data', 'Data_2D'))
    result = os.path.abspath(os.path.join(__file__, '..', 'Frame', 'Result'))
        
    # Distillation
    params = {
                'batch':        2,
                'lr':           1e-3,
                'model':        model,
                'device':       device,
                'data':         data,
                'result':       result,
                }
                    
    training = Distillation(**params)
    training.main()