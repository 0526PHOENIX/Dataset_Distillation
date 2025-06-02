"""
========================================================================================================================
Package
========================================================================================================================
"""
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import Conv2d
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import Sigmoid, GELU, Tanh


import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Model import *


"""
========================================================================================================================
Flatten Parameter Module
========================================================================================================================
"""
class SwitchParams(Module):

    """
    ===================================================================================================================
    Initialization
    ===================================================================================================================
    """
    def __init__(self, model: Module) -> None:

        super().__init__()
        
        # Model
        self.model = model

        # Parameter Information
        self.param_shape = [param.shape   for param in model.parameters()]
        self.param_numel = [param.numel() for param in model.parameters()]
        self.param_total = sum(self.param_numel)

        # Flatten Parameter
        self.flat_param = Parameter(torch.cat([param.reshape(-1) for param in model.parameters()], 0))

        return
    
    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor, flat_param: Tensor) -> Tensor:

        # Update Flatten Parameter
        self.flat_param.data.copy_(flat_param)

        # Unflatten the Flatten Parameter
        params = []
        indice = 0
        for shape, numel in zip(self.param_shape, self.param_numel):
            params.append(flat_param[indice : indice + numel].view(shape))
            indice += numel
        
        # Copy the flat params into the model
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), params):
                param.copy_(new_param)

        return self.model(img_in1)


"""
========================================================================================================================
Basic Model
========================================================================================================================
"""
class BasicModel(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self) -> None:

        super().__init__()

        #
        self.block0 = Conv2d(8, 8, kernel_size = 3, padding = 1)
        self.block1 = Conv2d(8, 8, kernel_size = 3, padding = 1)

        return

    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:

        img_out = self.block0(img_in1)
        img_out = self.block1(img_out)

        return img_out
    

"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    model = SwitchParams(BasicModel())

    print()
    print(model.param_numel)
    print(model.param_shape)
    print(model.param_total)
    print()

    print()
    print(model.flat_param)
    print()