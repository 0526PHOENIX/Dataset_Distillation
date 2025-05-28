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


"""
========================================================================================================================
Layer Normalization
========================================================================================================================
"""
class LayerNorm(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()
        
        # Scale and Shift
        self.scale = Parameter(torch.ones(1, filters, 1, 1))
        self.shift = Parameter(torch.zeros(1, filters, 1, 1))

        return
    
    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:
        
        # Mean and STD
        avg = img_in1.mean(dim = (2, 3), keepdim = True)
        std = img_in1.std(dim = (2, 3), keepdim = True, unbiased = False)

        # Z-score
        img_in1 = (img_in1 - avg) / (std + 1e-5)

        # Scale and Shift
        img_out = (self.scale * img_in1) + self.shift

        return img_out
    

"""
========================================================================================================================
Dynamic Tanh Block
========================================================================================================================
"""
class DyTBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()
        
        # Scale and Shift
        self.alpha = Parameter(torch.ones(1, 1, 1, 1) * 0.5)
        self.scale = Parameter(torch.ones(1, filters, 1, 1))
        self.shift = Parameter(torch.zeros(1, filters, 1, 1))

        # Tanh Activation
        self.block0 = Tanh()

        return
    
    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:
        
        # Dynamic Tanh
        img_out = self.block0(self.alpha * img_in1)

        # Scale and Shift
        img_out = (self.scale * img_out) + self.shift

        return img_out


"""
========================================================================================================================
Attention Block
========================================================================================================================
"""
class AttentionBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()

        """
        ----------------------------------------------------------------------------------------------------------------
        Channel Attention
        ----------------------------------------------------------------------------------------------------------------
        """
        # Pooling
        self.block0_1 = AdaptiveAvgPool2d(1)
        self.block0_2 = AdaptiveMaxPool2d(1)

        # Shared MLP
        self.block1 = Conv2d(filters, filters // 4, kernel_size = 1)
        self.block2 = GELU()
        self.block3 = Conv2d(filters // 4, filters, kernel_size = 1)

        # Activation
        self.block5 = Sigmoid()

        """
        ----------------------------------------------------------------------------------------------------------------
        Spatial Attention
        ----------------------------------------------------------------------------------------------------------------
        """
        # Conv
        self.block6 = Conv2d(2, 1, kernel_size = 7, padding = 3, padding_mode = 'replicate')

        # Activation
        self.block7 = Sigmoid()

        return

    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:

        """
        ----------------------------------------------------------------------------------------------------------------
        Channel Attention
        ----------------------------------------------------------------------------------------------------------------
        """
        # Average Pooling
        img_out1 = self.block0_1(img_in1)
        img_out1 = self.block1(img_out1)
        img_out1 = self.block2(img_out1)
        img_out1 = self.block3(img_out1)

        # Maximum Pooling
        img_out2 = self.block0_2(img_in1)
        img_out2 = self.block1(img_out2)
        img_out2 = self.block2(img_out2)
        img_out2 = self.block3(img_out2)

        # Activation
        img_out = self.block5(img_out1 + img_out2)

        # Attention
        img_in2 = img_in1 * img_out

        """
        ----------------------------------------------------------------------------------------------------------------
        Spatial Attention
        ----------------------------------------------------------------------------------------------------------------
        """
        # Average and Maximum Pooling
        img_out = torch.concat((img_in2.amax(dim = 1, keepdim = True), img_in2.mean(dim = 1, keepdim = True)), dim = 1)

        # Conv
        img_out = self.block6(img_out)

        # Activation
        img_out = self.block7(img_out)

        # Attention
        img_out = img_in2 * img_out

        return img_out