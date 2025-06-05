"""
========================================================================================================================
Package
========================================================================================================================
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import math

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import Conv2d
from torch.nn import GELU, Tanh
from torchsummary import summary

from Model.Sub_Module import LayerNorm, AttentionBlock
    

"""
========================================================================================================================
Dilated Block
========================================================================================================================
"""
class DilatedBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int, stage: int = 0) -> None:

        super().__init__()

        # Block Length
        self.len = min(math.floor(math.log2(filters)), 6)

        # Dilation Rate List
        dilations = [1, 3, 5, 8, 13, 21][:self.len]

        # Dynamic Rotation
        dilations = [1] + dilations[1:][stage % (self.len - 1):] + dilations[1:][:stage % (self.len - 1)]

        # Filter List
        filters = [filters] + [filters // pow(2, i + 1) for i in range(self.len - 1)] + [filters // pow(2, self.len - 1)]

        # Block
        block = []
        for i in range(self.len):
            # 
            block.append(Conv2d(filters[i], filters[i + 1], kernel_size = 3, dilation = dilations[i], padding = dilations[i], padding_mode = 'replicate'))

        # Reset Filter Number
        filters = filters[0]

        # Block
        self.block0 = ModuleList(block)
        self.block1 = LayerNorm(filters)
        self.block2 = Conv2d(filters, filters, kernel_size = 1)
        self.block3 = GELU()
        self.block4 = Conv2d(filters, filters, kernel_size = 1)

        return

    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:

        img_out = [img_in1]

        for i in range(self.len):
            img_out.append(self.block0[i](img_out[i]))

        # Concatenation
        img_out = torch.cat(img_out[1:], dim = 1)

        img_out = self.block1(img_out)
        img_out = self.block2(img_out)
        img_out = self.block3(img_out)
        img_out = self.block4(img_out)

        img_out = self.shuffle(img_out, groups = 4)

        # Skip Connection (Addition)
        img_out = img_out + img_in1
        # img_out = (img_out + img_in1) / 2

        return img_out
    
    """
    ====================================================================================================================
    Channel Shuffle (Inspired by ShuffleNet)
    ====================================================================================================================
    """
    def shuffle(self, img_in1: Tensor, groups: int = 4) -> Tensor:

        # Batch, Channel, Height, Width
        batch, channel, height, width = img_in1.size()

        # Check Compatibility
        assert channel % groups == 0

        # Shuffle
        img_out = img_in1.reshape(batch, groups, channel // groups, height, width)
        img_out = img_out.permute(0, 2, 1, 3, 4)
        img_out = img_out.reshape(batch, channel, height, width)

        return img_out
    

"""
========================================================================================================================
Initialization Block
========================================================================================================================
"""
class InitBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters) -> None:

        super().__init__()

        # 
        self.block0 = Conv2d(7, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block1 = LayerNorm(filters)
        self.block2 = DilatedBlock(filters)

        return

    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:
        
        img_out = self.block0(img_in1)
        img_out = self.block1(img_out)
        img_out = self.block2(img_out)

        return img_out


"""
========================================================================================================================
Isotropic Block
========================================================================================================================
"""
class IsotropicBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int, stage: int = 0) -> None:

        super().__init__()

        #
        self.block0 = DilatedBlock(filters, stage = stage)
        self.block1 = AttentionBlock(filters)

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
Final Block
========================================================================================================================
"""
class FinalBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()

        # 
        self.block0 = Conv2d(filters, 1, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block1 = Tanh()
        
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
Isotropic Model + Dilated CNN
========================================================================================================================
"""
class Iso_Dilate_Shuffle(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, width: int = 32) -> None:

        super().__init__()

        # Model Width
        self.width = width

        # Encode
        self.encode0 = InitBlock(width)

        self.encode1 = IsotropicBlock(width, stage = 0)
        self.encode2 = IsotropicBlock(width, stage = 1)
        self.encode3 = IsotropicBlock(width, stage = 2)
        self.encode4 = IsotropicBlock(width, stage = 3)

        self.encode5 = FinalBlock(width)

        return
    
    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor) -> Tensor:

        # Encode
        encode0 = self.encode0(img_in1)
        encode1 = self.encode1(encode0)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        encode5 = self.encode5(encode4)

        return encode5

"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    model = Iso_Dilate_Shuffle(32).to(torch.device('cuda'))

    print(summary(model, input_size = (7, 256, 256), batch_size = 16))
