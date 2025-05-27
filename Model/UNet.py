"""
========================================================================================================================
Package
========================================================================================================================
"""
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d
from torch.nn import BatchNorm2d, LeakyReLU, Tanh
from torchsummary import summary


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
    def __init__(self, filters: int) -> None:

        super().__init__()

        # 
        self.block0 = Conv2d(7, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block1 = BatchNorm2d(filters)
        self.block2 = LeakyReLU(inplace = True)

        self.block3 = Conv2d(filters, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block4 = BatchNorm2d(filters)
        self.block5 = LeakyReLU(inplace = True)

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
        img_out = self.block3(img_out)
        img_out = self.block4(img_out)
        img_out = self.block5(img_out)

        return img_out


"""
========================================================================================================================
Encode Block
========================================================================================================================
"""
class EncodeBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()

        # 
        self.block0 = MaxPool2d(kernel_size = 2, stride = 2)

        self.block1 = Conv2d(filters // 2, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block2 = BatchNorm2d(filters)
        self.block3 = LeakyReLU(inplace = True)

        self.block4 = Conv2d(filters, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block5 = BatchNorm2d(filters)
        self.block6 = LeakyReLU(inplace = True)

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
        img_out = self.block3(img_out)
        img_out = self.block4(img_out)
        img_out = self.block5(img_out)
        img_out = self.block6(img_out)

        return img_out


"""
========================================================================================================================
Bottle Block
========================================================================================================================
"""
class BottleBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()

        #
        self.block0 = MaxPool2d(kernel_size = 2, stride = 2)

        self.block1 = Conv2d(filters // 2, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block2 = BatchNorm2d(filters)
        self.block3 = LeakyReLU(inplace = True)

        self.block4 = Conv2d(filters, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block5 = BatchNorm2d(filters)
        self.block6 = LeakyReLU(inplace = True)

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
        img_out = self.block3(img_out)
        img_out = self.block4(img_out)
        img_out = self.block5(img_out)
        img_out = self.block6(img_out)

        return img_out


"""
========================================================================================================================
Decode Block
========================================================================================================================
"""
class DecodeBlock(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, filters: int) -> None:

        super().__init__()

        #
        self.block0 = ConvTranspose2d(filters * 2, filters, kernel_size = 2, stride = 2)

        self.block1 = Conv2d(filters * 2, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block2 = BatchNorm2d(filters)
        self.block3 = LeakyReLU(inplace = True)

        self.block4 = Conv2d(filters, filters, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.block5 = BatchNorm2d(filters)
        self.block6 = LeakyReLU(inplace = True)


        return
    
    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor, img_in2: Tensor) -> Tensor:

        img_out = self.block0(img_in1)

        img_out = torch.concat((img_out, img_in2), dim = 1)
        
        img_out = self.block1(img_out)
        img_out = self.block2(img_out)
        img_out = self.block3(img_out)
        img_out = self.block4(img_out)
        img_out = self.block5(img_out)
        img_out = self.block6(img_out)

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
Unet
========================================================================================================================
"""
class UNet(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, width: int = 64) -> None:

        super().__init__()

        # Model Width
        self.width = width

        # Filters
        filters = [width * pow(2, i) for i in range(6)]

        # Encode
        self.encode0 = InitBlock(filters[0])
        self.encode1 = EncodeBlock(filters[1])
        self.encode2 = EncodeBlock(filters[2])
        self.encode3 = EncodeBlock(filters[3])
        self.encode4 = EncodeBlock(filters[4])

        # Bottle
        self.bottle5 = BottleBlock(filters[5])

        # Decode
        self.decode4 = DecodeBlock(filters[4])
        self.decode3 = DecodeBlock(filters[3])
        self.decode2 = DecodeBlock(filters[2])
        self.decode1 = DecodeBlock(filters[1])
        self.decode0 = DecodeBlock(filters[0])

        # Ouput
        self.final = FinalBlock(filters[0])

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

        # Bottle
        bottle5 = self.bottle5(encode4)
        
        # Decode
        decode4 = self.decode4(bottle5, encode4)
        decode3 = self.decode3(decode4, encode3)
        decode2 = self.decode2(decode3, encode2)
        decode1 = self.decode1(decode2, encode1)
        decode0 = self.decode0(decode1, encode0)

        # Ouput
        final = self.final(decode0)

        return final


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    model = UNet(32).to(torch.device('cuda'))

    print(summary(model, input_size = (7, 256, 256), batch_size = 16))