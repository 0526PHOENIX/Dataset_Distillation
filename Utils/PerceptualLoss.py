"""
========================================================================================================================
Package
========================================================================================================================
"""
import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Conv2d, MaxPool2d
from torch.nn.functional import tanh

from torchvision import models


"""
========================================================================================================================
Hook-Based Perceptual Loss
========================================================================================================================
"""
class PerceptualLoss(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, device: torch.device = torch.device('cpu')) -> None:

        super().__init__()

        # Device: CPU or GPU
        self.device = device

        # Pretrained VGG19
        model = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1).to(self.device).features
        
        # Ectract Specific Layer: [0, 5, 10, 19]
        self.layer = [0, 5, 10, 19]
        self.model = Sequential(*([temp for layer in self.layer for temp in (model[layer], MaxPool2d(2))][:-1]))

        # Freeze Model Parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Buffer for Perceptual Score
        self.score = []

        # Capture Perceptual Score from Target Layers
        self.hook_layer()

        return

    """
    ====================================================================================================================
    Capture Perceptual Score from Target Layers
    ====================================================================================================================
    """
    def hook_layer(self) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        Capture Perceptual Score from Forward Pass
        ----------------------------------------------------------------------------------------------------------------
        """
        def forward_hook(module: Module, feature_in: Tensor, feature_out: Tensor) -> None:
            
            # Append Perceptual Score to Specific Buffer
            self.score.append(tanh(feature_out.detach()))

            return

        """
        ----------------------------------------------------------------------------------------------------------------
        Register Hook
        ----------------------------------------------------------------------------------------------------------------
        """
        # Register Hook for Every Layer in Model
        for layer in self.model:
            if isinstance(layer, Conv2d):
                layer.register_forward_hook(forward_hook)

        return

    """
    ====================================================================================================================
    Forward Pass
    ====================================================================================================================
    """
    def forward(self, fake_g: Tensor, real_g: Tensor) -> Tensor:
        
        with torch.no_grad():
            
            # Forward Pass: Fake
            self.score.clear()
            self.model(fake_g)
            score_fake = [score for score in self.score]
        
            # Forward Pass: Real
            self.score.clear()
            self.model(real_g)
            score_real = [score for score in self.score]
            
            # Buffer for Perceptual Loss
            perceptual = torch.tensor(0.0, device = self.device)

            # Perceptual Loss
            for fake, real in zip(score_fake, score_real):

                # MAE
                perceptual += torch.abs(fake - real).mean()

            # Normalize to [0, 2]
            perceptual /= len(self.layer)

        return perceptual