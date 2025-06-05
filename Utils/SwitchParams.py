"""
========================================================================================================================
Package
========================================================================================================================
"""
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import Conv2d

from typing import Generator

from contextlib import contextmanager


"""
========================================================================================================================
Flatten Parameter Module
========================================================================================================================
"""
class SwitchParams(Module):

    """
    ===================================================================================================================
    Get Module from Module Name
    ===================================================================================================================
    """
    def get_module(self, module_name: str) -> Module:
        
        # Edge Case
        if module_name == '':
            return self.model

        # Iterate Through Module Name
        module = self.model
        for child in module_name.split('.'):
            module = getattr(module, child)

        return module

    """
    ===================================================================================================================
    Initialization
    ===================================================================================================================
    """
    def __init__(self, model: Module) -> None:

        super().__init__()

        # Model
        self.model = model

        # Parameter Information Buffer
        self.param_infos = []
        self.param_shape = []
        self.param_numel = []
        self.param_total = 0

        # Parameter Information
        for module_name, module in self.model.named_modules():

            for param_name, param in module.named_parameters(recurse = False):

                self.param_infos.append([module_name, param_name])
                self.param_shape.append(param.size())
                self.param_numel.append(param.numel())
                self.param_total += param.numel()

        # Register Flatten Parameter
        self.flatten_param = Parameter(torch.cat([param.reshape(-1) for param in self.model.parameters()], 0))
        self.model.register_parameter('flatten_param', self.flatten_param)

        # Deregister Other Parameter
        for module_name, param_name in self.param_infos:
            delattr(self.get_module(module_name), param_name)

        # Register the Views as Plain Attributes
        self.unflatten_param(self.flatten_param)

        return
    
    """
    ===================================================================================================================
    Temporarily Assign the Unflatten Parameter
    ===================================================================================================================
    """
    def unflatten_param(self, flatten_param: Tensor) -> None:

        # Unflatten the Flatten Parameter
        unflatten_param = [chunk.view(shape) for (chunk, shape) in zip(flatten_param.split(self.param_numel), self.param_shape)]

        # Temporarily Assign the Unflatten Parameter
        for (module_name, param_name), param in zip(self.param_infos, unflatten_param):
            setattr(self.get_module(module_name), param_name, param)
    
    """
    ===================================================================================================================
    Forward Pass with Unflatten Parameter
    ===================================================================================================================
    """
    @contextmanager
    def forward_unflatten_param(self, flatten_param: Tensor) -> Generator[None, None, None]:
        
        # Save the Original Parameter to Restore
        original_param = [getattr(self.get_module(module_name), param_name) for module_name, param_name in self.param_infos]

        # Temporarily Assign the Unflatten Parameter
        self.unflatten_param(flatten_param)

        yield

        # Restore the Original Parameter
        for (module_name, param_name), param in zip(self.param_infos, original_param):
            setattr(self.get_module(module_name), param_name, param)

        return

    """
    ====================================================================================================================
    Forward
    ====================================================================================================================
    """
    def forward(self, img_in1: Tensor, flatten_param: Tensor) -> Tensor:
        
        # Default Flatten Parameter
        if flatten_param is None:
            flatten_param = self.flatten_param
        
        # Forward Pass with Flatten Parameter
        with self.forward_unflatten_param(flatten_param):
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

    # print()
    # print(model.param_numel)
    # print(model.param_shape)
    # print(model.param_total)
    # print()

    # print()
    # print(model.flatten_param)
    # print()