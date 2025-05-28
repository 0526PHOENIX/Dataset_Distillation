"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
import random
import nibabel as nib

from typing import Literal, Generator

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomAffine as RA
from torchvision.transforms import functional as TF


"""
========================================================================================================================
Dataset
========================================================================================================================
"""
class DS(Dataset):

    """
    ====================================================================================================================
    Initialization and Data Path
    ====================================================================================================================
    """
    def __init__(self,
                 root: str,
                 mode: str | Literal['Train', 'Val', 'Test'],
                 fold: str | Literal['01', '02', '03', '04'] = '01') -> None:

        # Check Mode
        if mode not in ['Train', 'Val', 'Test']:
            raise TypeError('Invalid Mode')
        
        # Critical Parameter
        self.root = root
        self.mode = mode
        self.fold = fold

        # Subject Numbers
        numbers = self.get_numbers()

        # Data Path
        images_path = os.path.join(root, 'MR', 'MR')
        labels_path = os.path.join(root, 'CT', 'CT')
        hmasks_path = os.path.join(root, 'HM', 'HM')
        skulls_path = os.path.join(root, 'SK', 'SK')

        # File List
        self.images, self.labels, self.hmasks, self.skulls = [], [], [], []

        # Get File Path
        for number in numbers:
            # MR File
            for images in sorted(os.listdir(images_path + number)):
                self.images.append(os.path.join(images_path + number, images))
            # CT File
            for labels in sorted(os.listdir(labels_path + number)):
                self.labels.append(os.path.join(labels_path + number, labels))
            # HM File
            for hmasks in sorted(os.listdir(hmasks_path + number)):
                self.hmasks.append(os.path.join(hmasks_path + number, hmasks))
            # SK File
            for skulls in sorted(os.listdir(skulls_path + number)):
                self.skulls.append(os.path.join(skulls_path + number, skulls))
    
        # Check Data Quantity
        if len(self.images) != len(self.labels):
            print(len(self.images), '\t', len(self.labels), '\n')
            raise ValueError('Unequal Amount of Images and Labels.')
        
        return
    
    """
    ====================================================================================================================
    Get Subject Number
    ====================================================================================================================
    """
    def get_numbers(self) -> list[str]:

        # Open File of Specifice Order
        with open(os.path.join(self.root, 'Fold_' + self.fold + '.txt'), 'r') as file:
            lines = file.readlines()

        # Split Out Numerical Part
        if self.mode == 'Train':
            numbers = lines[0].split()[1:]
        elif self.mode == 'Val':
            numbers = lines[1].split()[1:]
        elif self.mode == 'Test':
            numbers = lines[2].split()[1:]

        return numbers

    """
    ====================================================================================================================
    Number of Data
    ====================================================================================================================
    """
    def __len__(self) -> int:
        
        return len(self.images)

    """
    ====================================================================================================================
    Get Data
    ====================================================================================================================
    """
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        # Load MR Data: (7, 256, 256)
        image = torch.from_numpy(nib.load(self.images[index]).get_fdata().astype('float32'))
        
        # Load CT Data: (1, 256, 256)
        label = torch.from_numpy(nib.load(self.labels[index]).get_fdata().astype('float32'))

        # Load HM Data: (1, 256, 256)
        hmask = torch.from_numpy(nib.load(self.hmasks[index]).get_fdata().astype('bool'))

        # Load SK Data: (1, 256, 256)
        skull = torch.from_numpy(nib.load(self.skulls[index]).get_fdata().astype('float32'))

        # Normalize CT Data to [-1, 1]
        label -= -1000
        label /= 4000
        label = (label * 2) - 1
        
        return image, label, hmask, skull


"""
========================================================================================================================
Data Loader
========================================================================================================================
"""
class DL(DataLoader):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, 
                 root: str, 
                 mode: str | Literal['Train', 'Val', 'Test'], 
                 fold: str | Literal['01', '02', '03', '04'] = '01',
                 device: torch.device = torch.device('cpu'), 
                 batch_size: int = 16,
                 shuffle: bool = None,
                 augment: bool = None,
                 *args,
                 **kwargs) -> None:

        # Check Mode
        if mode not in ['Train', 'Val', 'Test']:
            raise TypeError('Invalid Mode')
        
        # Shuffle & Augment or Not
        self.shuffle = shuffle or (mode == 'Train')
        self.augment = augment or (mode == 'Train')
        
        # Device
        self.device = device
        
        # Initiatlization
        dataset = DS(root = root, mode = mode, fold = fold)

        # Parent Class Initialization
        super().__init__(dataset, batch_size, shuffle = (mode == 'Train'), drop_last = False, num_workers = 4, pin_memory = True)

        return

    """
    ====================================================================================================================
    Iterate
    ====================================================================================================================
    """
    def __iter__(self) -> Generator[tuple[Tensor, Tensor, Tensor, Tensor], None, None]:
        
        # Iterate Through DataLoader
        for batch in super().__iter__():
            # Augment or Not
            yield self.augmentation(batch, self.augment)

        return

    """
    ====================================================================================================================
    Data Augmentation
    ====================================================================================================================
    """
    def augmentation(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], augment: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        # Get Batch Data
        image, label, hmask, skull = batch

        # With Aumentation
        if augment:

            # Random Filp
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                hmask = TF.hflip(hmask)
                skull = TF.hflip(skull)
            
            # Rotate + Scalce + Shear
            params = RA.get_params(degrees      = [-3.5, 3.5],
                                   translate    = None,
                                   scale_ranges = [0.7, 1.3],
                                   shears       = [0.97, 1.03],
                                   img_size     = [256, 256])

            # Apply Augmentation
            image = TF.affine(image, *params, fill = -1)
            label = TF.affine(label, *params, fill = -1)
            hmask = TF.affine(hmask, *params, fill = 0)
            skull = TF.affine(skull, *params, fill = -1000)

        # Shift to GPU
        return (image.to(self.device), label.to(self.device), hmask.to(self.device), skull.to(self.device))