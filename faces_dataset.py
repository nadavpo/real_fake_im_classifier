"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if index < len(self.real_image_names):
            im_path = self.real_image_names[index]
            im_type = 'real'
            label = 0
        else:
            im_path = self.fake_image_names[index-len(self.real_image_names)]
            label = 1
            im_type = 'fake'
            
        im = Image.open(os.path.join(self.root_path,im_type,im_path))
        if self.transform is not None:
            im = self.transform(im)
            
        return im, label

    def __len__(self):
        return len(self.real_image_names)+len(self.fake_image_names)

