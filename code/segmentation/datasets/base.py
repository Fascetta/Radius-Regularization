#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data


class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(self, root, split, transform=None, ignore_label=255):
        self.root = root
        self.split = split
        self.transform = transform
        self.ignore_label = ignore_label

        self.files = []
        self._set_files()

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        image, label = self._load_data(index)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
