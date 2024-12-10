import random
from typing import Tuple
import numpy as np
import numbers
import collections
from PIL import Image

from torchvision.transforms import functional as F
import cv2
from collections.abc import Sequence
import torch

np.random.seed(0)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, label):
        if isinstance(label, np.ndarray):
            return F.to_tensor(image), torch.from_numpy(label).long()
        else:
            return F.to_tensor(image), torch.from_numpy(np.array(label)).long()


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, label):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, resize_label=True):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        image = F.resize(image, self.size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray):
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                label = F.resize(label, self.size, Image.NEAREST)
        return image, label


class RandomScale(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, scale, size=None, resize_label=True):
        assert isinstance(scale, collections.Iterable)
        if size is not None:
            assert isinstance(size, collections.Iterable) and len(size) == 2
        self.scale = scale
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label):
        h, w = image.shape[0], image.shape[1]
        if self.size:
            h, w = self.size
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        size = (int(h * temp_scale), int(w * temp_scale))
        image = F.resize(image, size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray):
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                label = F.resize(label, size, Image.NEAREST)
        return image, label


class RandomCrop:
    def __init__(self, crop_size, pad_if_needed=False, fill=0, label_fill=255, padding_mode='constant'):
        """
        Args:
            size (int or tuple): Crop size (height, width).
            pad_if_needed (bool): Whether to pad if the image is smaller than the crop size.
            fill (int): Fill value for image padding.
            label_fill (int): Fill value for label padding.
            padding_mode (str): Padding mode for image. One of 'constant', 'edge', 'reflect', or 'symmetric'.
        """
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.label_fill = label_fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get crop parameters."""
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, label):
        """Apply random crop."""
        # Apply padding to fit for the crop size
        if self.pad_if_needed:
            h, w = label.shape
            pad_h = max(self.crop_size[0] - h, 0)
            pad_w = max(self.crop_size[1] - w, 0)
            if pad_h > 0 or pad_w > 0:
                image = F.pad(image, (0, 0, pad_w, pad_h), self.fill, self.padding_mode)
                label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=self.label_fill)

        if image.size[0] < self.crop_size[1] or image.size[1] < self.crop_size[0]:
            raise ValueError(f"Image size {image.size} is smaller than crop size {self.crop_size} even after padding.")

        # Perform random cropping
        i, j, h, w = self.get_params(image, self.crop_size)
        image = F.crop(image, i, j, h, w)
        label = label[i:i + h, j:j + w]

        return image, label

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.crop_size}, padding={self.padding})'


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.hflip(image)
            if isinstance(label, np.ndarray):
                label = cv2.flip(label, 1)
            else:
                label = F.hflip(label)
        return image, label


class RandomScale(object):
    def __init__(self, scale: Tuple[float, float]):
        self.scale = scale

    def __call__(self, image, label):
        h, w = label.shape
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        h, w = int(h * scale_factor), int(w * scale_factor)
        image = F.resize(image, (h, w), Image.BICUBIC)
        label = F.resize(Image.fromarray(label), (h, w), Image.NEAREST)
        label = np.array(label, dtype=np.int64)
        return image, label