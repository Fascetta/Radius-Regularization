import os.path as osp
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

from datasets.base import _BaseDataset


class CocoStuff10k(_BaseDataset):
    """COCO-Stuff 10k dataset"""

    def __init__(self, **kwargs):
        super(CocoStuff10k, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list via {train, test, all}.txt
        if self.split in ["train", "test", "all"]:
            file_list = osp.join(self.root, "imageLists", self.split + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, "images", image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", image_id + ".mat")
        # Load an image and label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = sio.loadmat(label_path)["S"]
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255
        return image, label


class CocoStuff164k(_BaseDataset):
    """COCO-Stuff 164k dataset"""

    def __init__(self, **kwargs):
        super(CocoStuff164k, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list by parsing the "images" folder
        if self.split in ["train2017", "val2017"]:
            file_list = sorted(glob(osp.join(self.root, "images", self.split, "*.jpg")))
            assert len(file_list) > 0, "{} has no image".format(
                osp.join(self.root, "images", self.split)
            )
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", self.split, image_id + ".png")
        # Load an image and label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        return image_id, image, label