import os

import numpy as np
from PIL import Image
from torch.utils import data


class cityscapesDataSet(data.Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        transform=None,
        ignore_label=255,
        cfg=None,
    ):
        self.split = split
        self.data_root = data_root
        self.cfg = cfg

        if self.split == "train":
            data_list = os.path.join(data_root, "cityscapes_train_list.txt")
        else:
            data_list = os.path.join(data_root, "cityscapes_val_list.txt")

        with open(data_list, "r") as handle:
            content = handle.readlines()

        self.data_list = []
        for fname in content:
            name = fname.strip()
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.data_root, "leftImg8bit/%s/%s" % (self.split, name)
                    ),
                    "label": os.path.join(
                        self.data_root,
                        "gtFine/%s/%s"
                        % (
                            self.split,
                            name.split("_leftImg8bit")[0] + "_gtFine_labelIds.png",
                            # name + "_gtFine_labelIds.png",
                        ),
                    ),
                }
            )

        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }

        self.transform = transform
        self.ignore_label = ignore_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert("RGB")
        label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = np.array(label_copy, dtype=np.uint8)
        label.resize(label.shape[0], label.shape[1], 1)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label
