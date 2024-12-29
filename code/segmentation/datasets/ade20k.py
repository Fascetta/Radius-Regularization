import json
import os
import pickle as pkl

import cv2
import matplotlib._color_data as mcd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_NUMERALS = "0123456789abcdefABCDEF"
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = "x", "X"


def rgb(triplet):
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]


def loadAde20K(file):
    fileseg = file.replace(".jpg", "_seg.png")
    with Image.open(fileseg) as io:
        seg = np.array(io)

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:, :, 0]
    G = seg[:, :, 1]
    B = seg[:, :, 2]
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat

    level = 0
    PartsClassMasks = []
    PartsInstanceMasks = []
    while True:
        level = level + 1
        file_parts = file.replace(".jpg", "_parts_{}.png".format(level))
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io)
            R = partsseg[:, :, 0]
            G = partsseg[:, :, 1]
            B = partsseg[:, :, 2]
            PartsClassMasks.append((np.int32(R) / 10) * 256 + np.int32(G))
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks

        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace(".jpg", ".json")
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, "r") as f:
            input_info = json.load(f)

        contents = input_info["annotation"]["object"]
        instance = np.array([int(x["id"]) for x in contents])
        names = [x["raw_name"] for x in contents]
        corrected_raw_name = [x["name"] for x in contents]
        partlevel = np.array([int(x["parts"]["part_level"]) for x in contents])
        ispart = np.array([p > 0 for p in partlevel])
        iscrop = np.array([int(x["crop"]) for x in contents])
        listattributes = [x["attributes"] for x in contents]
        polygon = [x["polygon"] for x in contents]
        for p in polygon:
            p["x"] = np.array(p["x"])
            p["y"] = np.array(p["y"])

        objects["instancendx"] = instance[ispart == 0]
        objects["class"] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects["corrected_raw_name"] = [
            corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])
        ]
        objects["iscrop"] = iscrop[ispart == 0]
        objects["listattributes"] = [
            listattributes[x] for x in list(np.where(ispart == 0)[0])
        ]
        objects["polygon"] = [polygon[x] for x in list(np.where(ispart == 0)[0])]

        parts["instancendx"] = instance[ispart == 1]
        parts["class"] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts["corrected_raw_name"] = [
            corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])
        ]
        parts["iscrop"] = iscrop[ispart == 1]
        parts["listattributes"] = [
            listattributes[x] for x in list(np.where(ispart == 1)[0])
        ]
        parts["polygon"] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {
        "img_name": file,
        "segm_name": fileseg,
        "class_mask": ObjectClassMasks,
        "instance_mask": ObjectInstanceMasks,
        "partclass_mask": PartsClassMasks,
        "part_instance_mask": PartsInstanceMasks,
        "objects": objects,
        "parts": parts,
    }


def plot_polygon(img_name, info, show_obj=True, show_parts=False):
    colors = mcd.CSS4_COLORS
    color_keys = list(colors.keys())
    all_objects = []
    all_poly = []
    if show_obj:
        all_objects += info["objects"]["class"]
        all_poly += info["objects"]["polygon"]
    if show_parts:
        all_objects += info["parts"]["class"]
        all_poly += info["objects"]["polygon"]

    img = cv2.imread(img_name)
    thickness = 5
    for it, (obj, poly) in enumerate(zip(all_objects, all_poly)):
        curr_color = colors[color_keys[it % len(color_keys)]]
        pts = np.concatenate([poly["x"][:, None], poly["y"][:, None]], 1)[None, :]
        color = rgb(curr_color[1:])
        img = cv2.polylines(img, pts, True, color, thickness)
    return img


class ADE20KDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        super(ADE20KDataset, self).__init__()
        self.root = root
        self.transform = transform
        if split == "train":
            self.split = "training"
        else:
            self.split = "validation"
        self.load_data()
        print(f"Loaded {len(self)} images")

    def load_data(self):
        index_path = os.path.join(self.root, "ADE20K_2021_17_01", "index_ade20k.pkl")

        with open(index_path, "rb") as f:
            self.index = pkl.load(f)

        self.filenames = self.index["filename"]
        self.filefolders = self.index["folder"]
        self.filepaths = [
            os.path.join(self.root, f, n)
            for f, n in zip(self.filefolders, self.filenames)
        ]

        # load the RGB to class mapping
        mapping_file = os.path.join(
            self.root, "ADE20K_2021_17_01", "rgb_to_class_mapping.csv"
        )
        self.rgb_to_class = {}
        with open(mapping_file, "r") as f:
            # skip the header
            f.readline()
            # example line: 133,12,255,0
            for line in f:
                r, g, b, class_idx = line.strip().split(",")
                rgb = (int(r), int(g), int(b))
                self.rgb_to_class[rgb] = int(class_idx)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        info = loadAde20K(filepath)
        img = cv2.imread(info["img_name"])[:, :, ::-1]
        seg_mask = cv2.imread(info["segm_name"])[:, :, ::-1]

        # Convert image to PIL format
        img = Image.fromarray(img).convert("RGB")
        seg_mask = seg_mask.astype(np.uint8)

        # map RGB to class index
        for rgb, class_idx in self.rgb_to_class.items():
            seg_mask[(seg_mask == rgb).all(axis=-1)] = class_idx

        if self.transform is not None:
            img, seg_mask = self.transform(img, seg_mask)

        return img, seg_mask
