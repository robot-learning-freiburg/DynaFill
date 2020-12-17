
from pathlib import Path
from functools import partial
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose


class DynaFillSingleFrame(Dataset):

    FOV = 90 # degrees
    IMAGE_SIZE = 256
    MAX_DEPTH = 1000 # m
    DYNAMIC_CLASSES = {
        4, # Pedestrian
        10 # Vehicle
    }

    preprocess_rgb = Compose([
        Resize(IMAGE_SIZE, Image.BILINEAR),
        ToTensor(),
        lambda img: img * 2 - 1 # [0, 1] -> [-1, 1]
    ])

    preprocess_depth = Compose([
        torch.from_numpy,
        lambda img: img.unsqueeze(0).unsqueeze(0),
        partial(F.interpolate, size=(IMAGE_SIZE,) * 2),
        lambda img: img.squeeze(0)
    ])

    preprocess_semseg = Resize(IMAGE_SIZE)

    preprocess_mask = Compose([
        lambda img: cv2.dilate(
            img.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
            iterations=1
        ),
        torch.from_numpy,
        lambda img: img.unsqueeze(0)
    ])

    def __init__(self, directory):
        self.directory = Path(directory)
        self.paths_rgb = sorted((self.directory / "rgb" / "dynamic").glob("*"))
        self.paths_depth = sorted((self.directory / "depth" / "dynamic").glob("*"))
        self.paths_semseg = sorted((self.directory / "semseg" / "dynamic").glob("*"))
        assert len(self.paths_rgb) == len(self.paths_depth) == len(self.paths_semseg)
        self.poses = pd.read_csv(self.directory / "poses.csv").set_index("id")

        # Camera intrisic matrix
        f = self.IMAGE_SIZE / (2 * np.tan(self.FOV / 360 * np.pi))
        cx = self.IMAGE_SIZE / 2
        cy = cx
        self.K = torch.Tensor([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1]
        ])
        self.camera_local = torch.Tensor([2.0, 0.0, 1.8, 0, 0, 0])

    def semseg_to_mask(self, semseg):
        semseg = np.asarray(semseg)
        mask = np.zeros_like(semseg, dtype=np.bool_)
        for class_id in self.DYNAMIC_CLASSES:
            mask |= (semseg == class_id)
        return mask

    def __getitem__(self, idx):
        path_rgb = self.paths_rgb[idx]
        path_depth = self.paths_depth[idx]
        path_semseg = self.paths_semseg[idx]
        assert path_rgb.stem == path_depth.stem == path_semseg.stem

        # Load
        rgb = Image.open(path_rgb).convert("RGB")
        depth = np.load(path_depth) * self.MAX_DEPTH
        pose = self.poses.loc[path_rgb.stem]
        semseg = Image.open(path_semseg)
        mask = self.semseg_to_mask(self.preprocess_semseg(semseg))

        # Preprocess
        rgb = self.preprocess_rgb(rgb).float()
        depth = self.preprocess_depth(depth).float()
        mask = self.preprocess_mask(mask).float()
        pose = torch.from_numpy(pose.to_numpy()).float()

        return dict(
            rgb=rgb,
            depth=depth,
            mask=mask,
            pose=pose
        )

    def __len__(self):
        return len(self.paths_rgb)

class DynaFillTrajectory(DynaFillSingleFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectories = defaultdict(list)
        for idx, path_rgb in enumerate(self.paths_rgb):
            idx_trajectory = int(path_rgb.stem.split("_")[0])
            self.trajectories[idx_trajectory].append(idx)

    def __getitem__(self, idx):
        idxs = self.trajectories[idx]
        frames = list(map(super().__getitem__, idxs))
        frames = {key: torch.stack(tuple(frame[key] for frame in frames), dim=0) for key in frames[0].keys()}
        return frames

    def __len__(self):
        return len(self.trajectories)

