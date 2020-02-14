import torch
from torch.utils import data
import os
import pickle


class ShapeRecords(data.Dataset):

    def __init__(self, dataset_folder, cfg=None):
        self.dataset_folder = dataset_folder
        self.cfg = cfg
        self.file_names = []
        for filename in os.listdir(dataset_folder):
            if filename.endswith(".p"):
                self.file_names.append(filename)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        fname = os.path.join(self.dataset_folder, self.file_names[index])
        with open(fname, 'rb') as f:
            feature = pickle.load(f)
            image = feature['image']
            mask = feature['mask']
            if self.cfg.saved_camera:
                extrinsic = feature['extrinsic']
                cam_pos = feature['cam_pos']
            if self.cfg.saved_depth:
                depth = feature['depth']
            return image, mask, extrinsic, cam_pos, depth

