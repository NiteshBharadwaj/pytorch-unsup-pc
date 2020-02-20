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
        print(fname)
        with open(fname, 'rb') as f:
            input = {}
            feature = pickle.load(f)
            input['image'] = feature['image'].transpose(0,3,1,2)
            input['mask'] = feature['mask'].transpose(0,3,1,2)
            if self.cfg.saved_camera:
                input['extrinsic'] = feature['extrinsic']
                input['cam_pos'] = feature['cam_pos']
            if self.cfg.saved_depth:
                input['depth'] = feature['depth']
            return input
