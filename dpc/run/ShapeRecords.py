import torch
from torch.utils import data
import os
import pickle
import cv2
import numpy as np
import random

class ShapeRecords(data.Dataset):

    def __init__(self, dataset_folder, cfg, split):
        self.dataset_folder = dataset_folder
        self.cfg = cfg
        self.file_names = []
        self.split = split
        self.split_file_name = os.path.join(dataset_folder,'../splits/{}_{}.txt'.format(cfg.synth_set,split))
        with open(self.split_file_name) as f:
            split_lines = f.readlines()
        for filename in os.listdir(dataset_folder):
            if filename.endswith(".p"):
                for line in split_lines:
                    if line.split('\n')[0] in filename:
                        self.file_names.append(filename)
        random.shuffle(self.file_names)
        self.file_names = self.file_names[:cfg.num_dataset_samples]
        print('Initialized dataset {} with size {}'.format(split,len(self.file_names)))
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        fname = os.path.join(self.dataset_folder, self.file_names[index])
        with open(fname, 'rb') as f:
            input = {}
            feature = pickle.load(f)
            image = feature['image']
            mask = feature['mask']
            if feature['image'].shape[1]!=self.cfg.image_size:
                image_res = np.zeros((image.shape[0],self.cfg.image_size, self.cfg.image_size,image.shape[3]),dtype=np.float32)
                mask_res = np.zeros((mask.shape[0],self.cfg.image_size,self.cfg.image_size,mask.shape[3]),dtype=np.float32)
                for i in range(image.shape[0]):
                    image_res[i] = cv2.resize(image[i],(self.cfg.image_size,self.cfg.image_size))
                    mask_res[i] = cv2.resize(mask[i],(self.cfg.image_size,self.cfg.image_size)).reshape(self.cfg.image_size,self.cfg.image_size,1)
                input['image'] = image_res.transpose(0, 3, 1, 2)
                input['mask'] = mask_res.transpose(0, 3, 1, 2)
            else:
                input['image'] = image.transpose(0, 3, 1, 2)
                input['mask'] = mask.transpose(0, 3, 1, 2)
            if self.cfg.saved_camera:
                input['extrinsic'] = feature['extrinsic']
                input['cam_pos'] = feature['cam_pos']
            if self.cfg.saved_depth:
                import pdb
                pdb.set_trace()
                input['depth'] = feature['depth']
            return input
