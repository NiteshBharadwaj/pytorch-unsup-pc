#!/usr/bin/env python

import startup
import pdb

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from models import model_pc_to as model_pc
from run.ShapeRecords import ShapeRecords
from util.app_config import config as app_config
from util.system import setup_environment
#from util.train import get_trainable_variables, get_learning_rate
#from util.losses import regularization_loss
from util.fs import mkdir_if_missing
#from util.data import tf_record_compression
#
# def parse_tf_records(cfg, serialized):
#     num_views = cfg.num_views
#     image_size = cfg.image_size
#
#     # A dictionary from TF-Example keys to tf.FixedLenFeature instance.
#     features = {
#         'image': tf.FixedLenFeature([num_views, image_size, image_size, 3], tf.float32),
#         'mask': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32),
#     }
#
#     if cfg.saved_camera:
#         features.update(
#             {'extrinsic': tf.FixedLenFeature([num_views, 4, 4], tf.float32),
#              'cam_pos': tf.FixedLenFeature([num_views, 3], tf.float32)})
#     if cfg.saved_depth:
#         features.update(
#             {'depth': tf.FixedLenFeature([num_views, image_size, image_size, 1], tf.float32)})
#
#     return tf.parse_single_example(serialized, features)

import numpy as np
def train():
        cfg = app_config

        setup_environment(cfg)
        o = np.ones(3)
        z = np.zeros(3)
        v = np.stack([o, z])
        np.random.seed(0)
        cam = torch.from_numpy(np.random.random((128, 4))).float()
        myarr = torch.from_numpy(np.random.random((128,140,3))).float()
        scaling_factor = torch.from_numpy(np.random.random((128,1))).float()
        pc = myarr
        global_step=0

        pc.requires_grad=True
        from util.point_cloud_to import smoothen_voxels3d, pointcloud_project_fast
        model = model_pc.ModelPointCloud(cfg)
        _sigma_rel, _gauss_sigma, _gauss_kernel = model.setup_sigma(None,global_step)


        pc_out = pointcloud_project_fast(cfg,pc,cam,None,None, _gauss_kernel, scaling_factor=scaling_factor)
        proj = pc_out['proj']
        vx = pc_out['voxels']
        tr_pc = pc_out['tr_pc']
        drc_pr = pc_out['drc_probs']
        dp = pc_out['proj_depth']
        print('Proj ',proj.sum())
        print('Voxels ',vx.sum())
        print('Tr pc ',tr_pc.sum())
        print('DRC ',drc_pr.sum())
        print('Depth', dp.sum())
        import pdb
        pdb.set_trace()


def main():
    train()


if __name__ == '__main__':
    main()
