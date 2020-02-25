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
        myarr = torch.from_numpy(np.repeat(v, 8960, axis=0).reshape((128, 140, 3)))
        myarr = torch.from_numpy(np.random.random((128,140,3)))
        pc = myarr
        pc.requires_grad=True
        from util.point_cloud_to import pointcloud2voxels3d_fast
        pc_out = pointcloud2voxels3d_fast(cfg, pc, None)
        vx = pc_out[0]
        loss = torch.sum(vx ** 2) / 2.
        vx.register_hook(lambda x: print('output_grads_sum ',x.sum()))
        pc.register_hook(lambda x: print('input_grads_sum ',x.sum()))
        loss.backward()
        print('loss ',loss)
        import pdb
        pdb.set_trace()


def main():
    train()


if __name__ == '__main__':
    main()
