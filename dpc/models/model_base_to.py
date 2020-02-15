import numpy as np
import torch.nn as nn
import torch
from util.camera import camera_from_blender, quaternion_from_campos


def pool_single_view(cfg, tensor, view_idx):
    indices = np.arange(cfg.batch_size) * cfg.step_size + view_idx
    return tensor[indices]

def select_2d(data, indices):
    return data[[indices[:,0], indices[:,1]]]

class ModelBase(nn.Module):

    def __init__(self, cfg):
        super(ModelBase, self).__init__()
        self._params = cfg

    def cfg(self):
        return self._params

    # TODO: Heavily depends on correct understanding of masked select and it's relation with gather_nd
    def preprocess(self, raw_inputs, step_size, random_views=True):
        """Selects the subset of viewpoints to train on."""
        cfg = self.cfg()
        var_num_views = cfg.variable_num_views

        num_views = raw_inputs['image'].shape[1]
        quantity = cfg.batch_size
        if cfg.num_views_to_use == -1:
            max_num_views = num_views
        else:
            max_num_views = cfg.num_views_to_use

        inputs = dict()

        def batch_sampler(all_num_views):
            out = np.zeros((0, 2), dtype=np.int64)
            valid_samples = np.zeros((0), dtype=np.float32)
            for n in range(quantity):
                valid_samples_m = np.ones((step_size), dtype=np.float32)
                if var_num_views:
                    num_actual_views = int(all_num_views[n, 0])
                    ids = np.random.choice(num_actual_views, min(step_size, num_actual_views), replace=False)
                    if num_actual_views < step_size:
                        to_fill = step_size - num_actual_views
                        ids = np.concatenate((ids, np.zeros((to_fill), dtype=ids.dtype)))
                        valid_samples_m[num_actual_views:] = 0.0
                elif random_views:
                    ids = np.random.choice(max_num_views, step_size, replace=False)
                else:
                    ids = np.arange(0, step_size).astype(np.int64)

                ids = np.expand_dims(ids, axis=-1)
                batch_ids = np.full((step_size, 1), n, dtype=np.int64)
                full_ids = np.concatenate((batch_ids, ids), axis=-1)
                out = np.concatenate((out, full_ids), axis=0)

                valid_samples = np.concatenate((valid_samples, valid_samples_m), axis=0)

            return out, valid_samples

        num_actual_views = raw_inputs['num_views'] if var_num_views else 0

        indices, valid_samples = batch_sampler(num_actual_views)
        indices = torch.from_numpy(indices)
        valid_samples = torch.from_numpy(valid_samples)

        indices = indices.reshape(step_size*quantity, 2)
        inputs['valid_samples'] = valid_samples.reshape(step_size*quantity)
        inputs['masks'] =  select_2d(raw_inputs['mask'], indices)
        inputs['images'] = select_2d(raw_inputs['image'],indices)
        if cfg.saved_depth:
            inputs['depths'] = select_2d(raw_inputs['depth'],indices)
        inputs['images_1'] = pool_single_view(cfg, inputs['images'], 0)

        def fix_matrix(extr):
            out = np.zeros_like(extr)
            num_matrices = extr.shape[0]
            for k in range(num_matrices):
                out[k, :, :] = camera_from_blender(extr[k, :, :])
            return out

        def quaternion_from_campos_wrapper(campos):
            num = campos.shape[0]
            out = np.zeros([num, 4], dtype=np.float32)
            for k in range(num):
                out[k, :] = quaternion_from_campos(campos[k, :])
            return out

        if cfg.saved_camera:
            matrices = select_2d(raw_inputs['extrinsic'],indices)
            orig_shape = matrices.shape
            extr_tf = fix_matrix(matrices)
            inputs['matrices'] = extr_tf.reshape(orig_shape)

            cam_pos = select_2d(raw_inputs['cam_pos'],indices)
            orig_shape = cam_pos.shape
            quaternion = quaternion_from_campos_wrapper(cam_pos)
            inputs['camera_quaternion'] = quaternion.reshape(orig_shape[0], 4)
        return inputs