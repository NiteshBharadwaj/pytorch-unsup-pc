#!/usr/bin/env python

import startup

import os

import numpy as np
import scipy.io

from util.point_cloud import point_cloud_distance
from util.simple_dataset import Dataset3D
from util.app_config import config as app_config
from util.tools import partition_range, to_np_object
from util.quaternion import quaternion_rotate
from util.euler import ypr_from_campos

import torch
from torch.utils.tensorboard import SummaryWriter
from models import model_pc_to as model_pc
from util.system import setup_environment
from run.ShapeRecords import ShapeRecords
import pickle
import pdb


def compute_distance(cfg, source_np, target_np):
    """
    compute projection from source to target
    """
    num_parts = cfg.pc_eval_chamfer_num_parts
    partition = partition_range(source_np.shape[0], num_parts)
    min_dist_np = np.zeros((0,))
    idx_np = np.zeros((0,))
    source_pc = torch.from_numpy(source_np).cuda()
    target_pc = torch.from_numpy(target_np).cuda()
    for k in range(num_parts):
        r = partition[k, :]
        src = source_pc[r[0]:r[1]]
        _, min_dist, min_idx = point_cloud_distance(src, target_pc)
        min_dist_0_np = min_dist.cpu().numpy()
        idx_0_np = min_idx.cpu().numpy()
        min_dist_np = np.concatenate((min_dist_np, min_dist_0_np), axis=0)
        idx_np = np.concatenate((idx_np, idx_0_np), axis=0)

    return min_dist_np, idx_np

def get_group(pos):

    divs = 2
    scale = divs/2
    yaw, pitch, roll = ypr_from_campos(pos[0], pos[1], pos[2])
    yaw = yaw + np.pi

    # get everything from 0 to 2*pi
    yaw = yaw%(2*np.pi)+0.00000001
    pitch = pitch%(2*np.pi)+0.00000001
    roll = roll%(2*np.pi) + 0.00000001

    q1 = np.ceil(scale*yaw/np.pi)-1
    q2 = np.ceil(scale*pitch/np.pi)-1
    q3 = np.ceil(scale*roll/np.pi)-1

    return q1*np.square(divs)+q2*divs+q3


def run_eval():
    divs = 2
    cfg = app_config
    exp_dir = cfg.checkpoint_dir
    num_views = cfg.num_views
    eval_unsup = cfg.eval_unsupervised_shape
    dataset_folder = cfg.inp_dir

    gt_dir = os.path.join(cfg.gt_pc_dir, cfg.synth_set)

    # g = tf.Graph()
    # with g.as_default():
    #    source_pc = tf.placeholder(dtype=tf.float64, shape=[None, 3])
    #    target_pc = tf.placeholder(dtype=tf.float64, shape=[None, 3])
    #    quat_tf = tf.placeholder(dtype=tf.float64, shape=[1, 4])

    #    _, min_dist, min_idx = point_cloud_distance(source_pc, target_pc)

    #    source_pc_2 = tf.placeholder(dtype=tf.float64, shape=[1, None, 3])
    #    rotated_pc = quaternion_rotate(source_pc_2, quat_tf)

    #    sess = tf.Session(config=config)
    #    sess.run(tf.global_variables_initializer())
    #    sess.run(tf.local_variables_initializer())

    save_pred_name = "{}_{}".format(cfg.save_predictions_dir, cfg.eval_split)
    save_dir = os.path.join(exp_dir, cfg.save_predictions_dir)

    if eval_unsup:
        reference_rotation = scipy.io.loadmat("{}/final_reference_rotation.mat".format(exp_dir))["rotation"]

    dataset = ShapeRecords(dataset_folder, cfg, 'test')

    if cfg.models_list:
        model_names = parse_lines(cfg.models_list)
    else:
        model_names = dataset.file_names
    num_models = len(model_names)

    # making groups for samples and views according to 8 groups of yaw, pitch, roll
    chamfer_dict = {}
    images_dict = {}
    all_images = {}
    for j in range(np.power(divs, 3)):
        chamfer_dict[j] = np.zeros((0,2))
        images_dict[j] = np.zeros((0,3,128,128))
        all_images[j] = np.zeros((0,3,128,128)) 
    
    cnt = 0
    for k in range(num_models):
        sample = dataset.__getitem__(k)

        print("{}/{}".format(k, num_models))
        print(model_names[k])

        gt_filename = "{}/{}.mat".format(gt_dir, model_names[k]).replace('_features.p', '')
        mat_filename = "{}/{}_pc.pkl".format(save_dir, model_names[k])

        if not os.path.isfile(gt_filename) or not os.path.isfile(mat_filename):
            continue

        with open(mat_filename, 'rb') as handle:
            data = pickle.load(handle)
        all_pcs = np.squeeze(data["points"])
        if "num_points" in data:
            all_pcs_nums = np.squeeze(data["num_points"])
            has_number = True
        else:
            has_number = False
        obj = scipy.io.loadmat(gt_filename)
        Vgt = obj["points"]
        
        for i in range(num_views):
            chamfer_dists_current = np.zeros((2), dtype=np.float64)

            pred = all_pcs[i, :, :]
            if has_number:
                pred = pred[0:all_pcs_nums[i], :]

            if eval_unsup:
                pred = np.expand_dims(pred, 0)
                pred = quaternion_rotate(torch.from_numpy(pred).cuda(),
                                         torch.from_numpy(reference_rotation).cuda()).cpu().numpy()
                pred = np.squeeze(pred)

            pred_to_gt, idx_np = compute_distance(cfg, pred, Vgt)
            gt_to_pred, _ = compute_distance(cfg, Vgt, pred)
            chamfer_dists_current[0] = np.mean(pred_to_gt)
            chamfer_dists_current[1] = np.mean(gt_to_pred)

            is_nan = np.isnan(pred_to_gt)
            assert (not np.any(is_nan))

            campos = sample['cam_pos'][i]
            images = sample['image'][i]
            g = get_group(campos)
            chamfer_dict[g] = np.concatenate((chamfer_dict[g], np.expand_dims(chamfer_dists_current, 0)))
            #images_dict[g] = np.concatenate((images_dict[g], np.expand_dims(images,0)))
            #all_images[(k*num_views) +i] = np.concatenate((all_images[(k*num_views) +i],np.expand_dims(images, 0)))
            all_images[cnt] = np.concatenate((all_images[cnt],np.expand_dims(images, 0)))
            cnt+=1
         #current_mean = np.mean(chamfer_dists_current, 0)
         #print("total:", current_mean)


    for key in chamfer_dict:
        print(key, np.mean(chamfer_dict[key],0)*100)
   
    from sklearn.decomposition import PCA 
    out_all_images= np.zeros((1,128,128))
    for k in range(len(all_images)):
        print("Index:",k)
        if not all_images[k].any():
            print("No objects grouped under group",k)
            del all_images[k]
        else:
            images_val = all_images[k].reshape(all_images[k].shape[-1],-1)
            pca = PCA(128)
            pca_image = pca.fit_transform(images_val)
            all_images[k] = pca_image
            out_all_images=np.concatenate((out_all_images,np.expand_dims(all_images[k],0)),axis=0)
    # scipy.io.savemat(os.path.join(exp_dir, "chamfer_{}.mat".format(save_pred_name)),
    #                  {"chamfer": chamfer_dists,
    #                   "model_names": to_np_object(model_names)})
    #
    # file = open(os.path.join(exp_dir, "chamfer_{}.txt".format(save_pred_name)), "w")
    # file.write("{} {}\n".format(final[0], final[1]))
    # file.close()


def eval():
    cfg = app_config

    setup_environment(cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dir = cfg.checkpoint_dir

    split_name = "eval"
    dataset_folder = cfg.inp_dir

    dataset = ShapeRecords(dataset_folder, cfg, 'test')
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg.batch_size, shuffle=cfg.shuffle_dataset,
                                                 num_workers=4, drop_last=True)
    run_eval()

def test_experiment():
    cfg = app_config
    dataset_folder = cfg.inp_dir

    dataset = ShapeRecords(dataset_folder, cfg, 'test')
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg.batch_size, shuffle=cfg.shuffle_dataset,
                                                 num_workers=4, drop_last=True)
    sample = dataset.__getitem__(1)
    campos = sample['cam_pos']
    
    pdb.set_trace()
    
    g = get_group(sample['cam_pos'][i])
    
    yaw, pitch, roll = ypr_from_campos(pos[0], pos[1], pos[2])
    yaw = yaw + np.pi



def main():
    eval()
    #test_experiment()


if __name__ == '__main__':
    # tf.app.run()
    main()
