import os
import sys

import numpy as np
import scipy.io

import sys
sys.path.append("../../dpc/")
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

torch.cuda.empty_cache() 

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


def get_group(pos,divs):
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

divs = 4
cfg = app_config
exp_dir = cfg.checkpoint_dir
num_views = cfg.num_views
eval_unsup = cfg.eval_unsupervised_shape
dataset_folder = cfg.inp_dir

gt_dir = os.path.join(cfg.gt_pc_dir, cfg.synth_set)

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
all_images = np.zeros((0,128,128,3))
chamfer_all = np.zeros((0,2))
for j in range(np.power(divs, 3)):
    chamfer_dict[j] = np.zeros((0,2))
    images_dict[j] = np.zeros((0,128,128,3))

for k in range(150):
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
        images = sample['image'][i].transpose(1,2,0)
        g = get_group(campos, divs)

     #   chamfer_dict[g] = np.concatenate((chamfer_dict[g], np.expand_dims(chamfer_dists_current, 0)))
     #   images_dict[g] = np.concatenate((images_dict[g], np.expand_dims(images,0)))
        all_images = np.concatenate((all_images, np.expand_dims(images,0)))
        chamfer_all = np.concatenate((chamfer_all, np.expand_dims(chamfer_dists_current, 0)))
        #print(i, ":", chamfer_dists_current)

    # current_mean = np.mean(chamfer_dists_current, 0)
    # print("total:", current_mean)        


images_dict_cleaned = {}
chamfer_cleaned = {}
for k in images_dict.keys():
    if not images_dict[k].any():
        print("No objects grouped under group",k)
    else:
        #images_dict_cleaned[k] = images_dict[k]
        chamfer_cleaned[k] = chamfer_dict[k]
chamfer_dict = chamfer_cleaned
images_dict = images_dict_cleaned
for key in chamfer_dict:
        print(key, np.mean(chamfer_dict[key],0)*100)

from sklearn.decomposition import PCA
images_val = all_images.reshape(all_images.shape[0],-1)
pca = PCA(50)
pca_image = pca.fit_transform(images_val)
pca_image = pca_image.reshape(-1,50)
print('Explained variance {}'.format(pca.explained_variance_ratio_.sum()))
all_images_pca = pca_image
images_pca_dict = {}
for k in images_dict:
    print("Key:",k)
    images_val = images_dict[k].reshape(images_dict[k].shape[0],-1)
    pca = PCA(32)
    pca_image = pca.fit_transform(images_val)
    pca_image = pca_image.reshape(-1,32)
    print('Explained variance {}'.format(pca.explained_variance_ratio_.sum()))
    images_pca_dict[k] = pca_image


from sklearn.cluster import KMeans
n_clusters = 5
kmeans_all = KMeans(n_clusters=n_clusters,init='k-means++',random_state=0).fit(all_images_pca)

from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

all_centers = kmeans_all.cluster_centers_
sort_dist = {}
selected_indices = []
#pick 15*5 = 75 images
for ctr in all_centers:
       keep_keys=15
       min_keys = []
       for pca_id in range(len(all_images_pca)):
           sort_dist[pca_id] = np.sqrt(np.sum(preprocessing.normalize(euclidean_distances(all_images_pca[pca_id].reshape(-1,1),ctr.reshape(-1,1)),norm='l2')))
       while keep_keys>=0:
            min_value = min(sort_dist.values())
            min_keys = [ ke for ke in sort_dist if sort_dist[ke] == min_value]
            selected_indices = selected_indices + min_keys
            for mke in min_keys:
                del sort_dist[mke]
                keep_keys-=1

cnt = 0
f = open("all_pca_images.txt","a")
for k in range(150):
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
        if cnt in selected_indices:
                #print("{}/{}".format(k, num_models))
                #print(model_names[k])
                f.write(model_names[k]+':'+str(i)+':'+'\n')

        cnt += 1



kmeans_dict = {}
for k in images_pca_dict:
    kmeans_dict[k] = KMeans(n_clusters=n_clusters,init='k-means++',random_state=0).fit(images_pca_dict[k])

    cpose_all_centers = kmeans_dict.cluster_centers_
    cpose_sort_dist = {}
    cpose_selected_indices = []
    #pick 15*5 = 75 images
    for ctr in cpose_all_centers:
           keep_keys=15
           min_keys = []
           for pca_id in range(len(images_pca_dict[k)):
               cpose_sort_dist[pca_id] = np.sqrt(np.sum(preprocessing.normalize(euclidean_distances(images_pca_dict[pca_id].reshape(-1,1),ctr.reshape(-1,1)),norm='l2')))
           while keep_keys>=0:
                cp_min_value = min(cpose_sort_dist.values())
                cpose_min_keys = [ ke for ke in cpose_sort_dist if cpose_sort_dist[ke] == min_value]
                cpose_selected_indices = cpose_selected_indices + cpose_min_keys
                for mke in cpose_min_keys:
                    del cpose_sort_dist[mke]
                    keep_keys-=1

cp 
cnt = 0
f = open("cpose_pca_images.txt","a")
for k in range(150):
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
        if cnt in cpose_selected_indices:
                #print("{}/{}".format(k, num_models))
                #print(model_names[k])
                cp.write(model_names[k]+':'+str(i)+':'+'\n')

