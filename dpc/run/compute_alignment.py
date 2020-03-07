import startup

import os

import numpy as np
import scipy.io

import open3d
import pickle
from util.app_config import config as app_config
from util.quaternion_average import quatWAvgMarkley
from util.simple_dataset import Dataset3D
from util.camera import quaternion_from_campos
import torch
from util.quaternion import as_rotation_matrix, from_rotation_matrix, quaternion_rotate
from run.ShapeRecords import ShapeRecords

def draw_registration_result(src, trgt):
    source = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(src)
    target = open3d.geometry.PointCloud()
    target.points = open3d.utility.Vector3dVector(trgt)
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    open3d.draw_geometries([source, target])


def open3d_icp(src, trgt, init_rotation=np.eye(3, 3)):
    source = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(src)

    target = open3d.geometry.PointCloud()
    target.points = open3d.utility.Vector3dVector(trgt)

    init_rotation_4x4 = np.eye(4, 4)
    init_rotation_4x4[0:3, 0:3] = init_rotation

    threshold = 0.2
    reg_p2p = open3d.registration.registration_icp(source, target, threshold, init_rotation_4x4,
                                    open3d.registration.TransformationEstimationPointToPoint())

    return reg_p2p


def alignment_to_ground_truth(pc_pred, quat_pred, gt_pred, quat_gt):
    """
    Computes rotation that aligns predicted point cloud with the ground truth one.
    Basic idea is that quat_pred * pc_pred = quat_gt * gt_pred,
    and hence the desired alignment can be computed as quat_gt^(-1) * quat_pred.
    This alignment is then used as an initialization to ICP, which further refines it.

    :param pc_pred: predicted point cloud
    :param quat_pred: predicted camera rotation
    :param gt_pred: ground truth point cloud
    :param quat_gt: ground truth camera rotation
    :return: alignment quaternion
    """
    from util.quaternion import quaternion_normalise, quaternion_conjugate, \
        quaternion_multiply, quat2mat, mat2quat, from_rotation_matrix, as_rotation_matrix

    quat_gt = quat_gt.reshape(1, 4)
    quat_pred = quat_pred.reshape(1, 4)

    quat_pred_norm = quaternion_normalise(quat_pred)
    quat_gt_norm = quaternion_normalise(quat_gt)
    quat_unrot = quaternion_multiply(quaternion_conjugate(quat_gt_norm), quat_pred_norm)

    init_rotation_np = np.squeeze(as_rotation_matrix(quat_unrot))
    reg_p2p = open3d_icp(pc_pred.cpu().numpy(), gt_pred, init_rotation_np)

    T = np.array(reg_p2p.transformation)
    rot_matrix = T[:3, :3]
    assert (np.fabs(np.linalg.det(rot_matrix) - 1.0) <= 0.0001)
    rot_matrix = np.expand_dims(rot_matrix, 0)
    quat = from_rotation_matrix(rot_matrix)

    return quat, reg_p2p.inlier_rmse


def compute_alignment_candidates(cfg, dataset, all_rotations_file):
    exp_dir = cfg.checkpoint_dir
    save_pred_name = cfg.save_predictions_dir
    save_dir = os.path.join(exp_dir, save_pred_name)
    gt_dir = os.path.join(cfg.gt_pc_dir, cfg.synth_set)
    num_models = len(dataset)
    num_views = cfg.num_views
    num_show = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rmse = np.ones((num_models, num_views), np.float32)
    rotations = np.zeros((num_models, num_views, 4), np.float32)
    model_names = dataset.file_names
    for k in range(len(model_names)):
        sample = dataset[k]
        model_name = model_names[k]
        print(f"{k}/{num_models}")
        print(model_name)

        gt_filename = gt_filename = "{}/{}.mat".format(gt_dir, model_names[k]).replace('_features.p','')
        if not os.path.isfile(gt_filename):
            continue

        obj = scipy.io.loadmat(gt_filename)
        gt = obj["points"]
        mat_filename = "{}/{}_pc.pkl".format(save_dir, model_names[k])
        with open(mat_filename, 'rb') as handle:
            data = pickle.load(handle)
        all_pcs = torch.from_numpy(np.squeeze(data["points"])).to(device)
        all_cameras = torch.from_numpy(data["camera_pose"]).to(device)
        gt_cameras = torch.from_numpy(sample['cam_pos']).to(device)

        for view_idx in range(num_views):
            pred_pc_ref = all_pcs[view_idx, :, :]
            quat_pred = all_cameras[view_idx, :]
            cam_pos = gt_cameras[view_idx, :]
            quat_gt = torch.from_numpy(quaternion_from_campos(cam_pos)).to(device)
            quat, rmse_view = alignment_to_ground_truth(pred_pc_ref, quat_pred, gt, quat_gt)
            rmse[k, view_idx] = rmse_view
            rotations[k, view_idx, :] = np.squeeze(quat)

            #if cfg.vis_voxels and view_idx < num_show:
            #    rotated_with_quat = quaternion_rotate(np.expand_dims(pred_pc_ref, 0), quat)
            #    draw_registration_result(np.squeeze(rotated_with_quat), gt)

        print("rmse:", rmse[k, :])

    scipy.io.savemat(all_rotations_file, mdict={"rotations": rotations,
                                                "rmse": rmse})


def compute_alignment():
    cfg = app_config

    exp_dir = cfg.checkpoint_dir

    cfg.num_dataset_samples = 50
    dataset_folder = cfg.inp_dir
    dataset = ShapeRecords(dataset_folder, cfg,'test')

    num_to_estimate = 15
    num_models = len(dataset)

    all_rotations_file = f"{exp_dir}/reference_rotations.mat"
    compute_alignment_candidates(cfg, dataset, all_rotations_file)

    stuff = scipy.io.loadmat(all_rotations_file)
    rotations = stuff["rotations"]
    rmse = stuff["rmse"]

    # For each model filter out outlier views
    num_filtered = 2
    rotations_filtered = np.zeros((num_models, num_filtered, 4))
    rmse_filtered = np.zeros((num_models, num_filtered))

    for model_idx in range(num_models):
        rmse_m = rmse[model_idx, :]
        indices = np.argsort(rmse_m)
        indices = indices[0:num_filtered]
        rmse_filtered[model_idx, :] = rmse_m[indices]
        rotations_filtered[model_idx, :, :] = rotations[model_idx, indices, :]

    # Sort models by RMSE and choose num_to_estimate best ones
    model_mean_rmse = np.mean(rmse_filtered, axis=1)
    models_indices = np.argsort(model_mean_rmse)
    models_indices = models_indices[0:num_to_estimate]
    reference_rotations = rotations_filtered[models_indices, :, :]

    reference_rotations = np.reshape(reference_rotations, [-1, 4])
    print(reference_rotations)

    # Somehow NaNs slip in the computation, so filter them out
    nan = np.isnan(reference_rotations)
    good = np.logical_not(np.any(nan, axis=1))
    reference_rotations = reference_rotations[good, :]

    # Average quaternion rotations, may be better than arithmetic average
    reference_rotation = quatWAvgMarkley(reference_rotations)
    print("Global rotation:", reference_rotation)

    scipy.io.savemat(f"{exp_dir}/final_reference_rotation.mat",
                     mdict={"rotation": reference_rotation})


def main():
    compute_alignment()


if __name__ == '__main__':
    main()
