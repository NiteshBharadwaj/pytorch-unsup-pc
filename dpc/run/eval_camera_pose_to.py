import startup

import os

import numpy as np
import scipy.io

from run.ShapeRecords import ShapeRecords
from util.simple_dataset import Dataset3D
from util.app_config import config as app_config
from util.quaternion import quaternion_multiply, quaternion_conjugate, quaternion_conjugate_np, quaternion_multiply_np
from util.camera import quaternion_from_campos
from util.system import setup_environment
import pickle
import torch

def qmul(quat_inp, quat_inp_2):
    quat_conj = quaternion_conjugate_np(quat_inp)
    quat_mul = quaternion_multiply_np(np.split(quat_inp,4,1), np.split(quat_inp_2.squeeze().reshape(1,4),4,1))
    return quat_mul

def run_eval():
    cfg = app_config
    setup_environment(cfg)
    exp_dir = cfg.checkpoint_dir
    num_views = cfg.num_views
    dataset_folder = cfg.inp_dir

    save_pred_name = "{}_{}".format(cfg.save_predictions_dir, cfg.eval_split)
    save_dir = os.path.join(exp_dir, cfg.save_predictions_dir)

    reference_rotation = scipy.io.loadmat("{}/final_reference_rotation.mat".format(exp_dir))["rotation"]
    ref_conj_np = quaternion_conjugate_np(reference_rotation)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ShapeRecords(dataset_folder, cfg,'test')

    if cfg.models_list:
        model_names = parse_lines(cfg.models_list)
    else:
        model_names = dataset.file_names
    num_models = len(model_names)

    angle_error = np.zeros((num_models, num_views), dtype=np.float64)

    for k in range(num_models):
        sample = dataset.__getitem__(k)
        print("{}/{}".format(k, num_models))
        print(model_names[k])

        mat_filename = "{}/{}_pc.pkl".format(save_dir, model_names[k])
        
        if not os.path.isfile(mat_filename):
            continue
        
        with open(mat_filename, 'rb') as handle:
            data = pickle.load(handle)
        
        all_cameras = data["camera_pose"]
        for view_idx in range(num_views):
            cam_pos = sample["cam_pos"][view_idx, :]
            gt_quat_np = quaternion_from_campos(cam_pos)
            gt_quat_np = np.expand_dims(gt_quat_np, 0)
            pred_quat_np = all_cameras[view_idx, :]
            pred_quat_np /= np.linalg.norm(pred_quat_np)
            pred_quat_np = np.expand_dims(pred_quat_np, 0)

            pred_quat_aligned_np = qmul(pred_quat_np,ref_conj_np)

            q1 = gt_quat_np
            q2 = pred_quat_aligned_np

            q1_conj = quaternion_conjugate_np(q1)
            q_diff = qmul(q1_conj,q2)

            ang_diff = 2 * np.arccos(q_diff[0, 0])
            if ang_diff > np.pi:
                ang_diff -= 2*np.pi

            angle_error[k, view_idx] = np.fabs(ang_diff)

    all_errors = np.reshape(angle_error, (-1))
    angle_thresh_rad = cfg.pose_accuracy_threshold / 180.0 * np.pi
    correct = all_errors < angle_thresh_rad
    num_predictions = correct.shape[0]
    accuracy = np.count_nonzero(correct) / num_predictions
    median_error = np.sort(all_errors)[num_predictions // 2]
    median_error = median_error / np.pi * 180
    print("accuracy:", accuracy, "median angular error:", median_error)

    scipy.io.savemat(os.path.join(exp_dir, "pose_error_{}.mat".format(save_pred_name)),
                     {"angle_error": angle_error,
                      "accuracy": accuracy,
                      "median_error": median_error})

    f = open(os.path.join(exp_dir, "pose_error_{}.txt".format(save_pred_name)), "w")
    f.write("{} {}\n".format(accuracy, median_error))
    f.close()


def main():
    run_eval()


if __name__ == '__main__':
    main()
