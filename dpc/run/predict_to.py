import startup

import os

import numpy as np
import imageio
import scipy.io

#import tensorflow as tf
#import tensorflow.contrib.slim as slim
import torch
from torch.utils.tensorboard import SummaryWriter
import pdb

from util.common import parse_lines
from util.app_config import config as app_config
from util.system import setup_environment
from util.simple_dataset import Dataset3D
from util.fs import mkdir_if_missing
from util.camera import get_full_camera, quaternion_from_campos
from util.visualise import vis_pc, merge_grid, mask4vis
from util.point_cloud_to import pointcloud2voxels3d_fast, pointcloud_project_fast
#pointcloud2voxels, smoothen_voxels3d, pointcloud2voxels3d_fast, pointcloud_project_fast
from util.quaternion import as_rotation_matrix, quaternion_rotate

from models import model_pc_to as model_pc
from run.ShapeRecords import ShapeRecords
import pickle

def build_model(model, input, global_step):
    cfg = model.cfg()
    batch_size = cfg.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    code = 'images' if cfg.predict_pose else 'images_1'
    for k in input.keys():        
        try:
            input[k] = torch.from_numpy(input[k]).to(device)    
        except AttributeError:
            pass
    with torch.no_grad():
        outputs = model(input, global_step, is_training=False, run_projection=False)
    cam_transform = outputs['poses'] if cfg.predict_pose else None
    outputs["inputs"] = input[code]
    outputs["camera_extr_src"] = input['matrices']
    outputs["cam_quaternion"] = input['camera_quaternion']
    outputs["cam_transform"] = cam_transform
    return outputs


def model_student(inputs, model):
    cfg = model.cfg()
    outputs = model.model_predict(inputs, is_training=False,
                                  predict_for_all=False)
    points = outputs["points_1"]
    camera_pose = outputs["pose_student"]
    rgb = None
    transl = outputs["predicted_translation"] if cfg.predict_translation else None
    #proj_out = pointcloud_project_fast(model.cfg(), points, camera_pose, transl, rgb, None)
    #proj_out = pointcloud_project_fast(model.cfg(), points, camera_pose, transl, rgb, model.gauss_kernel())
    #proj = proj_out["proj_depth"]
    
    return camera_pose


def model_unrotate_points(cfg):
    """
    un_q = quat_gt^(-1) * predicted_quat
    pc_unrot = un_q * pc_np * un_q^(-1)
    """

    from util.quaternion import quaternion_normalise, quaternion_conjugate, \
        quaternion_rotate, quaternion_multiply
    gt_quat = tf.placeholder(dtype=tf.float32, shape=[1, 4])

    pred_quat_n = quaternion_normalise(pred_quat)
    gt_quat_n = quaternion_normalise(gt_quat)

    un_q = quaternion_multiply(quaternion_conjugate(gt_quat_n), pred_quat_n)
    pc_unrot = quaternion_rotate(input_pc, un_q)

    return input_pc, pred_quat, gt_quat, pc_unrot


def normalise_depthmap(depth_map):
    depth_map = np.clip(depth_map, 1.5, 2.5)
    depth_map -= 1.5
    return depth_map


def compute_predictions():
    cfg = app_config

    setup_environment(cfg)

    exp_dir = cfg.checkpoint_dir

    cfg.batch_size = 1
    cfg.step_size = 1

    pc_num_points = cfg.pc_num_points
    vox_size = cfg.vox_size
    save_pred = cfg.save_predictions
    save_voxels = cfg.save_voxels
    fast_conversion = True

    pose_student = cfg.pose_predictor_student and cfg.predict_pose

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model_pc.ModelPointCloud(cfg)
    model = model.to(device)

    log_dir = '../../dpc/run/model_run_data/'
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = cfg.weight_decay)
    global_step = 100000
    if global_step>0:
        checkpoint_path = os.path.join(log_dir,'model.ckpt_{}.pth'.format(global_step))
        print("Loading from path:",checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        global_step_val = checkpoint['global_step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    else:
        global_step_val = global_step
    print('Restored checkpoint at {} with loss {}'.format(global_step, loss))

    save_dir = os.path.join(exp_dir, '{}_vis_proj'.format(cfg.save_predictions_dir))
    mkdir_if_missing(save_dir)
    save_pred_dir = os.path.join(exp_dir, cfg.save_predictions_dir)
    mkdir_if_missing(save_pred_dir)

    vis_size = cfg.vis_size

    split_name = "test"
    dataset_folder = cfg.inp_dir

    dataset = ShapeRecords(dataset_folder, cfg, split_name)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg.batch_size, shuffle=cfg.shuffle_dataset,
                                                 num_workers=4,drop_last=True)
    pose_num_candidates = cfg.pose_predict_num_candidates
    num_views = cfg.num_views
    plot_h = 4
    plot_w = 6
    num_views = int(min(num_views, plot_h * plot_w / 2))

    if cfg.models_list:
        model_names = parse_lines(cfg.models_list)
    else:
        model_names = dataset.file_names
    num_models = len(model_names)
    
    for k in range(num_models):
        model_name = model_names[k]
        sample = dataset.__getitem__(k)
        images = sample['image']
        masks = sample['mask']
        if cfg.saved_camera:
            cameras = sample['extrinsic']
            cam_pos = sample['cam_pos']
        if cfg.vis_depth_projs:
            depths = sample['depth']
        if cfg.variable_num_views:
            num_views = sample['num_views']

        print("{}/{} {}".format(k, num_models, model_name))

        if pose_num_candidates == 1:
            grid = np.empty((plot_h, plot_w), dtype=object)
        else:
            plot_w = pose_num_candidates + 1
            if pose_student:
                plot_w += 1
            grid = np.empty((num_views, plot_w), dtype=object)

        if save_pred:
            all_pcs = np.zeros((num_views, pc_num_points, 3))
            all_cameras = np.zeros((num_views, 4))
            #all_voxels = np.zeros((num_views, vox_size, vox_size, vox_size))
            #all_z_latent = np.zeros((num_views, cfg.fc_dim))
        
      
        for view_idx in range(num_views):
            input_image_np = images[[view_idx], :, :, :]
            gt_mask_np = masks[[view_idx], :, :, :]
            if cfg.saved_camera:
                extr_mtr = cameras[view_idx, :, :]
                cam_quaternion_np = quaternion_from_campos(cam_pos[view_idx, :])
                cam_quaternion_np = np.expand_dims(cam_quaternion_np, axis=0)
            else:
                extr_mtr = np.zeros((4, 4))

            code = 'images' if cfg.predict_pose else 'images_1'
            input = {code: input_image_np,
                     'matrices': extr_mtr,
                     'camera_quaternion': cam_quaternion_np}
           
            out = build_model(model, input, global_step)
            input_image = out["inputs"]
            cam_matrix = out["camera_extr_src"]
            cam_quaternion = out["cam_quaternion"]
            point_cloud = out["points_1"]
            #gb = out["rgb_1"] if cfg.pc_rgb else None
            #rojs = out["projs"]
            #rojs_rgb = out["projs_rgb"]
            #rojs_depth = out["projs_depth"]
            cam_transform = out["cam_transform"]
            #_latent = out["z_latent"]

            #if cfg.pc_rgb:
            #    proj_tensor = projs_rgb
            #elif cfg.vis_depth_projs:
            #    proj_tensor = projs_depth
            #else:
            #    proj_tensor = projs

            if pose_student:
                camera_student_np = out["pose_student"]
                predicted_camera = camera_student_np
            else:
                predicted_camera = cam_transf_np

            #if cfg.vis_depth_projs:
            #    proj_np = normalise_depthmap(out["projs"])
            #    if depths is not None:
            #        depth_np = depths[view_idx, :, :, :]
            #        depth_np = normalise_depthmap(depth_np)
            #    else:
            #        depth_np = 1.0 - np.squeeze(gt_mask_np)
            #    if pose_student:
            #        proj_student_np = normalise_depthmap(proj_student_np)


            #if save_voxels:
            #    if fast_conversion:
            #        voxels, _ = pointcloud2voxels3d_fast(cfg, input_pc, None)
            #        voxels = tf.expand_dims(voxels, axis=-1)
            #        voxels = smoothen_voxels3d(cfg, voxels, model.gauss_kernel())
            #    else:
            #        voxels = pointcloud2voxels(cfg, input_pc, model.gauss_sigma())
            if cfg.predict_pose:
                if cfg.save_rotated_points:
                    ref_rot = scipy.io.loadmat("{}/final_reference_rotation.mat".format(exp_dir))
                    ref_rot = ref_rot["rotation"]

                    pc_unrot = quaternion_rotate(input_pc, ref_quat)
                    point_cloud = pc_np_unrot


            if cfg.pc_rgb:
                gt_image = input_image_np
            elif cfg.vis_depth_projs:
                gt_image = depth_np
            else:
                gt_image = gt_mask_np

#             if pose_num_candidates == 1:
#                 view_j = view_idx * 2 // plot_w
#                 view_i = view_idx * 2 % plot_w

#                 gt_image = np.squeeze(gt_image)
#                 grid[view_j, view_i] = mask4vis(cfg, gt_image, vis_size)

#                 curr_img = np.squeeze(out[projs])
#                 grid[view_j, view_i + 1] = mask4vis(cfg, curr_img, vis_size)

#                 if cfg.save_individual_images:
#                     curr_dir = os.path.join(save_dir, model_names[k])
#                     if not os.path.exists(curr_dir):
#                         os.makedirs(curr_dir)
#                     imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'rgb_gt')),
#                                     mask4vis(cfg, np.squeeze(input_image_np), vis_size))
#                     imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'mask_pred')),
#                                     mask4vis(cfg, np.squeeze(proj_np), vis_size))
#             else:
#                 view_j = view_idx

#                 gt_image = np.squeeze(gt_image)
#                 grid[view_j, 0] = mask4vis(cfg, gt_image, vis_size)

#                 for kk in range(pose_num_candidates):
#                     curr_img = np.squeeze(out["projs"][kk, :, :, :].detach().cpu())
#                     grid[view_j, kk + 1] = mask4vis(cfg, curr_img, vis_size)

#                     if cfg.save_individual_images:
#                         curr_dir = os.path.join(save_dir, model_names[k])
#                         if not os.path.exists(curr_dir):
#                             os.makedirs(curr_dir)
#                         imageio.imwrite(os.path.join(curr_dir, '{}_{}_{}.png'.format(view_idx, kk, 'mask_pred')),
#                                         mask4vis(cfg, np.squeeze(curr_img), vis_size))

#                 if cfg.save_individual_images:
#                     imageio.imwrite(os.path.join(curr_dir, '{}_{}.png'.format(view_idx, 'mask_gt')),
#                                     mask4vis(cfg, np.squeeze(gt_mask_np), vis_size))

#                 if pose_student:
#                     grid[view_j, -1] = mask4vis(cfg, np.squeeze(proj_student_np.detach().cpu()), vis_size)

            if save_pred:
                #pc_np = pc_np.detach().cpu().numpy()
                all_pcs[view_idx, :, :] = np.squeeze(point_cloud.detach().cpu())
                #all_z_latent[view_idx] = z_latent.detach().cpu()
                if cfg.predict_pose:
                    all_cameras[view_idx, :] = predicted_camera.detach().cpu()
#                 if save_voxels:
#                     # multiplying by two is necessary because
#                     # pc->voxel conversion expects points in [-1, 1] range
#                     pc_np_range = pc_np
#                     if not fast_conversion:
#                         pc_np_range *= 2.0
#                     voxels_np = sess.run(voxels, feed_dict={input_pc: pc_np_range})
#                     all_voxels[view_idx, :, :, :] = np.squeeze(voxels_np)

#             vis_view = view_idx == 0 or cfg.vis_all_views
#             if cfg.vis_voxels and vis_view:
#                 rgb_np = np.squeeze(rgb_np) if cfg.pc_rgb else None
#                 vis_pc(np.squeeze(pc_np), rgb=rgb_np)

        #grid_merged = merge_grid(cfg, grid)
        #imageio.imwrite("{}/{}_proj.png".format(save_dir, sample.file_names), grid_merged)
        
        if save_pred:
            if 0:
                save_dict = {"points": all_pcs}
                if cfg.predict_pose:
                    save_dict["camera_pose"] = all_cameras
                scipy.io.savemat("{}/{}_pc.mat".format(save_pred_dir, model_names[k]),
                                 mdict=save_dict)
            else:
                save_dict = {"points": all_pcs}
                if cfg.predict_pose:
                    save_dict["camera_pose"] = all_cameras
                with open("{}/{}_pc.pkl".format(save_pred_dir, model_names[k]), 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  

#             if save_voxels:
#                 np.savez("{}/{}_vox".format(save_pred_dir,model_names[k]), all_voxels)



#def main(_):
def main():
    compute_predictions()


if __name__ == '__main__':
    #tf.app.run()
    main()
