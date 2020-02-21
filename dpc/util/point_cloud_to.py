import numpy as np
import torch
import util.drc
from util.quaternion import quaternion_rotate
import torch.nn.functional as F
from util.camera import intrinsic_matrix
# from util.point_cloud_distance import *

#
# def multi_expand(inp, axis, num):
#     inp_big = inp
#     for i in range(num):
#         inp_big = tf.expand_dims(inp_big, axis)
#     return inp_big

#
# def pointcloud2voxels(cfg, input_pc, sigma):  # [B,N,3]
#     # TODO replace with split or tf.unstack
#     x = input_pc[:, :, 0]
#     y = input_pc[:, :, 1]
#     z = input_pc[:, :, 2]
#
#     vox_size = cfg.vox_size
#
#     rng = tf.linspace(-1.0, 1.0, vox_size)
#     xg, yg, zg = tf.meshgrid(rng, rng, rng)  # [G,G,G]
#
#     x_big = multi_expand(x, -1, 3)  # [B,N,1,1,1]
#     y_big = multi_expand(y, -1, 3)  # [B,N,1,1,1]
#     z_big = multi_expand(z, -1, 3)  # [B,N,1,1,1]
#
#     xg = multi_expand(xg, 0, 2)  # [1,1,G,G,G]
#     yg = multi_expand(yg, 0, 2)  # [1,1,G,G,G]
#     zg = multi_expand(zg, 0, 2)  # [1,1,G,G,G]
#
#     # squared distance
#     sq_distance = tf.square(x_big - xg) + tf.square(y_big - yg) + tf.square(z_big - zg)
#
#     # compute gaussian
#     func = tf.exp(-sq_distance / (2.0 * sigma * sigma))  # [B,N,G,G,G]
#
#     # normalise gaussian
#     if cfg.pc_normalise_gauss:
#         normaliser = tf.reduce_sum(func, [2, 3, 4], keep_dims=True)
#         func /= normaliser
#     elif cfg.pc_normalise_gauss_analytical:
#         # should work with any grid sizes
#         magic_factor = 1.78984352254  # see estimate_gauss_normaliser
#         sigma_normalised = sigma * vox_size
#         normaliser = 1.0 / (magic_factor * tf.pow(sigma_normalised, 3))
#         func *= normaliser
#
#     summed = tf.reduce_sum(func, axis=1)  # [B,G,G G]
#     voxels = tf.clip_by_value(summed, 0.0, 1.0)
#     voxels = tf.expand_dims(voxels, axis=-1)  # [B,G,G,G,1]
#
#     return voxels


def pointcloud2voxels3d_fast(cfg, pc, rgb):  # [B,N,3]
    vox_size = cfg.vox_size
    if cfg.vox_size_z != -1:
        vox_size_z = cfg.vox_size_z
    else:
        vox_size_z = vox_size

    batch_size = pc.shape[0]
    num_points = pc.shape[1]

    has_rgb = rgb is not None

    grid_size = 1.0
    half_size = grid_size / 2

    filter_outliers = True
    valid = (pc >= -half_size) * (pc <= half_size)
    valid = torch.prod(valid, dim=-1)

    vox_size_tf = torch.from_numpy(np.array([[[vox_size_z, vox_size, vox_size]]]))
    pc_grid = (pc + half_size) * (vox_size_tf - 1)
    indices_floor = torch.floor(pc_grid)
    indices_int = indices_floor.long()
    batch_indices = torch.arange(0, batch_size, 1)
    batch_indices = batch_indices.reshape(batch_indices.shape[0],1)
    batch_indices = batch_indices.repeat(1,num_points)
    batch_indices = batch_indices.reshape(batch_indices.shape[0], batch_indices.shape[1],1)

    indices = torch.cat((batch_indices, indices_int), dim=2)
    indices = indices.reshape(-1,4)
    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]

    if filter_outliers:
        valid = valid.reshape(-1)
        indices = indices[valid]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        updates = updates_raw.reshape(-1)

        if filter_outliers:
            updates = updates[valid]

        indices_loc = indices
        indices_shift = torch.from_numpy(np.array([[0] + pos]))
        num_updates = indices_loc.shape[0]
        indices_shift = indices_shift.repeat(num_updates,1)
        indices_loc = indices_loc + indices_shift
        voxels = torch.zeros((batch_size, vox_size_z, vox_size, vox_size),dtype=torch.float64)
        voxels[indices_loc[:,0],indices_loc[:,1],indices_loc[:,2],indices_loc[:,3]] = updates
        if has_rgb:
            if cfg.pc_rgb_stop_points_gradient:
                updates_raw = updates_raw.detach()
            updates_rgb = updates_raw. tf.expand_dims(updates_raw, axis=-1) * rgb
            updates_rgb = updates_rgb.reshape(-1,3)
            if filter_outliers:
                updates_rgb = updates_rgb[valid]
            voxels_rgb = torch.zeros((batch_size, vox_size_z, vox_size, vox_size), dtype=torch.float64)
            voxels_rgb[indices_loc[:,0],indices_loc[:,1],indices_loc[:,2],indices_loc[:,3]] = updates_rgb
        else:
            voxels_rgb = None

        return voxels, voxels_rgb

    voxels = []
    voxels_rgb = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                vx, vx_rgb = interpolate_scatter3d([k, j, i])
                voxels.append(vx)
                voxels_rgb.append(vx_rgb)

    voxels = torch.sum(torch.stack(voxels),0)
    voxels_rgb = torch.sum(torch.stack(voxels_rgb),0) if has_rgb else None

    return voxels, voxels_rgb


def smoothen_voxels3d(cfg, voxels, kernel):
    if cfg.pc_separable_gauss_filter:
        for krnl in kernel:
            pad = np.array(krnl.shape[2:])//2
            pad = tuple(pad.astype(int).tolist())
            voxels = F.conv3d(voxels, krnl.double(),stride=1, padding=pad)
    else:
        pad = np.array(kernel.shape[2:]) // 2
        pad = tuple(pad.astype(int).tolist())
        voxels = F.conv3d(voxels, kernel.double(),stride=1, padding=pad)
    return voxels


def convolve_rgb(cfg, voxels_rgb, kernel):
    channels = [voxels_rgb[:, :, :, :, k:k+1] for k in range(3)]
    for krnl in kernel:
        for i in range(3):
            pad = np.array(krnl.shape[2:]) // 2
            pad = tuple(pad.astype(int).tolist())
            channels[i] = F.conv3d(channels[i], krnl.double(), stride=1, padding=pad)
    out = torch.cat(channels, axis=1)
    return out


def pc_perspective_transform(cfg, point_cloud,
                             transform, predicted_translation=None,
                             focal_length=None):
    """
    :param cfg:
    :param point_cloud: [B, N, 3]
    :param transform: [B, 4] if quaternion or [B, 4, 4] if camera matrix
    :param predicted_translation: [B, 3] translation vector
    :return:
    """
    camera_distance = cfg.camera_distance

    if focal_length is None:
        focal_length = cfg.focal_length
    else:
        focal_length = focal_length.reshape(focal_length.shape[0], focal_length.shape[1],1)

    if cfg.pose_quaternion:
        pc2 = quaternion_rotate(point_cloud, transform)
        if predicted_translation is not None:
            predicted_translation = predicted_translation.reshape(predicted_translation.shape[0],1, predicted_translation.shape[1])
            pc2 += predicted_translation
        xs = pc2[:,:,2:3]
        ys = pc2[:,:,1:2]
        zs = pc2[:,:,0:1]

        # translation part of extrinsic camera
        zs += camera_distance
        # intrinsic transform
        xs *= focal_length
        ys *= focal_length
    else:
        xyz1 = F.pad(point_cloud, (0,1))

        extrinsic = transform
        intr = intrinsic_matrix(cfg, dims=4)
        intrinsic = torch.from_numpy(intr)
        intrinsic = intrinsic.reshape(1,intrinsic.shape[0],intrinsic.shape[1])
        intrinsic = intrinsic.repeat(extrinsic.shape[0],1,1)
        full_cam_matrix = torch.matmul(intrinsic, extrinsic)

        pc2 = torch.matmul(xyz1, full_cam_matrix.permute(0, 2, 1))

        # TODO unstack instead of split
        xs = pc2[:, :, 2:3]
        ys = pc2[:, :, 1:2]
        zs = pc2[:, :, 0:1]

    xs  = torch.div(xs,zs)
    ys = torch.div(ys,zs)

    zs = zs -camera_distance
    if predicted_translation is not None:
        zt = predicted_translation[:,:,0:1]
        zs -= zt

    xyz2 = torch.cat([zs, ys, xs], dim=2)
    return xyz2
#
#
# def pointcloud_project(cfg, point_cloud, transform, sigma):
#     tr_pc = pc_perspective_transform(cfg, point_cloud, transform)
#     voxels = pointcloud2voxels(cfg, tr_pc, sigma)
#     voxels = tf.transpose(voxels, [0, 2, 1, 3, 4])
#
#     proj, probs = util.drc.drc_projection(voxels, cfg)
#     proj = tf.reverse(proj, [1])
#     return proj, voxels
#
#
def pointcloud_project_fast(cfg, point_cloud, transform, predicted_translation,
                            all_rgb, kernel=None, scaling_factor=None, focal_length=None):
    has_rgb = all_rgb is not None
    tr_pc = pc_perspective_transform(cfg, point_cloud,
                                     transform, predicted_translation,
                                     focal_length)
    voxels, voxels_rgb = pointcloud2voxels3d_fast(cfg, tr_pc, all_rgb)
    voxels = voxels.reshape(voxels.shape[0], 1,voxels.shape[1], voxels.shape[2], voxels.shape[3])
    voxels_raw = voxels

    voxels = torch.clamp(voxels,0.0,1.0)

    if kernel is not None:
        # TODO: Uncomment for GPU
        #voxels = smoothen_voxels3d(cfg, voxels, kernel)
        if has_rgb:
            if not cfg.pc_rgb_clip_after_conv:
                voxels_rgb = torch.clamp(voxels_rgb, 0.0, 1.0)
            voxels_rgb = convolve_rgb(cfg, voxels_rgb, kernel)

    if scaling_factor is not None:
        sz = scaling_factor.shape[0]
        scaling_factor = scaling_factor.reshape(sz, 1, 1, 1, 1)
        voxels = voxels * scaling_factor
        voxels = torch.clamp(voxels, 0.0, 1.0)

    if has_rgb:
        if cfg.pc_rgb_divide_by_occupancies:
            voxels_div = voxels_raw.detach()
            voxels_div = smoothen_voxels3d(cfg, voxels_div, kernel)
            voxels_rgb = voxels_rgb / (voxels_div + cfg.pc_rgb_divide_by_occupancies_epsilon)

        if cfg.pc_rgb_clip_after_conv:
            voxels_rgb = torch.clamp(voxels_rgb, 0.0, 1.0)

    if cfg.ptn_max_projection:
        proj = torch.max(voxels,1)
        drc_probs = None
        proj_depth = None
    else:
        proj, drc_probs = util.drc.drc_projection(voxels, cfg)
        drc_probs = torch.flip(drc_probs, [2])
        proj_depth = util.drc.drc_depth_projection(drc_probs, cfg)

    proj = torch.flip(proj, [1])

    if voxels_rgb is not None:
        voxels_rgb = torch.flip(voxels_rgb, [2])
        proj_rgb = util.drc.project_volume_rgb_integral(cfg, drc_probs, voxels_rgb)
    else:
        proj_rgb = None

    output = {
        "proj": proj,
        "voxels": voxels,
        "tr_pc": tr_pc,
        "voxels_rgb": voxels_rgb,
        "proj_rgb": proj_rgb,
        "drc_probs": drc_probs,
        "proj_depth": proj_depth
    }
    return output


def select_3d(data, indices):
    return data[[indices[:,:,0], indices[:,:,1]]]

def pc_point_dropout(points, rgb, keep_prob):
    shape = points.shape
    num_input_points = shape[1]
    batch_size = shape[0]
    num_channels = shape[2]
    num_output_points = int(num_input_points * keep_prob)

    def sampler(num_output_points_np):
        all_inds = []
        for k in range(batch_size):
            ind = np.random.choice(num_input_points, num_output_points_np, replace=False)
            ind = np.expand_dims(ind, axis=-1)
            ks = np.ones_like(ind) * k
            inds = np.concatenate((ks, ind), axis=1)
            all_inds.append(np.expand_dims(inds, 0))
        return np.concatenate(tuple(all_inds), 0).astype(np.int64)

    selected_indices = sampler(num_output_points)
    selected_indices = torch.from_numpy(selected_indices)
    out_points = select_3d(points, selected_indices)
    out_points = out_points.reshape(batch_size, num_output_points, num_channels)
    if rgb is not None:
        num_rgb_channels = rgb.shape[2]
        out_rgb = select_3d(rgb, selected_indices)
        out_rgb = out_rgb.reshape(batch_size, num_output_points, num_rgb_channels)
    else:
        out_rgb = None
    return out_points, out_rgb


# def subsample_points(xyz, num_points):
#     idxs = np.random.choice(xyz.shape[0], num_points)
#     xyz_s = xyz[idxs, :]
#     return xyz_s
