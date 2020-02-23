import numpy as np
import scipy.io
import torch
import torch.nn as nn

from nets.img_encoder_to import Encoder
from nets.pc_decoder_to import Decoder
from nets.pose_net_to import PoseNet

from models.model_base_to import ModelBase, pool_single_view

# from util.losses import add_drc_loss, add_proj_rgb_loss, add_proj_depth_loss
# from util.point_cloud import pointcloud_project, pointcloud_project_fast, \

from util.point_cloud_to import  pc_point_dropout, pointcloud_project_fast
#from util.gauss_kernel import gauss_smoothen_image, smoothing_kernel
from util.gauss_kernel import smoothing_kernel
from util.quaternion import \
    quaternion_multiply as q_mul,\
    quaternion_normalise as q_norm,\
    quaternion_rotate as q_rotate,\
    quaternion_conjugate as q_conj

from nets.net_factory import get_network

def one_hot(y,num):
    batch_size = y.shape[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        y_onehot = torch.FloatTensor(batch_size,num)
    else:
        y_onehot = torch.cuda.FloatTensor(batch_size,num)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y.reshape(-1,1), 1)
    return y_onehot

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def tf_repeat_0(input, num):
    orig_shape = input.shape
    new_shape = [*input.shape].copy()
    new_shape.insert(1,1)
    e = input.reshape(new_shape)
    tiler = [1 for _ in range(len(orig_shape)+1)]
    tiler[1] = num
    tiled = e.repeat(tiler)
    new_shape = [-1]
    new_shape.extend(orig_shape[1:])
    final = tiled.reshape(new_shape)
    return final


def get_smooth_sigma(cfg, global_step):
    num_steps = cfg.max_number_of_steps
    diff = (cfg.pc_relative_sigma_end - cfg.pc_relative_sigma)
    sigma_rel = cfg.pc_relative_sigma + global_step / num_steps * diff
    return sigma_rel

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def get_dropout_prob(cfg, global_step):
    if not cfg.pc_point_dropout_scheduled:
        return cfg.pc_point_dropout

    exp_schedule = cfg.pc_point_dropout_exponential_schedule
    num_steps = cfg.max_number_of_steps
    keep_prob_start = cfg.pc_point_dropout
    keep_prob_end = 1.0
    start_step = cfg.pc_point_dropout_start_step
    end_step = cfg.pc_point_dropout_end_step
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    if exp_schedule:
        alpha = torch.log(keep_prob_end / keep_prob_start)
        keep_prob = keep_prob_start * torch.exp(alpha * x)
    else:
        keep_prob = k * x + b
    keep_prob = clamp(keep_prob, keep_prob_start, keep_prob_end)
    return keep_prob

#
# def get_st_global_scale(cfg, global_step):
#     num_steps = cfg.max_number_of_steps
#     keep_prob_start = 0.0
#     keep_prob_end = 1.0
#     start_step = 0
#     end_step = 0.1
#     global_step = tf.cast(global_step, dtype=tf.float32)
#     x = global_step / num_steps
#     k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
#     b = keep_prob_start - k * start_step
#     keep_prob = k * x + b
#     keep_prob = tf.clip_by_value(keep_prob, keep_prob_start, keep_prob_end)
#     keep_prob = tf.reshape(keep_prob, [])
#     return tf.cast(keep_prob, tf.float32)


def align_predictions(outputs, alignment):
    outputs["points_1"] = q_rotate(outputs["points_1"], alignment)
    outputs["poses"] = q_mul(outputs["poses"], q_conj(alignment))
    outputs["pose_student"] = q_mul(outputs["pose_student"], q_conj(alignment))
    return outputs


class ScalePredictor(nn.Module):  # pylint:disable=invalid-name
    """Inherits the generic Im2Vox model class and implements the functions."""

    def __init__(self, cfg, summary_writer):
        super(ScalePredictor, self).__init__()
        z_dim = cfg.z_dim
        self.fc = nn.Linear(z_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.fc.apply(init_weights) ##TODO: Change init
        self.cfg = cfg
        self.summary_writer = summary_writer
    def forward(self,x, is_training):
        x = self.fc(x)
        pred = self.sigmoid(x) * self.cfg.pc_occupancy_scaling_maximum
        if is_training:
            predcpu = pred.cpu()
            self.summary_writer.add_scalar("pc_occupancy_scaling_factor", predcpu.mean().detach().numpy())
        return pred

class FocalLengthPredictor(nn.Module):
    def __init__(self, cfg, summary_writer):
        super(FocalLengthPredictor, self).__init__()
        z_dim = cfg.z_dim
        self.fc = nn.Linear(z_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.cfg = cfg
        self.summary_writer = summary_writer
    def forward(self,x,is_training):
        pred = self.fc(x)
        out = self.cfg.focal_length_mean + self.sigmoid(pred)*self.cfg.focal_length_range
        if is_training:
            outcpu = out.cpu()
            self.summary_writer.add_scalar("meta/focal_length", outcpu.mean().detach().numpy())
        return out

class ModelPointCloud(ModelBase):  # pylint:disable=invalid-name
    """Inherits the generic Im2Vox model class and implements the functions."""

    def __init__(self, cfg, summary_writer, global_step=0):
        super(ModelPointCloud, self).__init__(cfg)
        self._gauss_sigma = None
        self._gauss_kernel = None
        self._sigma_rel = None
        self._global_step = global_step
        self.summary_writer = summary_writer
        self.setup_sigma()
        self.setup_misc()
        self._alignment_to_canonical = None
        if cfg.align_to_canonical and cfg.predict_pose:
            self.set_alignment_to_canonical()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.poseNet = PoseNet(cfg)
        self.scalePred = ScalePredictor(cfg,summary_writer)
        self.focalPred = FocalLengthPredictor(cfg,summary_writer)


    def setup_sigma(self):
        cfg = self.cfg()
        sigma_rel = get_smooth_sigma(cfg, self._global_step)
        self.summary_writer.add_scalar("meta/gauss_sigma_rel", sigma_rel)
        self._sigma_rel = sigma_rel
        self._gauss_sigma = sigma_rel / cfg.vox_size
        self._gauss_kernel = smoothing_kernel(cfg, sigma_rel)

    def gauss_sigma(self):
        return self._gauss_sigma

    def gauss_kernel(self):
        return self._gauss_kernel

    def setup_misc(self):
        if self.cfg().pose_student_align_loss:
            num_points = 2000
            sigma = 1.0
            values = np.random.normal(loc=0.0, scale=sigma, size=(num_points, 3))
            values = np.clip(values, -3*sigma, +3*sigma)
            self._pc_for_alignloss = values

    def set_alignment_to_canonical(self):
        exp_dir = self.cfg().checkpoint_dir
        stuff = scipy.io.loadmat(f"{exp_dir}/final_reference_rotation.mat")
        alignment = stuff["rotation"]
        self._alignment_to_canonical = alignment

    def model_predict(self, images, is_training=False, reuse=False, predict_for_all=False, alignment=None):
        outputs = {}
        cfg = self._params
        enc_outputs = self.encoder(images)
        ids = enc_outputs['ids']
        outputs['conv_features'] = enc_outputs['conv_features']
        outputs['ids'] = ids
        outputs['z_latent'] = enc_outputs['z_latent']

        # unsupervised case, case where convnet runs on all views, need to extract the first
        if ids.shape[0] != cfg.batch_size:
            ids = pool_single_view(cfg, ids, 0)
        outputs['ids_1'] = ids

        # Second, build the decoder and projector
        key = 'ids' if predict_for_all else 'ids_1'
        decoder_out = self.decoder(outputs[key], outputs)
        pc = decoder_out['xyz']
        outputs['points_1'] = pc
        outputs['rgb_1'] = decoder_out['rgb']
        outputs['scaling_factor'] = self.scalePred(outputs[key], is_training)
        outputs['focal_length'] = self.focalPred(outputs['ids'], is_training)

        if cfg.predict_pose:
            pose_out = self.poseNet(enc_outputs['poses'])
            outputs.update(pose_out)

        if self._alignment_to_canonical is not None:
            outputs = align_predictions(outputs, self._alignment_to_canonical)

        return outputs

    def get_dropout_keep_prob(self):
        cfg = self.cfg()
        return get_dropout_prob(cfg, self._global_step)

    def compute_projection(self, inputs, outputs, is_training, summary_writer):
        cfg = self.cfg()
        all_points = outputs['all_points']
        all_rgb = outputs['all_rgb']

        if cfg.predict_pose:
            camera_pose = outputs['poses']
        else:
            if cfg.pose_quaternion:
                camera_pose = inputs['camera_quaternion']
            else:
                camera_pose = inputs['matrices']

        if is_training and cfg.pc_point_dropout != 1:
            dropout_prob = self.get_dropout_keep_prob()
            if is_training and summary_writer is not None:
                summary_writer.add_scalar("meta/pc_point_dropout_prob", dropout_prob)
            all_points, all_rgb = pc_point_dropout(all_points, all_rgb, dropout_prob)

        if cfg.pc_fast:
            predicted_translation = outputs["predicted_translation"] if cfg.predict_translation else None
            proj_out = pointcloud_project_fast(cfg, all_points, camera_pose, predicted_translation,
                                               all_rgb, self.gauss_kernel(),
                                               scaling_factor=outputs['all_scaling_factors'],
                                               focal_length=outputs['all_focal_length'])
            proj = proj_out["proj"]
            outputs["projs_rgb"] = proj_out["proj_rgb"]
            outputs["drc_probs"] = proj_out["drc_probs"]
            outputs["projs_depth"] = proj_out["proj_depth"]
        else:
            proj, voxels = pointcloud_project(cfg, all_points, camera_pose, self.gauss_sigma())
            outputs["projs_rgb"] = None
            outputs["projs_depth"] = None

        outputs['projs'] = proj

        batch_size = outputs['points_1'].shape[0]
        outputs['projs_1'] = proj[0:batch_size, :, :, :]

        return outputs

    def replicate_for_multiview(self, tensor):
        cfg = self.cfg()
        new_tensor = tf_repeat_0(tensor, cfg.step_size)
        return new_tensor

    def forward(self, inputs, global_step, is_training=True, run_projection=True):
        cfg = self._params
        if is_training:
            self.train()
        else:
            self.eval()
        self._global_step = global_step

        code = 'images' if cfg.predict_pose else 'images_1'
        outputs = self.model_predict(inputs[code], is_training)
        pc = outputs['points_1']

        if run_projection:
            all_points = self.replicate_for_multiview(pc)
            num_candidates = cfg.pose_predict_num_candidates
            all_focal_length = None
            if num_candidates > 1:
                all_points = tf_repeat_0(all_points, num_candidates)
                if cfg.predict_translation:
                    trans = outputs["predicted_translation"]
                    outputs["predicted_translation"] = tf_repeat_0(trans, num_candidates)
                focal_length = outputs['focal_length']
                if focal_length is not None:
                    all_focal_length = tf_repeat_0(focal_length, num_candidates)

            outputs['all_focal_length'] = all_focal_length
            outputs['all_points'] = all_points
            if cfg.pc_learn_occupancy_scaling:
                all_scaling_factors = self.replicate_for_multiview(outputs['scaling_factor'])
                if num_candidates > 1:
                    all_scaling_factors = tf_repeat_0(all_scaling_factors, num_candidates)
            else:
                all_scaling_factors = None
            outputs['all_scaling_factors'] = all_scaling_factors
            if cfg.pc_rgb:
                all_rgb = self.replicate_for_multiview(outputs['rgb_1'])
                if num_candidates > 1:
                    all_rgb = tf_repeat_0(all_rgb, num_candidates)
            else:
                all_rgb = None
            outputs['all_rgb'] = all_rgb
          
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for k in outputs.keys():
                try:
                    outputs[k] = outputs[k].to(device)
                except AttributeError:
                    pass
            outputs = self.compute_projection(inputs, outputs, is_training, self.summary_writer)

        return outputs


    def add_proj_loss(self, inputs, outputs, weight_scale, summary_writer, add_summary):
        cfg = self.cfg()
        gt = inputs['masks']
        pred = outputs['projs']
        num_samples = pred.shape[0]

        gt_size = gt.shape[2]
        pred_size = pred.shape[2]
        assert gt_size >= pred_size, "GT size should not be higher than prediction size"
        if gt_size > pred_size:
            n_dsmpl = gt_size//pred_size
            # if cfg.bicubic_gt_downsampling:
            #     interp_method = tf.image.ResizeMethod.BICUBIC
            # else:
            #     interp_method = tf.image.ResizeMethod.BILINEAR
            gt = nn.AvgPool2d(n_dsmpl)(gt)
        if cfg.pc_gauss_filter_gt:
            print("Not implemented")
            # import pdb
            # pdb.set_trace()
            # sigma_rel = self._sigma_rel
            # smoothed = gauss_smoothen_image(cfg, gt, sigma_rel)
            # if cfg.pc_gauss_filter_gt_switch_off:
            #     gt = tf.where(tf.less(sigma_rel, 1.0), gt, smoothed)
            # else:
            #     gt = smoothed
        total_loss = 0
        num_candidates = cfg.pose_predict_num_candidates
        if num_candidates > 1:
            proj_loss, min_loss = self.proj_loss_pose_candidates(gt, pred, inputs, summary_writer)
            if cfg.pose_predictor_student:
                student_loss = self.add_student_loss(inputs, outputs, min_loss, summary_writer, add_summary)
                total_loss += student_loss
        else:
            proj_loss = nn.MSELoss(gt - pred)
            proj_loss /= num_samples

        total_loss += proj_loss

        if add_summary:
            summary_writer.add_scalar("losses/proj_loss", proj_loss)

        total_loss *= weight_scale
        return total_loss

    def get_loss(self, inputs, outputs, summary_writer, add_summary=True):
        """Computes the loss used for PTN paper (projection + volume loss)."""
        cfg = self.cfg()
        g_loss = 0

        if cfg.proj_weight:
            g_loss += self.add_proj_loss(inputs, outputs, cfg.proj_weight, summary_writer, add_summary)
       # 
       # if cfg.drc_weight:
       #     g_loss += add_drc_loss(cfg, inputs, outputs, cfg.drc_weight, add_summary)
       # 
       # if cfg.pc_rgb:
       #     g_loss += add_proj_rgb_loss(cfg, inputs, outputs, cfg.proj_rgb_weight, add_summary, self._sigma_rel)
       # 
       # if cfg.proj_depth_weight:
       #     g_loss += add_proj_depth_loss(cfg, inputs, outputs, cfg.proj_depth_weight, self._sigma_rel, add_summary)
       # 
       # if add_summary:
       #     summary_writer.add_scalar("losses/total_task_loss", g_loss)
        
        return g_loss

    def proj_loss_pose_candidates(self, gt, pred, inputs, summary_writer):
        """
        :param gt: [BATCH*VIEWS, IM_SIZE, IM_SIZE, 1]
        :param pred: [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        :return: [], [BATCH*VIEWS]
        """
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates
        gt = tf_repeat_0(gt, num_candidates) # [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        sq_diff = (gt - pred)**2
        all_loss = sq_diff.sum((1,2,3))# [BATCH*VIEWS*CANDIDATES]
        all_loss = all_loss.reshape(-1, num_candidates) # [BATCH*VIEWS, CANDIDATES]
        min_loss = all_loss.argmin(1) # [BATCH*VIEWS]
        if summary_writer is not None:
            summary_writer.add_histogram("winning_pose_candidates", min_loss)

        min_loss_mask = one_hot(min_loss, num_candidates) # [BATCH*VIEWS, CANDIDATES]
        num_samples = min_loss_mask.shape[0]

        min_loss_mask_flat = min_loss_mask.reshape(-1) # [BATCH*VIEWS*CANDIDATES]
        min_loss_mask_final = min_loss_mask_flat.reshape(-1, 1, 1, 1) # [BATCH*VIEWS*CANDIDATES, 1, 1, 1]
        loss_tensor = (gt - pred) * min_loss_mask_final
        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
            weights = tf_repeat_0(weights, num_candidates)
            weights = weights.reshape(weights.shape[0], 1, 1, 1)
            loss_tensor *= weights
        proj_loss = (loss_tensor**2).sum()
        proj_loss /= num_samples

        return proj_loss, min_loss

    def add_student_loss(self, inputs, outputs, min_loss, summary_writer, add_summary):
        cfg = self.cfg()
        num_candidates = cfg.pose_predict_num_candidates

        student = outputs["pose_student"]
        teachers = outputs["poses"]
        teachers = teachers.reshape(-1, num_candidates, 4)

        indices = min_loss
        indices = indices.reshape(indices.shape[0],1)
        batch_size = teachers.shape[0]
        batch_indices = torch.arange(0, batch_size, 1).long()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_indices = batch_indices.reshape(batch_indices.shape[0],1).to(device)
        indices = torch.cat([batch_indices, indices], dim=1)
        teachers= teachers[indices.transpose(0,1).long().cpu().numpy().tolist()]
        # use teachers only as ground truth
        teachers = teachers.detach()

        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
        else:
            weights = 1.0

        if cfg.pose_student_align_loss:
            ref_pc = self._pc_for_alignloss
            num_ref_points = ref_pc.shape.as_list()[0]
            #import pdb
            #pdb.set_trace()
            ref_pc_all = tf.tile(tf.expand_dims(ref_pc, axis=0), [teachers.shape[0], 1, 1])
            pc_1 = q_rotate(ref_pc_all, teachers)
            pc_2 = q_rotate(ref_pc_all, student)
            student_loss = tf.nn.l2_loss(pc_1 - pc_2) / num_ref_points
        else:
            #import pdb
            #pdb.set_trace()
            q_diff = q_norm(q_mul(teachers, q_conj(student)))
            angle_diff = q_diff[:, 0]
            student_loss = ((1.0 - angle_diff**2) * weights).sum()

        num_samples = min_loss.shape[0]
        student_loss /= num_samples

        if add_summary:
            summary_writer.add_scalar("losses/pose_predictor_student_loss", student_loss)
        student_loss *= cfg.pose_predictor_student_loss_weight

        return student_loss
