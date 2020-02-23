import numpy as np
import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class PoseBranch(nn.Module):
    def __init__(self, cfg):
        super(PoseBranch, self).__init__()
        self.cfg = cfg
        f_dim = 32
        lReLU = nn.LeakyReLU()
        num_layers = cfg.pose_candidates_num_layers
        layers = []
        for k in range(num_layers):
            if k == (num_layers - 1):
                out_dim = 4
                inp_dim = f_dim
                act_func = None
            else:
                out_dim = f_dim
                inp_dim = f_dim
                act_func = lReLU
            if k == 0:
                inp_dim = cfg.z_dim
            layers.append(nn.Linear(inp_dim, out_dim))
            if not act_func is None:
                layers.append(act_func)
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)


class PoseNet(nn.Module):
    def __init__(self, cfg):
        super(PoseNet, self).__init__()
        z_dim = cfg.z_dim
        self.num_candidates = cfg.pose_predict_num_candidates
        if self.num_candidates > 1:
            candidate_fcs = []
            for _ in range(self.num_candidates):
                candidate_fcs.append(PoseBranch(cfg))
            self.candidate_fcs = nn.ModuleList(candidate_fcs)
            self.student_fc = PoseBranch(cfg)
        else:
            self.single_candidate_fc = nn.Linear(z_dim, 4)
            self.single_candidate_fc.apply(init_weights)

        if cfg.predict_translation:
            trans_init_stddev = cfg.predict_translation_init_stddev
            self.trans_fc = nn.Linear(z_dim, 3)
            self.trans_fc.apply(init_weights)  ## TODO: Original implementation uses truncated normal intialization here
        self.tanh = nn.Tanh()
        self.translation_scaling = cfg.predict_translation_scaling_factor
        self.cfg = cfg

    def forward(self, inputs):
        out = {}
        if self.num_candidates > 1:
            outs = [self.candidate_fcs[i](inputs) for i in range(self.num_candidates)]
            q = torch.cat(outs, dim=1)
            q = q.reshape(-1, 4)  ## TODO: Is this correct?
            if self.cfg.pose_predictor_student:
                out["pose_student"] = self.student_fc(inputs)
        else:
            q = self.single_candidate_fc(inputs)

        if self.cfg.predict_translation:
            t = self.trans_fc(inputs)
            if self.cfg.predict_translation_tanh:
                t = self.tanh(t) * self.translation_scaling
        else:
            t = None
        out["poses"] = q
        out["predicted_translation"] = t
        return out
