import numpy as np
import torch.nn as nn
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        init_stddev = cfg.pc_decoder_init_stddev
        input_dim = cfg.fc_dim
        num_points = cfg.pc_num_points
        self.pts_raw_fc = nn.Linear(input_dim, num_points * 3)
        self.pts_raw_fc.apply(init_weights)  ## TODO: Original implementation uses truncated normal intialization here
        self.tanh = nn.Tanh()
        lReLU = nn.LeakyReLU()
        self.rgb_deep_decoder = nn.Sequential(nn.Linear(input_dim, input_dim), lReLU, nn.Linear(input_dim, input_dim),
                                              lReLU,
                                              nn.Linear(input_dim, input_dim), lReLU)
        self.rgb_deep_decoder.aply(init_weights)

        self.rgb_raw_dec = nn.Linear(input_dim, num_points * 3)
        self.rgb_raw_dec.apply(init_weights)  ## TODO: Original implementation uses truncated normal intialization here
        self.sigmoid = nn.Sigmoid()
        self.cfg = cfg
        self.num_points = num_points

    def forward(self, inputs, enc_all):
        pts_raw = self.pts_raw_fc(inputs)
        pred_pts = pts_raw.reshape((pts_raw.shape[0], self.num_points, 3))
        pred_pts = self.tanh(pred_pts)
        if self.cfg.pc_unit_cube:
            pred_pts = pred_pts / 2.0
        out = {}
        out["xyz"] = pred_pts
        if self.cfg.pc_rgb:
            if self.cfg.pc_rgb_deep_decoder:
                inp = enc_all["conv_features"]
                inp = self.rgb_deep_decoder(inp)
            else:
                inp = inputs
            rgb_raw = self.rgb_raw_dec(inp)
            rgb = rgb_raw.reshape((rgb_raw.shape[0], self.num_points, 3))
            rgb = self.sigmoid(rgb)
        else:
            rgb = None
        out["rgb"] = rgb
        return out
