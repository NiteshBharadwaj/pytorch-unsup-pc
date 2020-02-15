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

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        target_spatial_size = 4
        f_dim = cfg.f_dim
        fc_dim = cfg.fc_dim
        z_dim = cfg.z_dim
        image_size = cfg.input_shape[0]
        image_channels = cfg.input_shape[2]
        act_fun = nn.LeakyReLU()
        num_blocks = int(np.log2(image_size / target_spatial_size) - 1)
        conv_layers = []
        conv_layers.append(nn.Conv2d(image_channels, f_dim, (5,5),stride=2, padding=2))
        conv_layers.append(act_fun)
        for k in range(num_blocks):
            f_dim = f_dim*2
            conv_layers.append(nn.Conv2d(f_dim//2,f_dim,(3,3),stride=2,padding=1))
            conv_layers.append(act_fun)
            conv_layers.append(nn.Conv2d(f_dim,f_dim,(3,3), stride=1, padding=1))
            conv_layers.append(act_fun)
        self.conv_layers = nn.Sequential(*conv_layers)

        self.fc1 = nn.Sequential(nn.Linear(f_dim*target_spatial_size*target_spatial_size,fc_dim), act_fun)
        self.fc2 = nn.Sequential(nn.Linear(fc_dim,fc_dim), act_fun)
        self.fc3 = nn.Sequential(nn.Linear(fc_dim,z_dim), act_fun)
        if cfg.predict_pose:
            self.pose_fc = nn.Linear(fc_dim,z_dim)
            self.pose_fc.apply(init_weights)

        self.conv_layers.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)

    def forward(self, x):
        x = self._preprocess(x)
        batch_size = x.shape[0]
        x = self.conv_layers(x)
        x = x.reshape(batch_size,-1)
        outputs = {}
        outputs["conv_features"] = x
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        outputs["z_latent"] = fc1_out
        outputs["ids"] = fc3_out
        if self.cfg.predict_pose:
            outputs["poses"] = self.pose_fc(fc2_out)
        return outputs

    def _preprocess(self, images):
        return images * 2 - 1
