# MIT License
#
# Copyright (c) 2018 Chen-Hsuan Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

def point_cloud_distance(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt
    """
    VsN = Vs.shape[0]
    VtN = Vt.shape[0]
    Vt_rep = Vt.unsqueeze(0).repeat(VsN,1,1)  # [VsN,VtN,3]
    Vs_rep = Vs.unsqueeze(1).repeat(1,VtN,1) # [VsN,VtN,3]
    diff = Vt_rep-Vs_rep
    dist = torch.sqrt(torch.sum(diff**2, 2))  # [VsN,VtN]
    idx = torch.argmin(dist, dim=1).long()
    act_idxs = torch.stack([torch.arange(VsN).cuda(), idx], dim=1)
    proj = Vt_rep[act_idxs[:,0],act_idxs[:,1]]
    minDist = dist[act_idxs[:,0],act_idxs[:,1]]
    return proj, minDist, idx
