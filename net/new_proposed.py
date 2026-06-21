from __future__ import annotations

import functools
from typing import List

import torch
import torch.nn as nn

from net.mamba import SS_Conv_SSM
from net.feature_extractor import Feature_Extractor, ConvMixer, SB
from net.glca import ChannelAttention


class CovaBlock(nn.Module):
    """Covariance metric layer used by SC-MambaFew."""

    def __init__(self):
        super().__init__()

    def cal_covariance(self, support_features: List[torch.Tensor]):
        cova_matrix_list = []
        for support_set_sam in support_features:
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3).contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, dim=1, keepdim=True)
            support_set_sam = support_set_sam - mean_support
            covariance_matrix = support_set_sam @ support_set_sam.t()
            covariance_matrix = covariance_matrix / (h * w * B - 1)
            cova_matrix_list.append(covariance_matrix)
        return cova_matrix_list

    def cal_similarity(self, query_feature: torch.Tensor, cova_matrix_list):
        B, C, h, w = query_feature.size()
        cova_sim = []
        device = query_feature.device

        for i in range(B):
            query_sam = query_feature[i].view(C, -1)
            query_norm = torch.norm(query_sam, 2, dim=1, keepdim=True).clamp_min(1e-12)
            query_sam = query_sam / query_norm

            mea_sim = torch.zeros(1, len(cova_matrix_list) * h * w, device=device)
            for j, cova in enumerate(cova_matrix_list):
                temp_dis = query_sam.t() @ cova @ query_sam
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()
            cova_sim.append(mea_sim.view(1, -1))

        return torch.cat(cova_sim, dim=0)

    def forward(self, query_feature: torch.Tensor, support_features: List[torch.Tensor]):
        cova_matrix_list = self.cal_covariance(support_features)
        return self.cal_similarity(query_feature, cova_matrix_list)


class MainNet(nn.Module):
    """SC-MambaFew backbone for input-representation experiments.

    Parameters
    ----------
    h, w:
        Feature-map size after patch embedding. For input resolution 64, h=w=16.
        For input resolution 128, h=w=32.
    """

    def __init__(self, h: int = 16, w: int = 16, c: int = 64, dim: int = 64,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        self.c = int(c)
        input_res = self.h * 4

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features1 = ConvMixer(patch_size=4)
        self.features2 = Feature_Extractor(input_res=input_res)
        self.upper = SS_Conv_SSM(hidden_dim=dim, d_state=16)
        self.lower = ChannelAttention(dim // 4, 3)
        self.SB = SB(64)
        self.covariance = CovaBlock()

        self.classifier1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=self.h * self.w, stride=self.h * self.w, bias=use_bias),
        )

    def extract_fused_feature(self, x: torch.Tensor):
        f1 = self.upper(self.features1(x))
        f2 = self.lower(self.features2(x))
        fused, attention_vectors = self.SB(f1, f2)
        return fused, attention_vectors

    def forward(self, input1: torch.Tensor, input2: List[torch.Tensor]):
        query_feature, vec_q = self.extract_fused_feature(input1)

        support_features = []
        vec_s = None
        for support in input2:
            support_feature, vec_s = self.extract_fused_feature(support)
            support_features.append(support_feature)

        cova_score = self.covariance(query_feature, support_features)
        out = self.classifier1(cova_score.view(cova_score.size(0), 1, -1))
        output = out.squeeze(1)
        return output, vec_q, vec_s


# Optional ablation scaffold retained for compatibility with older experiments.
class Baseline(MainNet):
    pass
