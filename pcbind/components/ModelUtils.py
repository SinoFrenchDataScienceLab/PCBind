import torch
from torch.nn import Linear
import torch.nn as nn

from torchdrug import layers


class Transition(torch.nn.Module):
    def __init__(self, hidden_dim=128, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(hidden_dim)
        self.linear_1 = Linear(hidden_dim, n * hidden_dim)
        self.linear_2 = Linear(n * hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.linear_2((self.linear_1(x)).relu())
        return x


class InteractionModule(torch.nn.Module):
    # TODO: test opm False and True
    def __init__(self, node_hidden_dim, pair_hidden_dim, hidden_dim, opm=False):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.opm = opm

        self.layer_norm_p = nn.LayerNorm(node_hidden_dim)
        self.layer_norm_c = nn.LayerNorm(node_hidden_dim)

        if self.opm:
            self.linear_p = Linear(node_hidden_dim, hidden_dim)
            self.linear_c = Linear(node_hidden_dim, hidden_dim)
            self.linear_out = Linear(hidden_dim ** 2, pair_hidden_dim)
        else:
            self.linear_out = Linear(node_hidden_dim, pair_hidden_dim)

    def forward(self, p_embed, c_embed,
                p_mask=None, c_mask=None):
        # mask
        if p_mask is None:
            p_mask = p_embed.new_ones(p_embed.shape[:-1], dtype=torch.bool)
        if c_mask is None:
            c_mask = c_embed.new_ones(c_embed.shape[:-1], dtype=torch.bool)
        inter_mask = torch.einsum("...i,...j->...ij", p_mask, c_mask)  # (Np, Nc)

        p_embed = self.layer_norm_p(p_embed)  # (Np, C_node)
        c_embed = self.layer_norm_c(c_embed)  # (Nc, C_node)
        if self.opm:
            p_embed = self.linear_p(p_embed)  # (Np, C_hidden)
            c_embed = self.linear_c(c_embed)  # (Nc, C_hidden)
            inter_embed = torch.einsum("...bc,...de->...bdce", p_embed, c_embed)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        else:
            inter_embed = torch.einsum("...ik,...jk->...ijk", p_embed, c_embed)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        return inter_embed, inter_mask


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist[..., None] - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RBFDistanceModule(torch.nn.Module):
    def __init__(self, rbf_stop, distance_hidden_dim, num_gaussian=32, dropout=0.1):
        super(RBFDistanceModule, self).__init__()
        self.distance_hidden_dim = distance_hidden_dim
        self.rbf = GaussianSmearing(start=0, stop=rbf_stop, num_gaussians=num_gaussian)
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussian, distance_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(distance_hidden_dim, distance_hidden_dim)
        )

    def forward(self, distance):
        return self.mlp(self.rbf(distance))  # (..., C_hidden)


def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30, additional_mask=None, num_classses=16,
                         tankbind_style=False):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis = pair_dis.clamp(bin_min, bin_max)
    if additional_mask is not None:
        pair_dis[additional_mask] = bin_max
    if tankbind_style is True:
        pair_dis = pair_dis * (1 - torch.eye(pair_dis.shape[-1], device=pair_dis.device)[None:])
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=num_classses)
    return pair_dis_one_hot


def get_heter_pair_dis_one_hot(d1, d2, bin_size, bin_min, bin_max, num_classses=16):
    pair_dis = torch.cdist(d1, d2, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis = pair_dis.clamp(bin_min, bin_max)
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=num_classses)
    return pair_dis_one_hot


def get_dis_one_hot(d, bin_size, bin_min, bin_max):
    d = d.clamp(bin_min, bin_max)
    dis_bin_index = torch.div(d - bin_min, bin_size, rounding_mode='floor').long()
    dis_one_hot = torch.nn.functional.one_hot(dis_bin_index, num_classes=16)
    return dis_one_hot
