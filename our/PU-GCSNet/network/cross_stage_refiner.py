"""Correspondence-preserving feature fusion and local residual refinement."""
import math

import torch
from torch import nn

from .geometry_feature_bridge import GEO_CHANNELS, zero_last
from .point_ops import gather_operation, grouping_operation


def gather_parent(values, indices):
    """[B,N,C] -> [B,2N,C] using explicit Stage-2 correspondence."""
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))


class CrossStageRefiner(nn.Module):
    def __init__(self, use_f2_skip=True, use_geo_skip=False, use_neighborhood_refine=False,
                 use_tangent_normal_head=False, tangent_bound=0.1, normal_bound=0.05, k=16):
        super().__init__()
        if k < 2 or not all(math.isfinite(v) and v > 0 for v in (tangent_bound, normal_bound)):
            raise ValueError('Invalid cross-stage refinement bounds/neighborhood')
        self.use_f2_skip, self.use_geo_skip = use_f2_skip, use_geo_skip
        self.k, self.tangent_bound, self.normal_bound = k, tangent_bound, normal_bound
        channels = 128 + (128 if use_f2_skip else 0) + (128 + GEO_CHANNELS if use_geo_skip else 0)
        if use_neighborhood_refine:
            self.neighborhood = nn.Sequential(nn.Conv2d(131, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 1))
            channels += 64
        self.fusion = zero_last(nn.Sequential(nn.Conv1d(channels, 128, 1), nn.ReLU(), nn.Conv1d(128, 128, 1)))
        self.fusion_gate = nn.Conv1d(channels, 128, 1)
        if use_tangent_normal_head:
            self.residual_head = zero_last(nn.Sequential(nn.Conv1d(128 + GEO_CHANNELS, 64, 1), nn.ReLU(), nn.Conv1d(64, 3, 1)))
            self.residual_gate = nn.Conv1d(128 + GEO_CHANNELS, 1, 1)

    def fuse(self, base, points, aux):
        parts = [base]
        indices = aux['parent_idx']
        if self.use_f2_skip:
            parts.append(aux['f2'])
        if self.use_geo_skip:
            parts.extend((gather_operation(aux['parent_features'], indices),
                          gather_parent(aux['geometry_descriptor'], indices).transpose(1, 2)))
        if hasattr(self, 'neighborhood'):
            # Chunked NN queries avoid allocating a full 1024 x 1024 matrix.
            xyz = points.transpose(1, 2).detach()
            with torch.no_grad():
                neighbors = torch.cat([torch.cdist(chunk, xyz).topk(min(self.k, xyz.shape[1]), largest=False).indices
                                       for chunk in xyz.split(128, 1)], 1)
            scale = gather_parent(aux['local_scale'], indices).transpose(1, 2).unsqueeze(-1)
            relative = (grouping_operation(points, neighbors) - points.unsqueeze(-1)) / scale.clamp_min(1e-8)
            difference = grouping_operation(aux['f2'], neighbors) - aux['f2'].unsqueeze(-1)
            parts.append(self.neighborhood(torch.cat((relative, difference), 1)).amax(-1))
        joined = torch.cat(parts, 1)
        return base + self.fusion_gate(joined).sigmoid() * self.fusion(joined)

    def residual(self, features, aux):
        indices = aux['parent_idx']
        if not hasattr(self, 'residual_head'):
            return features.new_zeros(features.shape[0], 3, features.shape[-1])
        geometry = gather_parent(aux['geometry_descriptor'], indices).transpose(1, 2)
        joined = torch.cat((features, geometry), 1)
        raw = self.residual_head(joined).transpose(1, 2).tanh()
        confidence = gather_parent(aux['geometry_confidence'], indices)
        normal = gather_parent(aux['n_pred'], indices)
        tangent = gather_parent(aux['t1'], indices)
        bitangent = gather_parent(aux['t2'], indices)
        scale = gather_parent(aux['local_scale'], indices)
        # Confidence narrows only the added normal correction; original E6
        # learned displacement remains available even on unreliable geometry.
        delta = scale * (self.tangent_bound * (raw[..., :1] * tangent + raw[..., 1:2] * bitangent)
                         + self.normal_bound * confidence * raw[..., 2:3] * normal)
        return (self.residual_gate(joined).sigmoid().transpose(1, 2) * delta).transpose(1, 2)
