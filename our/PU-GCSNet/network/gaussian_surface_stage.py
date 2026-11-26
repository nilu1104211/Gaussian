"""Stage-2 Gaussian surface generation for PU-GCSNet."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .geometry_utils import local_pca_frame, tangent_basis


class GaussianSurfaceStage(nn.Module):
    def __init__(self, feature_extractor, mode='quadratic', sampling='deterministic',
                 k=16, sigma_min=0.15, sigma_max=0.75, center_bound=0.2,
                 normal_bound=0.2, height_bound=0.25, rho=1.0):
        super().__init__()
        if mode not in ('quadratic', 'tangent', 'gaussian3d'):
            raise ValueError('Unknown Gaussian generation mode')
        if sampling not in ('deterministic', 'random', 'antithetic'):
            raise ValueError('Unknown sampling mode')
        if not 0 < sigma_min < sigma_max or min(center_bound, normal_bound, height_bound) < 0 or rho <= 0 or k < 2:
            raise ValueError('Invalid Gaussian geometry bounds')
        self.feature_extractor = feature_extractor
        self.mode, self.sampling, self.k = mode, sampling, k
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.center_bound, self.normal_bound = center_bound, normal_bound
        self.height_bound, self.rho = height_bound, rho
        # 3D control uses center(3), scale(3), rotation axis-angle(3), direction(2).
        channels = 11 if mode == 'gaussian3d' else 12
        self.param_head = nn.Sequential(nn.Conv1d(132, 128, 1), nn.ReLU(),
                                       nn.Conv1d(128, 64, 1), nn.ReLU(),
                                       nn.Conv1d(64, channels, 1))
        nn.init.normal_(self.param_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.param_head[-1].bias)
        # A non-axis initial angle gives both scales and all height terms gradients.
        if mode != 'gaussian3d':
            with torch.no_grad():
                self.param_head[-1].bias[8] = math.atanh(0.25)

    def sample(self, direction):
        if self.sampling == 'random':
            return torch.randn(*direction.shape[:-1], 2, direction.shape[-1], device=direction.device, dtype=direction.dtype) * self.rho
        if self.sampling == 'antithetic':
            direction = torch.randn_like(direction)
        return torch.stack((direction, -direction), dim=-2) * self.rho

    def forward(self, points):
        return self.forward_with_aux(points)[0]

    def prepare_features(self, feature, frame, parent_features):
        return feature, {}

    def adjust_children(self, uv, feature, frame, extra):
        return uv

    def adjust_height(self, height, frame, extra):
        return height

    def finish_children(self, feature, uv, frame, extra):
        return extra

    def compute_frame(self, xyz):
        return local_pca_frame(xyz, self.k)

    def forward_with_aux(self, points, parent_features=None):
        feature = self.feature_extractor(points)
        xyz = points.transpose(1, 2)
        frame = self.compute_frame(xyz)
        feature, extra = self.prepare_features(feature, frame, parent_features)
        scale = frame['scale']
        geometry = torch.cat((frame['ratios'], scale), dim=-1)
        raw = self.param_head(torch.cat((feature, geometry.transpose(1, 2)), dim=1)).transpose(1, 2)
        mu = xyz + self.center_bound * scale * raw[..., :3].tanh()
        if self.mode == 'gaussian3d':
            sigma = scale * (self.sigma_min + (self.sigma_max - self.sigma_min) * raw[..., 3:6].sigmoid())
            axis = math.pi * raw[..., 6:9].tanh()
            zero = torch.zeros_like(axis[..., 0])
            x, y, z = axis.unbind(-1)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*axis.shape[:-1], 3, 3)
            rotation = torch.matrix_exp(skew)
            theta = math.pi * raw[..., 9].tanh()
            phi = math.pi / 4 + (math.pi / 2) * raw[..., 10].tanh()
            direction = torch.stack((phi.cos() * theta.cos(), phi.cos() * theta.sin(), phi.sin()), dim=-1)
            offsets = sigma.unsqueeze(-2) * self.sample(direction)
            generated = mu.unsqueeze(-2) + offsets @ rotation.transpose(-1, -2)
            aux = dict(mu=mu, sigma=sigma, local_scale=scale)
        else:
            sigma = scale * (self.sigma_min + (self.sigma_max - self.sigma_min) * raw[..., 3:5].sigmoid())
            normal = F.normalize(frame['normal'] + self.normal_bound * raw[..., 5:8].tanh(), dim=-1)
            tangent, bitangent = tangent_basis(normal, frame['tangent'])
            theta = math.pi * raw[..., 8:9].tanh()
            direction = torch.cat((theta.cos(), theta.sin()), dim=-1)
            uv = sigma.unsqueeze(-2) * self.sample(direction)
            uv = self.adjust_children(uv, feature, frame, extra)
            u, v = uv.unbind(-1)
            abc = self.height_bound * raw[..., 9:12].tanh()
            if self.mode == 'tangent':
                abc = torch.zeros_like(abc)
            a, b, c = abc.split(1, dim=-1)
            height = scale * (a * (u / scale).square() + b * (u / scale) * (v / scale) + c * (v / scale).square())
            height = self.adjust_height(height, frame, extra)
            generated = (mu.unsqueeze(-2) + u[..., None] * tangent.unsqueeze(-2)
                         + v[..., None] * bitangent.unsqueeze(-2) + height[..., None] * normal.unsqueeze(-2))
            aux = dict(mu=mu, n_pred=normal, t1=tangent, t2=bitangent, sigma=sigma,
                       abc=abc, local_scale=scale, height=height, frame_valid=frame['valid'],
                       pca_ratios=frame['ratios'])
            aux.update(self.finish_children(feature, uv, frame, extra))
        count = xyz.shape[1]
        aux['parent_idx'] = torch.arange(count, device=points.device).repeat_interleave(2)[None].expand(points.shape[0], -1)
        aux['child_idx'] = torch.arange(2, device=points.device).repeat(count)[None].expand(points.shape[0], -1)
        return generated.flatten(1, 2).transpose(1, 2).contiguous(), aux
