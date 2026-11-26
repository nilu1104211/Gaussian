"""E6 surface generator with bounded learned children and aligned features."""
import math

import torch
from torch import nn

from .gaussian_surface_stage import GaussianSurfaceStage
from .geometry_utils import local_pca_frame
from .geometry_feature_bridge import GeometryFeatureBridge, geometry_descriptor, GEO_CHANNELS, zero_last


class LearnableSurfaceGenerator(GaussianSurfaceStage):
    def __init__(self, feature_extractor, *, feature_bridge=False, learned_tangent=False,
                 learned_normal=False, geometry_gate=False, child_features=False,
                 tangent_bound=0.25, learned_normal_bound=0.1, **options):
        super().__init__(feature_extractor, **options)
        if self.mode != 'quadratic' or self.sampling != 'deterministic':
            raise ValueError('New surface modules require deterministic quadratic Stage-2')
        if not all(math.isfinite(v) and v > 0 for v in (tangent_bound, learned_normal_bound)):
            raise ValueError('Learned displacement bounds must be finite and positive')
        self.learned_tangent, self.learned_normal = learned_tangent, learned_normal
        self.tangent_bound, self.learned_normal_bound = tangent_bound, learned_normal_bound
        # Preserve the RNG stream of all original E6 backbone initializations.
        with torch.random.fork_rng(devices=[]):
            if feature_bridge:
                self.feature_bridge = GeometryFeatureBridge()
            if geometry_gate:
                self.confidence_head = nn.Sequential(nn.Conv1d(128 + GEO_CHANNELS, 64, 1), nn.ReLU(), nn.Conv1d(64, 1, 1))
                nn.init.zeros_(self.confidence_head[-1].weight)
                nn.init.constant_(self.confidence_head[-1].bias, 4.)
            if learned_tangent or learned_normal or child_features:
                self.child_embedding = nn.Embedding(2, 8)
            if learned_tangent or learned_normal:
                outputs = 2 * int(learned_tangent) + int(learned_normal)
                self.offset_head = zero_last(nn.Sequential(nn.Conv1d(138, 64, 1), nn.ReLU(), nn.Conv1d(64, outputs, 1)))
            if child_features:
                self.child_encoder = nn.Sequential(nn.Conv1d(140, 128, 1), nn.ReLU(), nn.Conv1d(128, 128, 1))

    def compute_frame(self, xyz):
        return local_pca_frame(xyz, self.k, return_neighbors=True)

    def prepare_features(self, feature, frame, parent_features):
        geometry, coefficients, residual = geometry_descriptor(frame)
        parent = feature if parent_features is None else parent_features
        if hasattr(self, 'feature_bridge'):
            if parent_features is None or parent_features.shape != feature.shape:
                raise ValueError('Feature bridge needs Stage-1 features aligned with P1')
            # Zero-initialized bridge affects the actual E6 parameter head, while
            # retaining its own Stage-2 extractor and original feature values.
            bridged = self.feature_bridge(parent, geometry)
            feature = feature + (bridged - parent)
            parent = bridged
        confidence = (self.confidence_head(torch.cat((feature, geometry.transpose(1, 2)), 1)).sigmoid().transpose(1, 2)
                      if hasattr(self, 'confidence_head') else torch.ones_like(frame['scale']))
        return feature, dict(geometry_descriptor=geometry, quad_fit=coefficients,
                             fit_residual=residual, geometry_confidence=confidence,
                             parent_features=parent)

    def adjust_children(self, uv, feature, frame, extra):
        if not hasattr(self, 'child_embedding'):
            return uv
        batch, parents, _, _ = uv.shape
        expanded = feature.repeat_interleave(2, -1)
        ids = torch.arange(2, device=uv.device).repeat(parents)
        embedding = self.child_embedding(ids).T[None].expand(batch, -1, -1)
        normalized = (uv / frame['scale'].unsqueeze(-2)).flatten(1, 2).transpose(1, 2)
        if hasattr(self, 'offset_head'):
            predicted = self.offset_head(torch.cat((expanded, embedding, normalized), 1)).transpose(1, 2).reshape(batch, parents, 2, -1).tanh()
            tangent = predicted[..., :2] if self.learned_tangent else torch.zeros_like(uv)
            normal = predicted[..., -1:] if self.learned_normal else torch.zeros_like(uv[..., :1])
            delta = torch.cat((tangent, normal), -1)
        else:
            delta = uv.new_zeros(batch, parents, 2, 3)
        extra['child_embedding'] = embedding
        extra['learned_child_delta'] = delta
        if self.learned_tangent:
            uv = uv + self.tangent_bound * frame['scale'].unsqueeze(-2) * delta[..., :2]
        return uv

    def adjust_height(self, height, frame, extra):
        height = height * extra['geometry_confidence']
        if self.learned_normal:
            height = height + self.learned_normal_bound * frame['scale'] * extra['learned_child_delta'][..., 2]
        return height

    def finish_children(self, feature, uv, frame, extra):
        if hasattr(self, 'child_encoder'):
            extra['f2'] = self.child_encoder(torch.cat((
                feature.repeat_interleave(2, -1), extra['child_embedding'],
                (uv / frame['scale'].unsqueeze(-2)).flatten(1, 2).transpose(1, 2),
                extra['learned_child_delta'][..., 2:3].flatten(1, 2).transpose(1, 2),
                extra['geometry_confidence'].repeat_interleave(2, 1).transpose(1, 2)), 1))
        return extra
