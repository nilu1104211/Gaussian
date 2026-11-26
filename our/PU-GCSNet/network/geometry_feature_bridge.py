"""Input-only, detached geometry and a zero-initialized feature bridge.

Fitted coefficients use dimensionless PCA coordinates (neighbor offsets / RMS
kNN radius). Scalar descriptors are invariant to PCA axis sign flips. The
coordinate generator still inherits E6's frame ambiguity near repeated roots;
these descriptors do not claim full rotation equivariance of the network.
"""
import torch
from torch import nn

GEO_CHANNELS = 13


def zero_last(module):
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


@torch.no_grad()
def geometry_descriptor(frame):
    with torch.autocast(device_type=frame['scale'].device.type, enabled=False):
        relative = frame['relative_neighbors'].float()
        x = (relative * frame['tangent'].unsqueeze(-2)).sum(-1)
        y = (relative * frame['bitangent'].unsqueeze(-2)).sum(-1)
        z = (relative * frame['normal'].unsqueeze(-2)).sum(-1)
        design = torch.stack((x.square(), x * y, y.square(), x, y, torch.ones_like(x)), -1)
        eye = torch.eye(6, device=x.device)
        # Ridge regularization also handles coincident/collinear neighborhoods.
        gram = design.transpose(-1, -2) @ design / x.shape[-1]
        rhs = (design.transpose(-1, -2) @ z.unsqueeze(-1)) / x.shape[-1]
        coefficients = torch.linalg.solve(gram + 1e-4 * eye, rhs).squeeze(-1)
        residual = ((design * coefficients.unsqueeze(-2)).sum(-1) - z).square().mean(-1, keepdim=True).clamp_min(0).sqrt()
        a, b, c, d, e, f = coefficients.unbind(-1)
        quad = torch.stack((a, b / 2, b / 2, c), -1).reshape(*a.shape, 2, 2)
        curvature = torch.linalg.eigvalsh(quad).abs().sort(-1).values
        distance = relative.norm(dim=-1)
        cv = distance.std(-1, unbiased=False, keepdim=True) / distance.mean(-1, keepdim=True).clamp_min(1e-8)
        ratios = frame['ratios']
        gaps = torch.cat((ratios[..., 1:2] - ratios[..., :1], ratios[..., 2:3] - ratios[..., 1:2]), -1)
        descriptor = torch.cat((ratios, frame['scale'].clamp_min(1e-8).log(), cv,
                                gaps, frame['valid'], residual, curvature,
                                (d.square() + e.square()).sqrt().unsqueeze(-1), f.abs().unsqueeze(-1)), -1)
    return descriptor, coefficients, residual


class GeometryFeatureBridge(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.residual = zero_last(nn.Sequential(nn.Conv1d(channels + GEO_CHANNELS, channels, 1),
                                                nn.ReLU(), nn.Conv1d(channels, channels, 1)))
        self.gate = nn.Conv1d(channels + GEO_CHANNELS, channels, 1)

    def forward(self, features, geometry):
        joined = torch.cat((features, geometry.transpose(1, 2)), 1)
        return features + 0.1 * self.gate(joined).sigmoid() * self.residual(joined)
