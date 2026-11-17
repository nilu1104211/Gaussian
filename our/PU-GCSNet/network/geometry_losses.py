"""GT geometry is used exclusively as a detached training target."""
import torch

from .geometry_utils import local_pca_frame
from .point_ops import nearest_indices


def gt_geometry(gt, k=24, confidence_tau=0.0):
    frame = local_pca_frame(gt, k)
    weight = frame['valid'].squeeze(-1)
    if confidence_tau > 0:
        weight = weight * (1 - frame['ratios'][..., 0] / confidence_tau).clamp(0, 1)
    return frame['normal'], weight


def match_geometry(query, gt, normals, weights):
    indices = nearest_indices(query.detach(), gt.detach())
    indices3 = indices[..., None].expand(-1, -1, 3)
    return gt.gather(1, indices3), normals.gather(1, indices3), weights.gather(1, indices)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def surface_loss(points, gt, normals, weights):
    closest, normal, weight = match_geometry(points, gt, normals, weights)
    return weighted_mean(((points - closest) * normal).sum(-1).square(), weight)


def normal_loss(centers, predicted_normals, gt, normals, weights):
    _, matched_normals, weight = match_geometry(centers, gt, normals, weights)
    cosine = (predicted_normals * matched_normals).sum(-1).clamp(-1, 1)
    return weighted_mean(1 - cosine.square(), weight)
