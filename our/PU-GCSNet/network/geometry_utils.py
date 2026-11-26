"""Detached local geometry, evaluated from the current point coordinates."""
import torch
import torch.nn.functional as F


def canonical_sign(vector):
    component = vector.gather(-1, vector.abs().argmax(-1, keepdim=True))
    return vector * torch.where(component < 0, -1.0, 1.0)


def tangent_basis(normal, preferred=None):
    axis = F.one_hot(normal.abs().argmin(-1), num_classes=3).to(normal.dtype)
    fallback = axis - (axis * normal).sum(-1, keepdim=True) * normal
    candidate = fallback if preferred is None else preferred - (preferred * normal).sum(-1, keepdim=True) * normal
    candidate = torch.where(candidate.norm(dim=-1, keepdim=True) > 1e-6, candidate, fallback)
    tangent = F.normalize(candidate, dim=-1, eps=1e-8)
    return tangent, F.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)


@torch.no_grad()
def local_pca_frame(points, k=16, chunk_size=256, return_neighbors=False):
    """points [B,N,3]. Exclude self by index, including for coincident points.

    A canonical frame handles zero neighborhoods. Line neighborhoods use a
    perpendicular to the principal direction. Isotropic planes use a fixed
    projected axis because their two tangent eigenvectors are ambiguous.
    """
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 2:
        raise ValueError("PCA expects [B,N,3] with N >= 2")
    if k < 2:
        raise ValueError("PCA neighborhood must contain at least two points")
    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.detach().float()
        count = points.shape[1]
        k = min(k, count - 1)
        batch = torch.arange(points.shape[0], device=points.device)[:, None, None]
        neighbors = []
        for start in range(0, count, chunk_size):
            query = points[:, start:start + chunk_size]
            distance = torch.cdist(query, points)
            row = torch.arange(query.shape[1], device=points.device)
            distance[:, row, row + start] = float('inf')
            indices = distance.topk(k, largest=False, sorted=True).indices
            neighbors.append(points[batch, indices])
        neighbors = torch.cat(neighbors, dim=1)
        centered = neighbors - neighbors.mean(2, keepdim=True)
        covariance = centered.transpose(-2, -1) @ centered / k
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0)
        largest = eigenvalues[..., 2:3]
        valid = (largest > 1e-12) & (eigenvalues[..., 1:2] > largest * 1e-5)
        principal = canonical_sign(vectors[..., 2])
        line_normal, _ = tangent_basis(principal)
        default_normal = torch.zeros_like(principal)
        default_normal[..., 2] = 1
        fallback = torch.where(largest > 1e-12, line_normal, default_normal)
        normal = canonical_sign(torch.where(valid, vectors[..., 0], fallback))
        tangent = canonical_sign(vectors[..., 2])
        isotropic = (eigenvalues[..., 2:3] - eigenvalues[..., 1:2]) <= largest * 1e-4
        stable_tangent, _ = tangent_basis(normal)
        tangent = torch.where(isotropic | ~valid, stable_tangent, tangent)
        tangent, bitangent = tangent_basis(normal, tangent)
        scale = (neighbors - points[:, :, None]).square().sum(-1).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        ratios = eigenvalues / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-12)
    result = dict(normal=normal, tangent=tangent, bitangent=bitangent,
                  scale=scale, ratios=ratios, valid=valid.float())
    if return_neighbors:
        result['relative_neighbors'] = (neighbors - points[:, :, None]) / scale.unsqueeze(-2)
    return result
