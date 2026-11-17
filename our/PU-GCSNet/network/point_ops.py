"""PointNet++ operators with a differentiable PyTorch reference backend."""
import warnings

import torch

try:
    from pointnet2_ops import pointnet2_utils as _cuda_ops
except (ImportError, OSError):
    _cuda_ops = None
    warnings.warn("pointnet2_ops unavailable, using the slower PyTorch reference backend.")


def grouping_operation(features, indices):
    if _cuda_ops is not None and features.is_cuda:
        return _cuda_ops.grouping_operation(features.contiguous(), indices.int().contiguous())
    batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
    grouped = features.transpose(1, 2)[batch, indices.long()]
    return grouped.permute(0, 3, 1, 2).contiguous()


def gather_operation(features, indices):
    if _cuda_ops is not None and features.is_cuda:
        return _cuda_ops.gather_operation(features.contiguous(), indices.int().contiguous())
    return features.gather(2, indices.long()[:, None].expand(-1, features.shape[1], -1))


@torch.no_grad()
def furthest_point_sample(points, count):
    if not 0 < count <= points.shape[1]:
        raise ValueError("FPS count must be between 1 and the number of input points")
    if _cuda_ops is not None and points.is_cuda:
        return _cuda_ops.furthest_point_sample(points.contiguous(), count)
    batch, size, _ = points.shape
    indices = torch.empty(batch, count, dtype=torch.long, device=points.device)
    distance = torch.full((batch, size), float('inf'), device=points.device)
    farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for i in range(count):
        indices[:, i] = farthest
        center = points[rows, farthest, None]
        distance = torch.minimum(distance, (points - center).square().sum(-1))
        farthest = distance.argmax(-1)
    return indices


@torch.no_grad()
def nearest_indices(query, target, chunk_size=256):
    """Chunked matching avoids retaining a quadratic distance graph."""
    return torch.cat([
        torch.cdist(chunk.float(), target.float()).argmin(-1)
        for chunk in query.split(chunk_size, dim=1)
    ], dim=1)


def nearest_distances(first, second):
    forward_idx = nearest_indices(first, second)
    backward_idx = nearest_indices(second, first)
    forward = second.gather(1, forward_idx[..., None].expand(-1, -1, 3))
    backward = first.gather(1, backward_idx[..., None].expand(-1, -1, 3))
    return ((first - forward).square().sum(-1), forward_idx,
            (second - backward).square().sum(-1), backward_idx)
