"""Local support bounds and correspondence-aware tail risk for refinement."""
import math

import torch


def bound_refinement_offset(offset, local_scale, radius_multiplier):
    """Smoothly bound each [B,N,3] displacement by multiplier * parent scale.

    The detached support radius cannot be enlarged through this gradient path.
    The mapping preserves direction and is approximately identity for small moves.
    """
    if not math.isfinite(radius_multiplier) or radius_multiplier <= 0:
        raise ValueError('Refinement radius multiplier must be finite and positive')
    radius = radius_multiplier * local_scale.detach().clamp_min(1e-8)
    shrink = torch.rsqrt(1 + (offset / radius).square().sum(-1, keepdim=True))
    return offset * shrink, shrink


def tail_mean(distances, fraction):
    """Mean of the largest ceil(fraction * N) errors, independently per patch."""
    if not 0 < fraction <= 1:
        raise ValueError('Tail fraction must be in (0, 1]')
    if distances.ndim != 2 or distances.shape[1] == 0:
        raise ValueError('Expected nonempty squared distances with shape [B,N]')
    count = max(1, math.ceil(distances.shape[1] * fraction))
    return distances.topk(count, dim=1, sorted=False).values.mean(1)


def refinement_tail_losses(p2_forward, p3_forward, p3_backward, fraction):
    """Final bidirectional tail risk and positive per-point error growth.

    P2/P3 must retain their pre-FPS point correspondence. Detaching the P2
    reference prevents reducing the growth penalty by directly worsening P2.
    These are soft training objectives, not guarantees on test-set Hausdorff.
    """
    if p2_forward.shape != p3_forward.shape:
        raise ValueError('P2/P3 distances must have the same point correspondence')
    forward = tail_mean(p3_forward, fraction).mean()
    backward = tail_mean(p3_backward, fraction).mean()
    growth = tail_mean((p3_forward - p2_forward.detach()).relu(), fraction).mean()
    return dict(tail_forward=forward, tail_backward=backward,
                tail=0.5 * (forward + backward), tail_growth=growth)
