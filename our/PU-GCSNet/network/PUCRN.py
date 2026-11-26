
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from .point_ops import grouping_operation
from .gaussian_surface_stage import GaussianSurfaceStage
from .hd_guard import bound_refinement_offset
from .learnable_surface_generator import LearnableSurfaceGenerator
from .cross_stage_refiner import CrossStageRefiner, gather_parent



class CRNet(torch.nn.Module):
    """
    Point Cloud Upsampling via Cascaded Refinement Network 

    Input:
        points: Input points, (B, 3, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, up_ratio=4, stage2='quadratic', refinement='legacy',
                 refine_radius=0.5, feature_bridge=False, learned_tangent=False,
                 learned_normal=False, geometry_gate=False, use_f2_skip=False,
                 use_geo_skip=False, use_neighborhood_refine=False,
                 use_tangent_normal_head=False, tangent_bound=0.25,
                 learned_normal_bound=0.1, refine_tangent_bound=0.1,
                 refine_normal_bound=0.05, **geometry_options):
        super(CRNet, self).__init__()
        if up_ratio != 4:
            raise ValueError('PU-GCSNet currently supports 4x upsampling')
        self.stage2 = stage2
        if refinement not in ('legacy', 'local'):
            raise ValueError('Unknown refinement mode')
        if refinement == 'local' and stage2 == 'baseline':
            raise ValueError('Local refinement requires Stage-2 parent geometry')
        if not np.isfinite(refine_radius) or refine_radius <= 0:
            raise ValueError('refine_radius must be finite and positive')
        self.refinement, self.refine_radius = refinement, refine_radius
        self.use_feature_bridge = feature_bridge
        cross_stage = use_f2_skip or use_geo_skip or use_neighborhood_refine or use_tangent_normal_head
        new_surface = feature_bridge or learned_tangent or learned_normal or geometry_gate or cross_stage
        if new_surface and stage2 != 'quadratic':
            raise ValueError('New geometry modules require quadratic Stage-2')
        step_up_rate = int(np.sqrt(up_ratio))
        self.upsampling_stage_1 = SubNetwork(up_ratio = step_up_rate)
        surface_class = LearnableSurfaceGenerator if new_surface else GaussianSurfaceStage
        new_options = dict(feature_bridge=feature_bridge, learned_tangent=learned_tangent,
                           learned_normal=learned_normal, geometry_gate=geometry_gate,
                           child_features=cross_stage, tangent_bound=tangent_bound,
                           learned_normal_bound=learned_normal_bound) if new_surface else {}
        self.upsampling_stage_2 = (SubNetwork(up_ratio=step_up_rate) if stage2 == 'baseline'
                                   else surface_class(Transformer_extractor(128, 64),
                                                      mode=stage2, **new_options, **geometry_options))
        self.refinement_stage = SubNetwork(up_ratio = 1)
        if cross_stage:
            self.cross_stage_refiner = CrossStageRefiner(use_f2_skip, use_geo_skip,
                use_neighborhood_refine, use_tangent_normal_head,
                refine_tangent_bound, refine_normal_bound)

    def forward(self, point_cloud, gt=None, return_aux=False):
        point_cloud = point_cloud.float().contiguous() 
        if self.use_feature_bridge:
            stage1 = self.upsampling_stage_1(point_cloud, return_aux=True)
            p1_pred, f1 = stage1['points'], stage1['up_feat']
        else:
            p1_pred, f1 = self.upsampling_stage_1(point_cloud), None
        aux = {}
        if (return_aux or self.refinement == 'local' or hasattr(self, 'cross_stage_refiner') or self.use_feature_bridge) and self.stage2 != 'baseline':
            p2_pred, aux = self.upsampling_stage_2.forward_with_aux(p1_pred, parent_features=f1)
        else:
            p2_pred = self.upsampling_stage_2(p1_pred)
        if hasattr(self, 'cross_stage_refiner'):
            # up_ratio=1 preserves P2 ordering; fuse before the original regressor.
            stage3 = self.refinement_stage(p2_pred, return_aux=True)
            f3 = self.cross_stage_refiner.fuse(stage3['up_feat'], p2_pred, aux)
            p3_pred = p2_pred + torch.tanh(self.refinement_stage.regressor(f3))
        else:
            p3_pred = self.refinement_stage(p2_pred)
        if self.refinement == 'local':
            # Stage-2 flattens (parent, child), so each scale belongs to two
            # consecutive children. This correspondence is preserved by Stage-3.
            scale = gather_parent(aux['local_scale'], aux['parent_idx'])
            offset = (p3_pred - p2_pred).transpose(1, 2)
            bounded, shrink = bound_refinement_offset(offset, scale, self.refine_radius)
            p3_pred = p2_pred + bounded.transpose(1, 2)
            if return_aux:
                aux.update(refine_raw_move_over_d=offset.norm(dim=-1) / scale.squeeze(-1),
                           refine_move_over_d=bounded.norm(dim=-1) / scale.squeeze(-1),
                           refine_shrink=shrink.squeeze(-1))

        if hasattr(self, 'cross_stage_refiner'):
            extra_move = self.cross_stage_refiner.residual(f3, aux)
            p3_pred = p3_pred + extra_move
            if return_aux:
                aux['refine_geo_move_over_d'] = extra_move.transpose(1, 2).norm(dim=-1) / gather_parent(aux['local_scale'], aux['parent_idx']).squeeze(-1)

        p3_pred = p3_pred.permute(0, 2, 1).contiguous()
        p2_pred = p2_pred.permute(0, 2, 1).contiguous()
        p1_pred = p1_pred.permute(0, 2, 1).contiguous()

        if self.training:
            outputs = [p1_pred, p2_pred, p3_pred]
            return (outputs, gt, aux) if return_aux else (outputs, gt)
        else:
            return (p3_pred, dict(aux, p1=p1_pred, p2=p2_pred)) if return_aux else p3_pred




class SubNetwork(nn.Module):
    """
    Upsampling or Refinement Subnetwork

    Input:
        points: Input points, (B, 3, N_input)
    Output:
        up_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, up_ratio=2):
        super(SubNetwork, self).__init__()
        self.feature_extractor = Transformer_extractor(128, 64)
        self.up_unit = Upsampling_unit(up_ratio=up_ratio)  
        self.regressor = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        self.up_ratio = up_ratio

    def forward(self, points, return_aux=False):
        point_feat = self.feature_extractor(points)
        up_feat, duplicated_point = self.up_unit(point_feat, points)
        offest = self.regressor(up_feat)
        up_point = duplicated_point + torch.tanh(offest)

        if return_aux:
            # Both ConvTranspose1d(kernel=stride=ratio) and nearest duplication
            # put consecutive children next to their corresponding parent.
            count = points.shape[-1]
            parent = torch.arange(count, device=points.device).repeat_interleave(self.up_ratio)
            child = torch.arange(self.up_ratio, device=points.device).repeat(count)
            return dict(points=up_point, point_feat=point_feat, up_feat=up_feat,
                        parent_idx=parent[None].expand(points.shape[0], -1),
                        child_idx=child[None].expand(points.shape[0], -1))
        return up_point


class Transformer_extractor(nn.Module):
    """
    Point-wise feature extractor.

    Input:
        points: input points, (B, 3, N_input)
    Output:
        point_feat: ouput feature, (B, dim_feat, N_input)
    """
    def __init__(self, dim_feat, hidden_dim):
        super(Transformer_extractor, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat*2, dim_feat])
        self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

    def forward(self, points):
        feature_1 = self.mlp_1(points)
        global_feature = torch.max(feature_1, 2, keepdim=True)[0]
        feature_2 = torch.cat([feature_1, global_feature.repeat((1, 1, feature_1.size(2)))], 1)
        feature_3 = self.mlp_2(feature_2)
        point_feat = self.point_transformer(feature_3, points)
        return point_feat



class Upsampling_unit(nn.Module):
    """
    Point upsampling unit
    
    Input:
        point_feat: input feature, (B, dim_feat, N_input)
        points: input points, (B, 3, N_input)
    Output:
        up_feat: upsampled feature, (B, dim, up_ratio * N_input)
        duplicated_point: upsampled results, (B, 3, up_ratio * N_input)
    """
    def __init__(self, up_ratio=2):
        super(Upsampling_unit, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.mlp_2 = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.deconv_branch = nn.ConvTranspose1d(32, 128, up_ratio, up_ratio, bias=False) 
        self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)

    def forward(self, point_feat, points):
        deconved_feat = self.deconv_branch(self.mlp_1(point_feat)) 
        duplicated_feat = self.duplicated_branch(point_feat)
        up_feat = self.mlp_2(torch.cat([deconved_feat, duplicated_feat], 1))
        up_feat = torch.relu(up_feat)
        duplicated_point = self.duplicated_branch(points)
        return up_feat, duplicated_point




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()


class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

class Transformer(nn.Module):
    """
    [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)

    feed forward of transformer
    Args:
        x: Tensor of features, (B, in_channel, n)
        pos: Tensor of positions, (B, 3, n)

    Returns:
        y: Tensor of features with attention, (B, in_channel, n)
        
    """

    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity


# Descriptive public name, with CRNet retained for the original entry points.
PUGCSNet = CRNet
