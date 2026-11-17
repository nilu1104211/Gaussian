from collections import OrderedDict
import os
import torch
from math import log
from collections import defaultdict


from network.model_loss import ChamferLoss
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.point_ops import furthest_point_sample, gather_operation
from network.geometry_losses import gt_geometry, surface_loss, normal_loss



class Model(object):
    def __init__(self, net, phase, opt):
        self.net = net
        self.phase = phase
        self.opt = opt
        self.lambda_surf = getattr(opt, 'lambda_surf', 0.0)
        self.lambda_normal = getattr(opt, 'lambda_normal', 0.0)
        self.surf_eta = getattr(opt, 'surf_eta', 0.0)
        self.last_metrics = {}

        if self.phase == 'train':
            self.error_log = defaultdict(int)
            self.chamfer_criteria = ChamferLoss()
            self.old_lr = opt.lr_init
            self.lr = opt.lr_init
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr_init,
                                                    betas=(0.9, 0.999))
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.7)
            self.decay_step = opt.decay_iter 

        self.step = 0

    def set_input(self, input_pc, radius, label_pc=None):
        """`
        :param
            input_pc       Bx3xN
            up_ratio       int
            label_pc       Bx3xN'
        """
        self.input = input_pc.detach()
        self.radius = radius
        # gt point cloud
        if label_pc is not None:
            self.gt = label_pc.detach()
        else:
            self.gt = None

    def forward(self):
        if self.gt is not None:
            self.predicted, self.gt, self.aux = self.net(self.input, gt=self.gt, return_aux=True)
        else:
            self.predicted = self.net(self.input)
        
    def get_lr(self, optimizer):
        """Get the current learning rate from optimizer.
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def optimize(self, steps=None, epoch=None):
        """
        run forward and backward, apply gradients
        """
        self.optimizer.zero_grad()
        self.net.train()
        self.forward()

        P1, P2, P3  = self.predicted
        gt_downsample_idx = furthest_point_sample(self.gt.permute(0, 2, 1).contiguous(), P1.shape[1])
        gt_downsample = gather_operation(self.gt, gt_downsample_idx)
        cd_1 =  self.compute_chamfer_loss(P1, gt_downsample) 
        cd_2 =  self.compute_chamfer_loss(P2, self.gt) 
        cd_3 =  self.compute_chamfer_loss(P3, self.gt) 
        loss = cd_1 + cd_2 + cd_3 
        losses = [cd_1, cd_2, cd_3]

        surf2, surf3, normals_loss = [loss.new_zeros(()) for _ in range(3)]
        if self.lambda_surf > 0 or self.lambda_normal > 0:
            target = self.gt.transpose(1, 2).contiguous()
            normals, weights = gt_geometry(target, getattr(self.opt, 'gt_k', 24),
                                           getattr(self.opt, 'confidence_tau', 0.0))
            if self.lambda_surf > 0:
                surf2 = surface_loss(P2, target, normals, weights)
                if self.surf_eta > 0:
                    surf3 = surface_loss(P3, target, normals, weights)
            if self.lambda_normal > 0:
                if 'n_pred' not in self.aux:
                    raise ValueError('Normal supervision requires a tangent or quadratic stage')
                normals_loss = normal_loss(self.aux['mu'], self.aux['n_pred'], target, normals, weights)
        weighted_surf = self.lambda_surf * (surf2 + self.surf_eta * surf3)
        weighted_normal = self.lambda_normal * normals_loss
        loss = loss + weighted_surf + weighted_normal

        with torch.no_grad():
            metrics = dict(cd1=cd_1, cd2=cd_2, cd3=cd_3, surf2=surf2, surf3=surf3,
                           normal=normals_loss, weighted_surf=weighted_surf,
                           weighted_normal=weighted_normal, total=loss,
                           refinement_move=(P3 - P2).norm(dim=-1).mean())
            if self.aux:
                d = self.aux['local_scale']
                metrics['center_move_over_d'] = ((self.aux['mu'] - P1).norm(dim=-1, keepdim=True) / d).mean()
                for j in range(self.aux['sigma'].shape[-1]):
                    metrics['sigma%d_over_d' % (j + 1)] = (self.aux['sigma'][..., j:j+1] / d).mean()
                if 'abc' in self.aux:
                    for j, name in enumerate(('a', 'b', 'c')):
                        metrics[name + '_abs'] = self.aux['abc'][..., j].abs().mean()
                    metrics['valid_frame_fraction'] = self.aux['frame_valid'].mean()
            self.last_metrics = {name: value.item() for name, value in metrics.items()}

        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss before optimizer step')

        loss.backward()
        self.optimizer.step()
        if steps % self.decay_step == 0 and steps!=0:
            self.lr_scheduler.step()

        lr = self.get_lr(self.optimizer)
        return losses, lr

    def compute_chamfer_loss(self, pc, pc_label):

        loss_chamfer = self.chamfer_criteria(
            pc.transpose(1, 2).contiguous(),
            pc_label.transpose(1, 2).contiguous(), self.radius)

        return loss_chamfer
