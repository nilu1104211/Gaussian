"""Adapters reuse native models, losses, optimizers and inference functions.

One model per worker process avoids the upstream projects' colliding module names.
"""
import argparse
import importlib
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from retrain_common import ROOT, stable_seed


class NativeAdapter:
    def __init__(self, config):
        self.config = config
        self.kind, self.dataset = config['model'], config['dataset']
        self.last_prediction = None
        self.query_grad_norm = 0.0
        # Upstream configs.py parses argv at import time.
        sys.argv = [sys.argv[0]]
        if self.kind in ('E0', 'E6'):
            project = ROOT / 'our/PU-GCSNet'
            sys.path.insert(0, str(project))
            from main import parse_args, make_network
            from model import Model
            self.args = parse_args()
            values = dict(batch_size=config['batch_size'], random_seed=config['seed'], device='cuda',
                          lr_init=0.001, decay_iter=config['decay_iter'], save_stages=False)
            if self.kind == 'E0':
                values.update(stage2='baseline', lambda_surf=0., lambda_tail=0., lambda_growth=0.)
            else:
                values.update(stage2='quadratic', sampling='deterministic', refinement='local',
                    refine_radius=0.5, lambda_surf=0.1, surf_eta=0., lambda_normal=0.,
                    lambda_tail=0.05, lambda_growth=0.05, tail_fraction=0.02, tail_warmup_epochs=10)
            vars(self.args).update(values)
            self.net, _ = make_network(self.args, torch.device('cuda'))
            self.native = Model(self.net, 'train', self.args)
            self.optimizer, self.scheduler = self.native.optimizer, self.native.lr_scheduler
            self.aug = importlib.import_module('utils.pc_utils')
            self.test_module = importlib.import_module('main')
        else:
            project = ROOT / 'compare' / f'{self.kind}-main'
            sys.path.insert(0, str(project))
            if self.kind == 'PU-Gaussian':
                from configs import args as config_class
                self.args = config_class.args
                vars(self.args).update(dataset=self.dataset, batch_size=config['batch_size'],
                    random_seed=config['seed'], training_stage=1, num_samples=6, r=4,
                    up_rate=4, num_points=256, patch_rate=3, decay_iter=config['decay_iter'])
                from model.models import PU_Gaussian
                from train import PointCloudModel
                self.net = PU_Gaussian(self.args).cuda()
                self.native = PointCloudModel(self.net, 'train', self.args)
                # Suppress the upstream per-layer debug printing, retaining the same norm.
                self.native.get_gradient_norm = lambda: float(torch.stack([
                    p.grad.detach().norm() for p in self.net.parameters() if p.grad is not None]).norm())
                self.optimizer, self.scheduler = self.native.optimizer, self.native.lr_scheduler
                self.aug = importlib.import_module('data')
            else:
                package = 'args' if self.kind == 'Grad-PU' else 'cfgs.upsampling'
                parser_module = importlib.import_module(f'{package}.{self.dataset}_args')
                self.args = getattr(parser_module, f'parse_{self.dataset}_args')()
                vars(self.args).update(seed=config.get('model_seed', config['seed']), dataset=self.dataset, batch_size=config['batch_size'])
                if self.kind == 'Grad-PU':
                    self.args.lr = config.get('grad_lr', self.args.lr)
                self.aug = importlib.import_module('dataset.utils')
                if self.kind == 'Grad-PU':
                    from models.P2PNet import P2PNet
                    self.net = P2PNet(self.args).cuda()
                    # Optional, explicitly recorded initialization repair; ReLU/loss unchanged.
                    if config.get('grad_head_bias') is not None:
                        torch.nn.init.constant_(self.net.p2p_regressor.mlp_3.bias, config['grad_head_bias'])
                    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr,
                                                       weight_decay=self.args.weight_decay)
                    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                        step_size=self.args.lr_decay_step, gamma=self.args.gamma)
                    self.net.p2p_regressor.mlp_3.register_forward_hook(self._head_probe)
                else:
                    from models.repkpu import RepKPU
                    self.net = RepKPU(self.args).cuda()
                    deform = [p for n, p in self.net.named_parameters() if 'deform' in n]
                    other = [p for n, p in self.net.named_parameters() if 'deform' not in n]
                    self.optimizer = torch.optim.Adam([{'params': other}, {'params': deform, 'lr': self.args.lr * .1}],
                        lr=self.args.lr, weight_decay=self.args.weight_decay)
                    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.05 ** (1 / 150))
            self.test_module = importlib.import_module('test')
            self.utils = importlib.import_module('models.utils' if self.kind != 'PU-Gaussian' else 'utils.model_utils')
        self.project = project
        if 'prepared' in config:
            self.args.h5_file_path = str(Path(config['prepared']) / ('PU1K' if self.dataset == 'pu1k' else 'PU-GAN') / 'train.h5')
        self.args.input_policy = config.get('input_policy', 'native')
        self.health = dict(updates=0, zero_gradient_streak=0)
        self.optimizer.register_step_pre_hook(self._before_step)
        self.optimizer.register_step_post_hook(self._after_step)

    def _head_probe(self, module, inputs, raw):
        raw = raw.detach()
        self.head_health = dict(raw_min=float(raw.min()), raw_max=float(raw.max()),
                                positive_fraction=float((raw > 0).float().mean()))

    def _before_step(self, optimizer, args, kwargs):
        pairs = [(p, p.grad.detach().float().norm()) for p in self.net.parameters() if p.grad is not None]
        if not pairs:
            raise RuntimeError('No parameter gradients; validation may have left the model frozen')
        norms = torch.stack([n for _, n in pairs])
        if not torch.isfinite(norms).all():
            raise FloatingPointError('Nonfinite parameter gradient before optimizer.step')
        index = int(norms.argmax())
        self.health['gradient_norm'] = float(norms.norm())
        self.health['zero_gradient_streak'] = self.health['zero_gradient_streak'] + 1 if float(norms[index]) == 0 else 0
        if self.health['zero_gradient_streak'] >= 3:
            raise RuntimeError(f'Three consecutive zero-gradient steps; {getattr(self, "head_health", {})}')
        self.probed_parameter = pairs[index][0]
        self.probed_before = self.probed_parameter.detach().clone()

    def _after_step(self, optimizer, args, kwargs):
        self.health['parameter_step_max'] = float((self.probed_parameter.detach() - self.probed_before).abs().max())
        self.health['updates'] += 1
        del self.probed_before

    def train_batch(self, batch, step, epoch):
        self.net.train()
        if self.kind in ('E0', 'E6'):
            self.native.set_input(batch['input'].cuda().transpose(1, 2), batch['radius'].cuda(),
                                  batch['gt'].cuda().transpose(1, 2))
            self.native.optimize(step, epoch)
            result = dict(self.native.last_metrics)
        elif self.kind == 'PU-Gaussian':
            stage = 1 if epoch <= self.config['stage1_epochs'] else 2
            self.net.training_stage = self.native.training_stage = self.args.training_stage = stage
            self.native.set_input(batch['input'].cuda(), 0, x6=batch['x6'].cuda(), x20=batch['x20'].cuda())
            result = {k: v for k, v in self.native.optimize(epoch).items() if isinstance(v, (float, int))}
            result['stage'] = stage
        else:
            self.optimizer.zero_grad(set_to_none=True)
            points, gt = batch['input'].cuda().transpose(1, 2).contiguous(), batch['gt'].cuda().transpose(1, 2).contiguous()
            if self.kind == 'Grad-PU':
                interpolated = self.utils.midpoint_interpolate(self.args, points)
                queries = self.utils.get_query_points(interpolated, self.args)
                loss = self.utils.get_p2p_loss(self.args, self.net(interpolated, queries), queries, gt)
            else:
                prediction, regularizer = self.net(points)
                loss = self.utils.get_cd_loss(self.args, prediction, gt)
                if regularizer is not None:
                    loss = loss + regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native training loss')
            loss.backward()
            self.optimizer.step()
            result = dict(total_loss=float(loss))
        result.update(self.health)
        result['lr'] = self.optimizer.param_groups[0]['lr']
        if self.kind == 'Grad-PU':
            result.update(self.head_health)
        return result

    def epoch_end(self):
        if self.kind in ('Grad-PU', 'RepKPU'):
            self.scheduler.step()

    def infer(self, points):
        training = self.net.training
        flags = [p.requires_grad for p in self.net.parameters()]
        old_truncation = getattr(self.args, 'truncate_distance', None)
        self.net.eval()
        try:
            if self.kind in ('E0', 'E6'):
                center = points.mean(axis=0, keepdims=True)
                radius = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-12)
                tensor = torch.as_tensor((points - center) / radius, device='cuda').transpose(0, 1)[None]
                result = self.test_module.predict_shape(self.net, tensor, self.args)['p3']
                output = result.detach().transpose(1, 2)[0].cpu().numpy() * radius + center
            else:
                tensor = torch.as_tensor(points, dtype=torch.float32, device='cuda').transpose(0, 1)[None]
                tensor, center, radius = self.utils.normalize_point_cloud(tensor)
                if self.kind == 'Grad-PU':
                    original = self.net.regress_distance
                    self.query_grad_norm = 0.0
                    def probed(original_pts, query_pts, global_feats, local_feats):
                        if query_pts.requires_grad:
                            query_pts.register_hook(lambda g: setattr(self, 'query_grad_norm', max(self.query_grad_norm, float(g.detach().norm()))))
                        return original(original_pts, query_pts, global_feats, local_feats)
                    self.net.regress_distance = probed
                    self.args.truncate_distance = True  # native test entrypoint default
                    try:
                        with torch.enable_grad():
                            interpolated = self.utils.midpoint_interpolate(self.args, tensor)
                            result = self.test_module.pcd_update(self.args, self.net, interpolated)
                        if self.query_grad_norm == 0:
                            raise RuntimeError('Grad-PU inference has zero query gradient')
                    finally:
                        self.net.regress_distance = original
                else:
                    with torch.no_grad():
                        result = self.test_module.upsampling(self.args, self.net, tensor)
                output = (center + result * radius).detach().transpose(1, 2)[0].cpu().numpy()
            if output.shape != (8192, 3) or not np.isfinite(output).all():
                raise ValueError(f'Expected 8192 finite predictions, got {output.shape}')
            return output
        finally:
            if old_truncation is not None:
                self.args.truncate_distance = old_truncation
            for parameter, flag in zip(self.net.parameters(), flags):
                parameter.requires_grad_(flag)
            self.net.train(training)


class PatchDataset(torch.utils.data.Dataset):
    """Lazy H5 loading; native augmentations; model-independent sample randomness."""
    def __init__(self, path, adapter, seed, input_policy='fixed'):
        self.path, self.adapter, self.seed, self.input_policy = str(path), adapter, seed, input_policy
        with h5py.File(path, 'r') as file:
            self.count = len(file['poisson_256'])
        self.file, self.epoch = None, 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        adapter, aug = self.adapter, self.adapter.aug
        native_random = (adapter.kind in ('E0', 'E6') or getattr(adapter.args, 'use_random_input', False))
        random_input = self.input_policy == 'native' and native_random
        points = self.file['poisson_1024' if random_input else 'poisson_256'][index]
        gt = self.file['poisson_1024'][index]
        center = points.mean(0, keepdims=True)
        radius = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-12)
        points, gt = (points - center) / radius, (gt - center) / radius
        state = np.random.get_state()
        np.random.seed(stable_seed(self.seed, 'augmentation', self.epoch, index))
        try:
            if random_input:
                points = points[aug.nonuniform_sampling(len(points), 256)]
            if adapter.kind in ('E0', 'E6'):
                points, gt = aug.rotate_point_cloud_and_gt(points[None], gt[None])
                points, gt, scale = points[0], gt[0], 1.
            elif adapter.kind == 'PU-Gaussian':
                x6 = (self.file['poisson_1536'][index] - center) / radius
                x20 = (self.file['poisson_5120'][index] - center) / radius
                if adapter.dataset == 'pugan':
                    points = aug.jitter_perturbation_point_cloud(points, sigma=adapter.args.jitter_sigma, clip=adapter.args.jitter_max)
                points, x20, gt, x6 = aug.rotate_point_cloud_and_gt(points, x20, gt, x6)
                points, x20, gt, x6, scale = aug.random_scale_point_cloud_and_gt(points, x20, gt, x6, scale_low=.8, scale_high=1.2)
            else:
                # Decouple fixed base input from the PU-GAN input-jitter augmentation.
                if adapter.dataset == 'pugan':
                    points = aug.jitter_perturbation_point_cloud(points, sigma=adapter.args.jitter_sigma, clip=adapter.args.jitter_max)
                points, gt = aug.rotate_point_cloud_and_gt(points, gt)
                points, gt, scale = aug.random_scale_point_cloud_and_gt(points, gt, scale_low=.8, scale_high=1.2)
            result = dict(input=np.asarray(points, dtype=np.float32), gt=np.asarray(gt, dtype=np.float32),
                          radius=np.asarray(scale, dtype=np.float32))
            if adapter.kind == 'PU-Gaussian':
                result.update(x6=np.asarray(x6, dtype=np.float32), x20=np.asarray(x20, dtype=np.float32))
            return result
        finally:
            np.random.set_state(state)
