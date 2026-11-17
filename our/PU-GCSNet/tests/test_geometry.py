"""Run with python -m unittest discover -s tests -v."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from network.PUCRN import PUGCSNet
from network.geometry_utils import local_pca_frame
from network.geometry_losses import surface_loss, normal_loss, gt_geometry
from network.model_loss import ChamferLoss
from network import point_ops
from main import predict_shape


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_shapes_frames_and_all_head_gradients(self):
        net = PUGCSNet().to(self.device).train()
        inputs = torch.randn(2, 3, 256, device=self.device, requires_grad=True)
        outputs, _, aux = net(inputs, return_aux=True)
        self.assertEqual([list(x.shape) for x in outputs], [[2, 512, 3], [2, 1024, 3], [2, 1024, 3]])
        for value in outputs + list(aux.values()):
            self.assertTrue(torch.isfinite(value).all())
        frame = torch.stack((aux['t1'], aux['t2'], aux['n_pred']), dim=-1)
        eye = torch.eye(3, device=self.device).expand(2, 512, 3, 3)
        torch.testing.assert_close(frame.transpose(-1, -2) @ frame, eye, atol=1e-5, rtol=1e-5)
        ratios = aux['sigma'] / aux['local_scale']
        self.assertTrue(((ratios >= 0.15) & (ratios <= 0.75)).all())
        target = torch.randn(2, 1024, 3, device=self.device)
        ChamferLoss()(outputs[1], target, torch.ones(2, device=self.device)).backward()
        gradient = net.upsampling_stage_2.param_head[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.flatten(1).abs().sum(1) > 0).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0)

    def test_degenerate_neighborhoods(self):
        clouds = torch.zeros(3, 32, 3, device=self.device)
        clouds[1, :, 0] = torch.linspace(-1, 1, 32, device=self.device)
        clouds[2, :16] = 1e-9
        frame = local_pca_frame(clouds)
        for tensor in frame.values():
            self.assertTrue(torch.isfinite(tensor).all())
        net = PUGCSNet().to(self.device).eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(net(clouds.transpose(1, 2))).all())

    def test_deterministic_and_permutation(self):
        net = PUGCSNet().to(self.device).eval()
        inputs = torch.randn(1, 3, 64, device=self.device)
        order = torch.randperm(64, device=self.device)
        inverse = order.argsort()
        with torch.no_grad():
            first = net(inputs)
            second = net(inputs)
            permuted = net(inputs[:, :, order]).reshape(1, 64, 4, 3)[:, inverse].reshape(1, 256, 3)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first, permuted, atol=2e-4, rtol=2e-4)

    def test_baseline_matches_original(self):
        original_path = Path(__file__).resolve().parents[3] / 'baseline/3DVision-main/PointUpsampling/PUCRN/network/PUCRN.py'
        if not original_path.exists():
            self.skipTest('Original baseline checkout is needed for parity test')
        # Both implementations use the same grouping backend for this comparison.
        module = types.ModuleType('pointnet2_ops.pointnet2_utils')
        module.grouping_operation = point_ops.grouping_operation
        with patch.dict(sys.modules, {'pointnet2_ops': types.ModuleType('pointnet2_ops'),
                                      'pointnet2_ops.pointnet2_utils': module}):
            spec = importlib.util.spec_from_file_location('original_pucrn', original_path)
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
        reference = original.CRNet(4).to(self.device).eval()
        net = PUGCSNet(stage2='baseline').to(self.device).eval()
        net.load_state_dict(reference.state_dict())
        inputs = torch.randn(2, 3, 32, device=self.device)
        with torch.no_grad():
            torch.testing.assert_close(net(inputs), reference(inputs), atol=0, rtol=0)

    def test_losses_and_normal_sign(self):
        gt = torch.randn(1, 64, 3, device=self.device)
        gt[..., 2] = 0
        normals, weights = gt_geometry(gt)
        points = (gt + torch.tensor([0, 0, 0.2], device=self.device)).requires_grad_()
        loss = surface_loss(points, gt, normals, weights)
        torch.testing.assert_close(loss, torch.tensor(0.04, device=self.device))
        loss.backward()
        self.assertGreater(points.grad[..., 2].abs().sum().item(), 0)
        positive = normal_loss(gt, normals, gt, normals, weights)
        negative = normal_loss(gt, -normals, gt, normals, weights)
        torch.testing.assert_close(positive, negative)
        self.assertLess(positive.item(), 1e-6)

    def test_ablation_modes_and_batch_interfaces(self):
        for mode in ('baseline', 'gaussian3d', 'tangent', 'quadratic'):
            for sampling in ('deterministic', 'random', 'antithetic'):
                net = PUGCSNet(stage2=mode, sampling=sampling).to(self.device)
                inputs = torch.randn(1, 3, 16, device=self.device)
                outputs, gt = net(inputs)
                self.assertEqual(outputs[-1].shape, (1, 64, 3))
                self.assertTrue(torch.isfinite(outputs[-1]).all())
                net.eval()
                with torch.no_grad():
                    self.assertEqual(net(inputs).shape, (1, 64, 3))

    def test_full_shape_patch_fps(self):
        net = PUGCSNet().to(self.device).eval()
        inputs = torch.randn(1, 3, 2048, device=self.device)
        args = types.SimpleNamespace(num_point=256, patch_num_ratio=3, save_stages=True)
        output = predict_shape(net, inputs, args)
        self.assertEqual(output['p1'].shape, (1, 3, 4096))
        for name in ('p2', 'p3'):
            self.assertEqual(output[name].shape, (1, 3, 8192))
            self.assertTrue(torch.isfinite(output[name]).all())


if __name__ == '__main__':
    unittest.main()
