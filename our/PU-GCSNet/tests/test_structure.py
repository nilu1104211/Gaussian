"""Scientific invariants of the new structure; CPU and synthetic data only."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from network.PUCRN import PUGCSNet, SubNetwork
from network.cross_stage_refiner import gather_parent
from network.geometry_utils import local_pca_frame
from network.geometry_feature_bridge import geometry_descriptor
from network.variants import NEW_VARIANTS, variant_options
from main import make_network, predict_shape
from model import Model


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_all_off_and_zero_initialized_backbone_parity(self):
        torch.manual_seed(21)
        reference = PUGCSNet(refinement='local').eval()
        x = torch.randn(2, 3, 32)
        with torch.no_grad():
            expected = reference(x)
        for name in ('E6', 'M0', 'V1a', 'V1', 'V2a', 'V2', 'V3'):
            torch.manual_seed(21)
            net = PUGCSNet(**variant_options(name)).eval()
            # Same seed initializes common backbone tensors identically.
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(net.state_dict()[key], value, atol=0, rtol=0)
            missing, unexpected = net.load_state_dict(reference.state_dict(), strict=False)
            self.assertFalse(unexpected)
            if name == 'E6':
                self.assertFalse(missing)
            with torch.no_grad():
                actual, aux = net(x, return_aux=True)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(net(x), actual, atol=0, rtol=0)
            self.assertEqual(aux['parent_idx'].shape, (2, 128))

    def test_parent_and_feature_alignment(self):
        subnet = SubNetwork(2).eval()
        points = torch.randn(2, 3, 32)
        aux = subnet(points, return_aux=True)
        duplicated = points.gather(2, aux['parent_idx'][:, None].expand(-1, 3, -1))
        expected = duplicated + torch.tanh(subnet.regressor(aux['up_feat']))
        torch.testing.assert_close(aux['points'], expected, atol=0, rtol=0)
        self.assertEqual(aux['child_idx'][0].tolist(), [0, 1] * 32)
        net = PUGCSNet(**variant_options('V3')).eval()
        _, aux = net(points, return_aux=True)
        self.assertEqual(aux['f2'].shape, (2, 128, 128))
        source = torch.arange(64).float()[None, :, None].expand(2, -1, -1).requires_grad_()
        children = gather_parent(source, aux['parent_idx'])
        torch.testing.assert_close(children[:, ::2], source)
        torch.testing.assert_close(children[:, 1::2], source)
        children.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 2))

    def test_loss3_reaches_new_branches_after_zero_init_opens(self):
        net = PUGCSNet(**variant_options('V3')).train()
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        inputs = torch.randn(2, 3, 32, requires_grad=True)
        tracked = ('upsampling_stage_2.feature_bridge.residual.0.weight',
                   'upsampling_stage_2.offset_head.0.weight',
                   'upsampling_stage_2.child_encoder.0.weight',
                   'cross_stage_refiner.fusion.0.weight',
                   'cross_stage_refiner.residual_head.0.weight')
        for _ in range(3):
            optimizer.zero_grad()
            outputs, _, aux = net(inputs, return_aux=True)
            outputs[-1].square().mean().backward()
            optimizer.step()
        for key in tracked:
            grad = dict(net.named_parameters())[key].grad
            self.assertIsNotNone(grad, key)
            self.assertTrue(torch.isfinite(grad).all(), key)
            self.assertGreater(grad.abs().sum().item(), 0, key)
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        self.assertFalse(aux['local_scale'].requires_grad)

    def test_inference_independent_of_gt_and_files(self):
        net = PUGCSNet(**variant_options('V3')).eval()
        # No data/file access occurs during forward; the unused gt argument
        # remains solely for compatibility with the native training wrapper.
        x = torch.randn(1, 3, 32)
        with patch('builtins.open', side_effect=AssertionError('Inference read a file')), torch.no_grad():
            plain = net(x)
            supplied = net(x, gt=torch.full((1, 3, 128), float('nan')))
        torch.testing.assert_close(plain, supplied, atol=0, rtol=0)

    def test_geometry_scalar_stability_and_degenerate_inputs(self):
        points = torch.randn(1, 64, 3)
        frame = local_pca_frame(points, return_neighbors=True)
        original, _, residual = geometry_descriptor(frame)
        for axis in ('normal', 'tangent', 'bitangent'):
            flipped = dict(frame)
            flipped[axis] = -frame[axis]
            descriptor, _, _ = geometry_descriptor(flipped)
            torch.testing.assert_close(descriptor, original, atol=2e-5, rtol=2e-5)
        rotation = torch.linalg.qr(torch.randn(3, 3)).Q
        moved = local_pca_frame((points @ rotation) * 3 + 2, return_neighbors=True)
        descriptor, _, _ = geometry_descriptor(moved)
        delta = descriptor - original
        # log(RMS scale) is deliberately scale-aware; all other scalars are
        # dimensionless and invariant away from repeated PCA eigenvalues.
        torch.testing.assert_close(delta[..., 3], torch.full_like(delta[..., 3], 3.).log(), atol=2e-5, rtol=2e-5)
        keep = [i for i in range(13) if i != 3]
        torch.testing.assert_close(descriptor[..., keep], original[..., keep], atol=2e-4, rtol=2e-4)
        net = PUGCSNet(**variant_options('V3')).eval()
        cloud = torch.zeros(3, 3, 32)
        cloud[1, 0] = torch.linspace(-1, 1, 32)
        cloud[2, :, :16] = 1e-9
        with torch.no_grad():
            output, aux = net(cloud, return_aux=True)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(aux['geometry_descriptor']).all())

    def test_optional_modules_bounds_and_checkpoint(self):
        options = variant_options('V3')
        options.update(learned_normal=True, geometry_gate=True, use_neighborhood_refine=True)
        net = PUGCSNet(**options).eval()
        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            net.upsampling_stage_2.offset_head[-1].bias.fill_(100)
            output, aux = net(x, return_aux=True)
            self.assertLessEqual(aux['learned_child_delta'].abs().max().item(), 1.)
            self.assertTrue(torch.isfinite(output).all())
            saved = dict(network_config=options, net_state_dict=net.state_dict())
            restored, _ = make_network(SimpleNamespace(up_ratio=4), 'cpu', saved)
            restored.eval()
            torch.testing.assert_close(restored(x), output, atol=0, rtol=0)
        net.train()
        net(x)[0][-1].square().mean().backward()
        gradient = net.cross_stage_refiner.residual_head[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_native_loss_and_gradient_logging(self):
        options = SimpleNamespace(lr_init=.001, decay_iter=200000, lambda_surf=.1,
            lambda_tail=.05, lambda_growth=.05, tail_fraction=.02, tail_warmup_epochs=10)
        net = PUGCSNet(**variant_options('V3'))
        wrapper = Model(net, 'train', options)
        wrapper.set_input(torch.randn(2, 3, 32), torch.ones(2), torch.randn(2, 3, 128))
        wrapper.optimize(0, 5)
        m = wrapper.last_metrics
        self.assertAlmostEqual(m['tail_ramp'], .5)
        self.assertAlmostEqual(m['total'], sum(m[key] for key in
            ('cd1', 'cd2', 'cd3', 'weighted_surf', 'weighted_normal', 'weighted_tail', 'weighted_growth')), places=5)
        for name in ('grad_feature_bridge', 'grad_surface_offset', 'grad_cross_fusion', 'grad_cross_head'):
            self.assertGreater(m[name], 0, name)

    def test_full_patch_shapes_and_presets(self):
        net = PUGCSNet(**variant_options('V3')).train()
        with torch.no_grad():
            outputs, _ = net(torch.randn(2, 3, 256))
        self.assertEqual([tuple(p.shape) for p in outputs], [(2, 512, 3), (2, 1024, 3), (2, 1024, 3)])
        for name in NEW_VARIANTS:
            PUGCSNet(**variant_options(name))
        self.assertEqual(variant_options('Tail-only')['refinement'], 'legacy')


if __name__ == '__main__':
    unittest.main()
