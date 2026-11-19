import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import diagnose
from main import make_network, validate_patches
from model import Model
from network.PUCRN import PUGCSNet
from network.hd_guard import bound_refinement_offset, refinement_tail_losses, tail_mean
from network.model_loss import ChamferLoss


class HDGuardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_displacement_bound_rotation_scale_and_zero_gradient(self):
        offset = torch.tensor([[[10., 2., 3.], [0., 0., 0.], [1e-6, 0., 0.]]], requires_grad=True)
        scale = torch.full((1, 3, 1), 0.1, requires_grad=True)
        bounded, _ = bound_refinement_offset(offset, scale, 0.5)
        self.assertTrue((bounded.norm(dim=-1) <= 0.050001).all())
        torch.testing.assert_close(bounded[:, 2], offset[:, 2], atol=1e-10, rtol=1e-5)
        rotation = torch.linalg.qr(torch.randn(3, 3)).Q
        rotated, _ = bound_refinement_offset(offset @ rotation, scale, 0.5)
        torch.testing.assert_close(rotated, bounded @ rotation)
        scaled, _ = bound_refinement_offset(offset * 7, scale * 7, 0.5)
        torch.testing.assert_close(scaled, bounded * 7)
        bounded.sum().backward()
        self.assertIsNone(scale.grad)
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertGreater(offset.grad[:, 1].abs().sum().item(), 0)

    def test_per_patch_tail_and_correspondence_growth(self):
        errors = torch.tensor([[0., 0., 0., 100.], [1., 2., 3., 4.]])
        torch.testing.assert_close(tail_mean(errors, 0.5), torch.tensor([50., 3.5]))
        torch.testing.assert_close(tail_mean(errors, 0.01), torch.tensor([100., 4.]))
        # The worst error decreases, but a different point deteriorates.
        p2 = torch.tensor([[10., 0.]], requires_grad=True)
        p3 = torch.tensor([[0., 8.]], requires_grad=True)
        backward = torch.tensor([[2., 6.]], requires_grad=True)
        values = refinement_tail_losses(p2, p3, backward, 0.5)
        self.assertEqual(values['tail'].item(), 7.)
        self.assertEqual(values['tail_growth'].item(), 8.)
        values['tail_growth'].backward(retain_graph=True)
        self.assertIsNone(p2.grad)
        torch.testing.assert_close(p3.grad, torch.tensor([[0., 1.]]))
        values['tail'].backward()
        self.assertGreater(backward.grad.abs().sum().item(), 0)
        self.assertEqual(refinement_tail_losses(p3, p3, backward, 0.5)['tail_growth'].item(), 0.)

    def test_reused_chamfer_matches_loss_and_gradient(self):
        first = torch.randn(2, 16, 3, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        target = torch.randn(2, 24, 3)
        criteria = ChamferLoss()
        original = criteria(first, target, torch.ones(2))
        reused, forward, backward = criteria(second, target, torch.ones(2), return_distances=True)
        torch.testing.assert_close(original, reused, atol=0, rtol=0)
        torch.testing.assert_close(reused, 0.5 * (forward.mean() + backward.mean()))
        original.backward()
        reused.backward()
        torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)

    def test_checkpoint_compatibility_and_parent_child_bound(self):
        legacy = PUGCSNet().eval()
        old = dict(network_config=dict(stage2='quadratic'), net_state_dict=legacy.state_dict())
        restored, config = make_network(SimpleNamespace(up_ratio=4), 'cpu', old)
        self.assertEqual(config['refinement'], 'legacy')
        local = PUGCSNet(refinement='local', refine_radius=0.5).eval()
        local.load_state_dict(legacy.state_dict())
        inputs = torch.randn(1, 3, 32)
        with torch.no_grad():
            restored.eval()
            torch.testing.assert_close(restored(inputs), legacy(inputs), atol=0, rtol=0)
            output, aux = local(inputs, return_aux=True)
            torch.testing.assert_close(output, local(inputs), atol=0, rtol=0)
            radii = aux['local_scale'].repeat_interleave(2, dim=1).squeeze(-1) * 0.5
            self.assertTrue(((output - aux['p2']).norm(dim=-1) <= radii + 1e-6).all())
            saved = dict(network_config=dict(stage2='quadratic', refinement='local', refine_radius=0.5),
                         net_state_dict=local.state_dict())
            loaded, _ = make_network(SimpleNamespace(up_ratio=4), 'cpu', saved)
            loaded.eval()
            torch.testing.assert_close(loaded(inputs), output, atol=0, rtol=0)

    def test_optimizer_step_and_warmup(self):
        net = PUGCSNet(refinement='local')
        options = SimpleNamespace(lr_init=0.001, decay_iter=50000, lambda_tail=0.05, lambda_growth=0.05,
                                  tail_fraction=0.02, tail_warmup_epochs=10, lambda_surf=0.1)
        model = Model(net, 'train', options)
        tracked = list(net.refinement_stage.parameters())[-1]
        before = tracked.detach().clone()
        model.set_input(torch.randn(2, 3, 16), torch.ones(2), torch.randn(2, 3, 64))
        model.optimize(0, 1)
        m = model.last_metrics
        self.assertAlmostEqual(m['tail_ramp'], 0.1, places=6)
        self.assertAlmostEqual(m['weighted_tail'], m['tail'] * 0.005, places=6)
        self.assertAlmostEqual(m['total'], sum(m[k] for k in
                               ('cd1', 'cd2', 'cd3', 'weighted_surf', 'weighted_normal',
                                'weighted_tail', 'weighted_growth')), places=5)
        self.assertTrue(all(np.isfinite(v) for v in m.values()))
        self.assertLessEqual(m['refine_move_over_d_max'], 0.500001)
        self.assertFalse(torch.equal(before, tracked.detach()))

    def test_validation_records_directional_hd(self):
        class Fixed(torch.nn.Module):
            def forward(self, inputs, return_aux=False):
                points = inputs.transpose(1, 2)
                return points, {'p2': points}
        inputs = torch.tensor([[[0., 4.], [0., 0.], [0., 0.]]])
        target = torch.tensor([[[0., 0.], [0., 0.], [0., 0.]]])
        metrics = validate_patches(Fixed(), [(inputs, target, torch.ones(1))], 'cpu')
        self.assertEqual(metrics['hd3'], 16.)
        self.assertEqual(metrics['hd_forward3'], 16.)
        self.assertEqual(metrics['hd_backward3'], 0.)
        self.assertEqual(metrics['cd3'], 8.)

    def test_shared_evaluation_preserves_scale_and_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('gt', 'ref', 'candidate', 'inputs', 'meshes'):
                (root / folder).mkdir()
            cloud = np.random.default_rng(42).normal(size=(32, 3))
            for folder in ('gt', 'ref', 'inputs'):
                np.savetxt(root / folder / 'one.xyz', cloud)
            np.savetxt(root / 'candidate/one.xyz', 2 * cloud + 5)
            (root / 'meshes/one.off').touch()
            for protocol in ('official', 'shared'):
                args = ['diagnose.py', 'paired', '--reference', str(root / 'ref'),
                        '--candidate', str(root / 'candidate'), '--gt', str(root / 'gt'),
                        '--pugan-meshes', str(root / 'meshes'), '--out', str(root / protocol)]
                if protocol == 'shared':
                    args += ['--shared-inputs', str(root / 'inputs')]
                with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                    diagnose.main()
            official = json.loads((root / 'official/summary.json').read_text())['groups']['PU1K']
            shared = json.loads((root / 'shared/summary.json').read_text())['groups']['PU1K']
            self.assertLess(official['HD_squared']['candidate_mean'], 1e-10)
            self.assertGreater(shared['HD_squared']['candidate_mean'], 1.)


if __name__ == '__main__':
    unittest.main()
