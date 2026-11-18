import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import diagnose
from data import H5Dataset
from main import distance_metrics, evaluate, validate_patches
from network.PUCRN import PUGCSNet


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.root = Path(tempdir.name)
        self.rng = np.random.default_rng(42)
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def command(self, *args):
        with patch.object(sys, 'argv', ['diagnose.py', *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            diagnose.main()

    def checkpoint(self, mode, state=None):
        net = PUGCSNet(stage2=mode)
        if state is not None:
            net.load_state_dict(state)
        path = self.root / (mode + '.pth')
        torch.save(dict(net_state_dict=net.state_dict(), network_config=dict(stage2=mode), epoch=4), path)
        return net, path

    def test_direction_labels(self):
        values = distance_metrics(torch.tensor([[0., 9.], [1., 1.]]), torch.tensor([[4.], [16.]]))
        for key, expected in [('HD_pred_to_gt_sq', [9., 1.]), ('HD_gt_to_pred_sq', [4., 16.]),
                              ('HD_squared', [9., 16.]), ('HD_legacy_sum', [13., 17.]), ('CD', [8.5, 17.])]:
            torch.testing.assert_close(values[key], torch.tensor(expected))

    def test_saved_predictions_then_fps_attribution(self):
        for directory in ('ref', 'candidate', 'gt', 'inputs', 'meshes'):
            (self.root / directory).mkdir()
        for name in ('one', 'two'):
            cloud = self.rng.normal(size=(128, 3))
            np.savetxt(self.root / 'gt' / (name + '.xyz'), cloud)
            np.savetxt(self.root / 'ref' / (name + '.xyz'), cloud)
            np.savetxt(self.root / 'inputs' / (name + '.xyz'), cloud[:32])
            if name == 'one':
                cloud[0] += 5
            np.savetxt(self.root / 'candidate' / (name + '.xyz'), cloud)
        (self.root / 'meshes/one.off').touch()
        output = self.root / 'pair'
        self.command('paired', '--reference', self.root / 'ref', '--candidate', self.root / 'candidate',
                     '--gt', self.root / 'gt', '--pugan-meshes', self.root / 'meshes', '--out', output)
        groups = json.loads((output / 'summary.json').read_text())['groups']
        official = evaluate(self.root / 'candidate', self.root / 'gt')
        for key, value in official.items():
            self.assertEqual(groups['PU1K'][key]['candidate_mean'], value)
        self.assertEqual(groups['PU-GAN']['HD_squared']['worsened'], 1)
        self.assertEqual(groups['ShapeNet']['HD_squared']['worsened'], 0)
        _, ckpt = self.checkpoint('baseline')
        self.command('fps', '--ckpt', ckpt, '--inputs', self.root / 'inputs', '--gt', self.root / 'gt',
                     '--paired-csv', output / 'paired.csv', '--device', 'cpu', '--num-point', 16,
                     '--top', 1, '--out', self.root / 'stages')
        with (self.root / 'stages/fps.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row['name'] for row in rows}, {'one'})
        for before, after in ((rows[0], rows[1]), (rows[2], rows[3])):
            self.assertEqual((before['fps'], after['fps']), ('before', 'after'))
            self.assertEqual((int(before['points']), int(after['points'])), (384, 128))
            self.assertGreaterEqual(float(before['HD_pred_to_gt_sq']), float(after['HD_pred_to_gt_sq']))
            self.assertLessEqual(float(before['HD_gt_to_pred_sq']), float(after['HD_gt_to_pred_sq']))

    def test_fixed_patch_and_height_probe(self):
        path = self.root / 'val.h5'
        dense = self.rng.normal(size=(3, 64, 3)).astype(np.float32)
        with h5py.File(path, 'w') as h5:
            h5['poisson_16'] = dense[:, :16]
            h5['poisson_64'] = dense
        net = PUGCSNet(stage2='quadratic')
        with torch.no_grad():
            net.upsampling_stage_2.param_head[-1].bias[9:].fill_(1.)
        _, e3 = self.checkpoint('quadratic', net.state_dict())
        _, tangent = self.checkpoint('tangent', net.state_dict())
        dataset = H5Dataset(str(path), 16, 16, phase='val', use_random_input=False)
        validation = validate_patches(net, DataLoader(dataset, batch_size=2), 'cpu')
        summaries = []
        for i, (ckpt, factor) in enumerate([(e3, 1), (e3, 0.5), (e3, 0), (tangent, 1)]):
            out = self.root / str(i)
            self.command('patches', '--ckpt', ckpt, '--height-scale', factor, '--val-h5', path,
                         '--device', 'cpu', '--num-point', 16, '--batch-size', 2, '--out', out)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['patches'], 3)
            summaries.append(summary['mean_of_patch_metrics'])
        self.assertAlmostEqual(summaries[0]['p2_CD'], validation['cd2'], places=6)
        self.assertAlmostEqual(summaries[0]['p3_CD'], validation['cd3'], places=6)
        self.assertGreater(summaries[0]['height_over_d_max'], 0)
        self.assertAlmostEqual(summaries[1]['height_over_d_max'], summaries[0]['height_over_d_max'] / 2)
        self.assertEqual(summaries[2], summaries[3])
