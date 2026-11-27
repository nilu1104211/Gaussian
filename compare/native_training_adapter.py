"""Method-native augmentation on lazily read publisher H5 patches."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch

from retrain_adapters import NativeAdapter
from retrain_common import ROOT, stable_seed


class PublisherAdapter(NativeAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.args.h5_file_path = config['data']['training_h5']
        self.args.num_workers = 0
        if self.kind in ('E0', 'E6'):
            self.args.num_shape_point = self.args.num_point = 256
            self.args.h5_data = config['data']['training_h5']
            self.args.max_epoch, self.args.random_seed = config['epochs'], config['model_seed']
            path = ROOT/'baseline/3DVision-main/PointUpsampling/PUCRN/data.py'
            spec = importlib.util.spec_from_file_location('publisher_pucrn_data', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.dataset_class = module.H5Dataset
        else:
            if self.kind == 'PU-Gaussian':
                from data import PUDataset as GaussianDataset
                self.dataset_class = GaussianDataset
                self.args.max_epochs, self.args.stage_1_max_epochs = config['epochs'], config['stage1_epochs']
                self.args.random_seed = config['model_seed']
                self.net.Gaussian.num_samples = self.args.num_samples = 20
                self.net.register_forward_hook(self._gaussian_shapes)
            else:
                # `data` is not a module in Grad-PU/RepKPU; their datasets live here.
                from dataset.dataset import PUDataset
                self.dataset_class = PUDataset
                self.args.epochs = config['epochs']

    def _gaussian_shapes(self, module, inputs, output):
        if not module.training:
            return
        predicted, _, target = output
        expected = inputs[0].shape[-1] * self.net.Gaussian.num_samples
        if predicted.shape[-1] != expected or target.shape[-1] != expected:
            raise RuntimeError(f'Gaussian stage {self.native.training_stage}: '
                               f'prediction={predicted.shape}, target={target.shape}, expected={expected}')
        self.gaussian_shapes = dict(gaussian_samples=self.net.Gaussian.num_samples,
                                    predicted_points=predicted.shape[-1], target_points=target.shape[-1])

    def train_batch(self, batch, step, epoch):
        if self.kind == 'PU-Gaussian':
            samples = 20 if epoch <= self.config['stage1_epochs'] else 6
            self.net.Gaussian.num_samples = self.args.num_samples = samples
        result = super().train_batch(batch, step, epoch)
        if self.kind == 'PU-Gaussian':
            result.update(self.gaussian_shapes)
        return result

    def infer(self, points):
        if self.kind != 'PU-Gaussian':
            return super().infer(points)
        samples, argument = self.net.Gaussian.num_samples, self.args.num_samples
        self.net.Gaussian.num_samples = self.args.num_samples = 6
        try:
            return super().infer(points)
        finally:
            self.net.Gaussian.num_samples, self.args.num_samples = samples, argument


class PublisherDataset(torch.utils.data.Dataset):
    """Invoke the original __getitem__; avoid loading multi-GB H5s into RAM.

    Normalization is performed per patch with the same input centroid/radius as
    the original batch normalization in load_h5_data. No point reorder or new FPS.
    """
    def __init__(self, path, adapter, seed, limit=None):
        self.path, self.adapter, self.seed = str(Path(path).resolve()), adapter, seed
        self.file, self.epoch = None, 1
        with h5py.File(self.path, 'r') as file:
            count = len(file['poisson_1024'])
        self.count = min(count, limit) if limit is not None else count

    def __len__(self):
        return self.count

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self.file is None:
            self.file = h5py.File(self.path, 'r', rdcc_nbytes=1024*1024)
        adapter = self.adapter
        backbone = adapter.kind in ('E0', 'E6')
        random_input = backbone or adapter.args.use_random_input
        key = 'poisson_1024' if random_input else 'poisson_256'
        points = self.file[key][index:index+1]
        center = points.mean(axis=1, keepdims=True)
        points = points - center
        radius = np.sqrt((points**2).sum(axis=-1)).max(axis=1, keepdims=True)
        if not np.isfinite(radius).all() or np.any(radius <= 1e-12):
            raise ValueError(f'Degenerate input patch {index}: {self.path}')
        points = points / radius[..., None]
        target = (self.file['poisson_1024'][index:index+1] - center) / radius[..., None]
        native = object.__new__(adapter.dataset_class)
        native.args = adapter.args
        if backbone:
            native.phase, native.batch_size, native.num_patch_point = 'train', 1, 256
            native.use_random_input, native.curr_scales = True, [4]
            native.jitter = adapter.args.jitter
            native.jitter_sigma, native.jitter_max = adapter.args.jitter_sigma, adapter.args.jitter_max
            native.input_array, native.label_array = points, {'x4': target}
            native.data_radius = np.ones(1)
        else:
            native.input_data, native.gt_data = points, target
            native.radius_data = np.ones((1, 1))
            native.augment = True  # Local Grad-PU validation extension; native train is True.
            if adapter.kind == 'PU-Gaussian':
                native.compact = None
                native.gt_data = (self.file['poisson_5120'][index:index+1]-center)/radius[..., None]
                native.val = target
                native.x6 = (self.file['poisson_1536'][index:index+1]-center)/radius[..., None]
        state = np.random.get_state()
        np.random.seed(stable_seed(self.seed, 'publisher-augmentation', self.epoch, index))
        try:
            values = native[0]
        finally:
            np.random.set_state(state)
        points, target, scale = values[:3]
        if backbone:
            points, target = points.transpose(0, 1), target.transpose(0, 1)
        result = dict(input=points.float(), gt=target.float(), radius=torch.as_tensor(scale).float())
        if adapter.kind == 'PU-Gaussian':
            result.update(x20=target.float(), gt=values[3].float(), x6=values[4].float())
        return result
