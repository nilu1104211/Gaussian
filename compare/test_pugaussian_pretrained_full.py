"""Test official PU-Gaussian weights on full PU1K (127) and PU-GAN (27) sets.

Reuses the formal retraining inference adapter and unified metric functions.
Each run saves original input/GT/meshes, all predictions, PLY, metrics and logs
in a new directory, then packages the run for download. No training is run.
"""
import argparse
import re
from pathlib import Path

import test_examples as examples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, default=examples.DEFAULT_RUN,
                        help='Completed formal run used only as inference config/data reference')
    parser.add_argument('--prepared-root', type=Path, help='Prepared root containing PU1K/ and PU-GAN/')
    parser.add_argument('--pretrained-root', type=Path, default=examples.PRETRAINED_ROOT)
    parser.add_argument('--datasets', choices=examples.DATASETS, nargs='+', default=list(examples.DATASETS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=Path, default=examples.ROOT / 'compare/test-runs')
    parser.add_argument('--tag', default='pugaussian-pretrained-full')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Check full object counts, names and hashes without inference/writes')
    modes.add_argument('--package-run', type=Path, help='Package an existing export without inference')
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate datasets are not allowed')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('--tag must contain only ASCII letters, digits, hyphen and underscore')
    # Fixed full-test profile: this entry point has no sample-size or shape filter.
    args.models = [examples.PRETRAINED_MODEL]
    args.all_shapes = True
    args.shapes = None
    args.num_examples = None
    args.metrics = True
    args.no_previews = True
    args.entrypoint = __file__
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.package_run:
        examples.repackage_run(args.package_run)
    else:
        examples.execute(args)


if __name__ == '__main__':
    main()
