"""Test the official shared PU-CRN release on full PU1K (127) / PU-GAN (27).

The official model.pth is supplied with PU1K evaluation files. This entry point
uses that SAME checkpoint on both datasets; it does not claim a dataset-specific
PU-GAN release. NativeAdapter's E0 backbone and complete inference path are used
with strict loading of all official parameters. Outputs are saved separately.
"""
import argparse
import re
from pathlib import Path

import test_examples as examples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, default=examples.DEFAULT_RUN,
                        help='Completed formal E0 run used only as inference config/data reference')
    parser.add_argument('--prepared-root', type=Path, help='Prepared root containing PU1K/ and PU-GAN/')
    parser.add_argument('--pretrained-root', type=Path, dest='pucrn_pretrained_root',
                        default=examples.PUCRN_PRETRAINED_ROOT)
    parser.add_argument('--datasets', choices=examples.DATASETS, nargs='+', default=list(examples.DATASETS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=Path, default=examples.ROOT / 'compare/test-runs')
    parser.add_argument('--tag', default='pucrn-pretrained-full')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Check complete counts and hashes without inference/writes')
    modes.add_argument('--package-run', type=Path, help='Package an existing completed export without inference')
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate datasets are not allowed')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('--tag must contain only ASCII letters, digits, hyphen and underscore')
    args.models = [examples.PUCRN_PRETRAINED_MODEL]
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
        if not args.plan:
            print('Official PU-CRN: the SAME release checkpoint is used on both datasets. '
                  'The supplied release evaluation targets PU1K; no separate PU-GAN weight is claimed.', flush=True)
        examples.execute(args)


if __name__ == '__main__':
    main()
