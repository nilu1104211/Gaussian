"""Presets for the structure study; all use the original E6 training loss."""
NEW_VARIANTS = ('M0', 'V1a', 'V1', 'V2a', 'V2', 'V3', 'Tail-only')
NEW_FLAGS = ('feature_bridge', 'learned_tangent', 'learned_normal', 'geometry_gate',
             'use_f2_skip', 'use_geo_skip', 'use_neighborhood_refine', 'use_tangent_normal_head')
NEW_BOUNDS = dict(tangent_bound=0.25, learned_normal_bound=0.1,
                  refine_tangent_bound=0.1, refine_normal_bound=0.05)


def variant_options(name):
    if name not in ('E6', *NEW_VARIANTS):
        raise ValueError('Unknown PU-GCSNet variant: ' + name)
    options = dict(stage2='quadratic', sampling='deterministic', refinement='local', refine_radius=0.5,
                   k=16, sigma_min=0.15, sigma_max=0.75, center_bound=0.2,
                   normal_bound=0.2, height_bound=0.25, rho=1.0,
                   **dict.fromkeys(NEW_FLAGS, False), **NEW_BOUNDS)
    if name in ('V1a', 'V1', 'V3'):
        options['learned_tangent'] = True
    if name in ('M0', 'V1', 'V3'):
        options['feature_bridge'] = True
    if name in ('V2a', 'V2', 'V3'):
        options['use_f2_skip'] = True
    if name in ('V2', 'V3'):
        options.update(use_geo_skip=True, use_tangent_normal_head=True)
    if name == 'Tail-only':
        options['refinement'] = 'legacy'
    return options


def add_structure_arguments(parser):
    for name in NEW_FLAGS:
        parser.add_argument('--' + name.replace('_', '-'), action='store_true', default=False)
    for name, default in NEW_BOUNDS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=default)
