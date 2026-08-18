'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import faulthandler
faulthandler.enable()
print("Faulthandler enabled")

import torch
assert torch.cuda.is_available(), "No CUDA!"
_ = torch.zeros(1, device='cuda')
del _
print(f"* PyTorch CUDA OK: {torch.cuda.get_device_name(0)}")

# Note: the Numba @cuda.jit eval kernels are loaded lazily by
# utils/kitti_eval/eval.py only when evaluation runs. Importing them eagerly
# here initialized a Numba CUDA context at startup, which crashes when the
# Numba/CUDA driver versions do not match the host. Inference does not need
# them, so the eager import is removed.
import argparse
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Pipeline Evaluation')
    parser.add_argument('--config', type=str, default='./configs/ASF_v2_0_final.yml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='./pretrained/A2F_v2_0_final_10.pt',
                        help='Path to pretrained model')
    parser.add_argument('--conf_thr', type=float, nargs='+', default=[0.0],
                        help='List of confidence thresholds (e.g., --conf_thr 0.3 0.5 0.7)')
    parser.add_argument('--subset', action='store_true',
                        help='Use subset for validation')
    parser.add_argument('--print_memory', action='store_true',
                        help='Print memory usage')
    parser.add_argument('--noise-config', type=str, default=None,
                        help='Path to a noise-injection YAML (datasets/effects). '
                             'Omit for a clean run — the noise-injection module is '
                             'not even imported in that case.')
    parser.add_argument('--blackout-policy', type=str, default='empty',
                        choices=('empty', 'zero_feature', 'last_frame_hold'),
                        help='How a blacked-out modality (frame_deletion / loss_complete) '
                             'is represented to the model. Only used with --noise-config. '
                             'See datasets/effects/eval_wrapper.py for the rationale.')
    parser.add_argument('--dump-sensors', type=str, default=None,
                        help='Directory to write each frame\'s corrupted sensor data to '
                             '(point clouds as .npy, camera images as .png, plus a _meta.txt), '
                             'with clean copies alongside for comparison. Only used with '
                             '--noise-config.')
    parser.add_argument('--dump-meta-only', action='store_true',
                        help='With --dump-sensors, write only <stem>_meta.txt per frame '
                             '(frame index, seq, blackout list, sensor indices, and a '
                             'changed=True/False flag per modality) and skip the actual '
                             '.npy/.png sensor arrays. For large sweeps where full sensor '
                             'dumps would be a real disk-space risk but the changed-flag '
                             'bookkeeping is still wanted.')

    args = parser.parse_args()

    PATH_CONFIG = args.config
    PATH_MODEL = args.model

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='test')
    pline.load_dict_model(PATH_MODEL)

    pline.network.eval()

    if args.noise_config is not None:
        from datasets.effects import NoiseInjectedDataset, install_blackout_hooks, load_injector

        injector = load_injector(args.noise_config)
        print(f'[noise-injection] Loaded config from {args.noise_config} '
              f'(seed={injector.config.seed}, blackout_policy={args.blackout_policy}, '
              f'{injector._metadata()})')
        pline.dataset_test = NoiseInjectedDataset(
            pline.dataset_test, injector,
            blackout_policy=args.blackout_policy,
            dump_dir=args.dump_sensors,
            dump_arrays=not args.dump_meta_only,
        )
        if args.dump_sensors:
            mode = 'meta only (no .npy/.png arrays)' if args.dump_meta_only else 'full arrays + meta'
            print(f'[noise-injection] Dumping per-frame sensor data to {args.dump_sensors} ({mode})')
        if args.blackout_policy == 'zero_feature':
            n_hooks = install_blackout_hooks(pline.network)
            print(f'[noise-injection] Installed {n_hooks} blackout forward hooks')

    # Save the code for identification
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

    pline.validate_kitti_conditional(
        list_conf_thr=args.conf_thr,
        is_subset=args.subset,
        is_print_memory=args.print_memory
    )
