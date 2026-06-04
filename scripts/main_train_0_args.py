'''
Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.cuda.init()

import argparse
from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument('--config', type=str, default='./configs/cfg_RTNH_wide.yml',
                        help='Path to config file')
    
    args = parser.parse_args()
    PATH_CONFIG = args.config
    
    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='train')

    ### Save this file for checking ###
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    ### Save this file for checking ###

    pline.train_network()

    # conditional evaluation for last epoch
    pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
