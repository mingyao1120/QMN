import argparse
import time
import os
from pathlib import Path

from utils import load_json
import shutil
import zipfile


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='config file path')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    parser.add_argument('--tag', default='base', type=str, help='experiment tag')
    parser.add_argument('--vote',default=False, type=bool, help='use vote-based strategy during inference')
    parser.add_argument('--seed', default=8, type=int, help='random seed')
    

    parser.add_argument('--Lambda', default=0, type=float, help='loss lambda')
    # 加一行用来平衡自己的proposal和原始proposal
    parser.add_argument('--ratio', default=1, type=float, help='proposal fusing ratio')
    parser.add_argument('--margin-dis', default=0, type=float, help='proposal guass bias ratio')
    parser.add_argument('--cc-loss-cof', default=0, type=float, help='cc loss ef')
    parser.add_argument('--vtc-loss-cof', default=0, type=float, help='vtc loss ef')
    
    parser.add_argument('--num-props', default=8, type=int, help='number of prpos')
    
    return parser.parse_args()


def main(kargs):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    seed = kargs.seed
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = load_json(kargs.config_path) # 这里可以改内容
    if kargs.Lambda != 0: # 为0就不更改cofig里面的值
        args['loss']['lambda'] = kargs.Lambda
    # 读取ratio
    if kargs.ratio != 1:
        args['model']['config']['ratio'] = kargs.ratio

    if kargs.margin_dis != 0:
        args['model']['config']['margin_dis'] = kargs.margin_dis

    if kargs.cc_loss_cof !=0:
        args['loss']['cc_loss_cof'] = kargs.cc_loss_cof

    if kargs.vtc_loss_cof != 0:
        args['loss']['vtc_loss_cof'] = kargs.vtc_loss_cof

    if kargs.num_props !=0:
        args['model']['config']['num_props'] = kargs.num_props
    print( kargs.num_props)
    
    if kargs.log_dir: # 这里规定log位置
        log_filename = os.path.join('checkpoints', "{}/{}_{}.log".format(args['dataset']['dataset'], kargs.seed, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    args['train']['model_saved_path'] = os.path.join(args['train']['model_saved_path'], str(kargs.seed)+"_"+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())) # 这里添加路径为随机种子
    print(kargs.vote)
    args['vote'] = kargs.vote
    logging.info(str(args))
    runner = MainRunner(args)

    if kargs.resume:
        runner._load_model(kargs.resume)
    if kargs.eval:
        runner.eval()
        return
    runner.train()


if __name__ == '__main__':  
    args = parse_args()
    main(args)
