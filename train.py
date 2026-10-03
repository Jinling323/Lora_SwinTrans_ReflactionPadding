"""Train the MAN head with a Swin backbone, then adapt it with Swin LoRA."""

import argparse
import copy
import gc
import os
import random

import numpy as np
import torch

from utils.regression_trainer_cosine_multibatch import RegTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Swin + MAN')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_t_trans',
                        help='Swin-L has 24 attention blocks; Swin-T has 12')
    parser.add_argument('--stage', choices=['both', 'baseline', 'lora'], default='both',
                        help='run baseline then LoRA, or select one stage')
    parser.add_argument('--data-dir', default='SHA/clean',
                        help='clean baseline root containing train/ and val/')
    parser.add_argument('--lora-train-dir', default='SHA/hazy/train')
    parser.add_argument('--lora-val-dir', default='SHA/mix/val')
    parser.add_argument('--baseline-checkpoint', default='',
                        help='trained clean baseline .pth or .tar for LoRA-only training')
    parser.add_argument('--pretrained-path',
                        default='pre_models/swin_large_patch4_window12_384_22k.pth',
                        help='official ImageNet-22K Swin-L checkpoint; also works with 256 crops')
    parser.add_argument('--no-pretrained-backbone', dest='pretrained_backbone',
                        action='store_false')
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument('--lora-rank', type=int, default=4)
    parser.add_argument('--lora-alpha', type=float, default=4.0)
    parser.add_argument(
        '--parallel-dwconv', dest='parallel_dwconv', action='store_true',
        help='add a trainable 5x5 DWConv V adapter in the LoRA stage (default)')
    parser.add_argument(
        '--no-parallel-dwconv', dest='parallel_dwconv', action='store_false',
        help='disable the parallel V-map DWConv branch')
    parser.add_argument(
        '--reflection-attention', dest='reflection_attention', action='store_true',
        help='enable reflection-padding window attention')
    parser.add_argument(
        '--no-reflection-attention', dest='reflection_attention',
        action='store_false',
        help='use original cyclic shifted-window attention and masks')
    parser.set_defaults(parallel_dwconv=True, reflection_attention=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='model')
    parser.add_argument('--save-all', action='store_true')
    parser.add_argument('--lr', type=float, default=5*1e-6)
    parser.add_argument('--lora-lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--resume', default='',
                        help='continue the selected stage from its .tar checkpoint')
    parser.add_argument('--max-model-num', type=int, default=1)
    parser.add_argument('--pretrain-epochs', type=int, default=500,
                        help='number of clean baseline training epochs')
    parser.add_argument('--pretrain-val-epoch', type=int, default=5,
                        help='validate the baseline every N pretrain epochs')
    parser.add_argument('--pretrain-val-start', type=int, default=5,
                        help='first zero-based pretrain epoch eligible for validation')
    parser.add_argument('--lora-epochs', '--max-epoch', dest='lora_epochs',
                        type=int, default=1500,
                        help='number of LoRA training epochs')
    parser.add_argument('--lora-val-epoch', '--val-epoch', dest='lora_val_epoch',
                        type=int, default=5,
                        help='validate LoRA every N LoRA training epochs')
    parser.add_argument('--lora-val-start', '--val-start', dest='lora_val_start',
                        type=int, default=200,
                        help='first zero-based LoRA epoch eligible for validation')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--tensorboard-log-interval', type=int, default=10,
                        help='batch interval for TensorBoard and train.log loss')
    parser.add_argument('--is-gray', action='store_true')
    parser.add_argument(
        '--baseline-crop-size', type=int, default=256,
        help='training crop size for the clean baseline stage')
    parser.add_argument(
        '--lora-crop-size', type=int, default=384,
        help='training crop size for the LoRA adaptation stage')
    parser.add_argument(
        '--crop-size', type=int, default=None,
        help='legacy shortcut that sets both stage crop sizes')
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--use-background', type=bool, default=True)
    parser.add_argument('--sigma', type=float, default=8.0)
    parser.add_argument('--background-ratio', type=float, default=0.15)
    args = parser.parse_args()
    if args.crop_size is not None:
        args.baseline_crop_size = args.crop_size
        args.lora_crop_size = args.crop_size
    if args.stage == 'lora' and not (args.baseline_checkpoint or args.resume):
        parser.error('--stage lora needs --baseline-checkpoint or --resume')
    if args.stage == 'both' and args.baseline_checkpoint:
        parser.error('--baseline-checkpoint is for --stage lora')
    for name, epochs, val_epoch, val_start in (
        ('pretrain', args.pretrain_epochs, args.pretrain_val_epoch,
         args.pretrain_val_start),
        ('lora', args.lora_epochs, args.lora_val_epoch, args.lora_val_start),
    ):
        if epochs <= 0 or val_epoch <= 0 or val_start < 0:
            parser.error('{} epochs and validation interval must be positive; '
                         'validation start must be nonnegative'.format(name))
        if ((args.stage == 'both' or
             (args.stage == 'baseline' and name == 'pretrain') or
             (args.stage == 'lora' and name == 'lora')) and
                val_start >= epochs):
            parser.error('--{}-val-start must be less than --{}-epochs '
                         'to select a best checkpoint'.format(name, name))
    for option, crop_size in (
        ('--baseline-crop-size', args.baseline_crop_size),
        ('--lora-crop-size', args.lora_crop_size),
    ):
        if crop_size <= 0 or crop_size % args.downsample_ratio:
            parser.error(
                '{} must be positive and divisible by --downsample-ratio'.format(
                    option
                )
            )
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error('LoRA rank and alpha must be positive')
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_stage(args, stage, baseline_checkpoint=''):
    stage_args = copy.copy(args)
    stage_args.stage = stage
    if stage == 'baseline':
        stage_args.max_epoch = args.pretrain_epochs
        stage_args.val_epoch = args.pretrain_val_epoch
        stage_args.val_start = args.pretrain_val_start
        stage_args.crop_size = args.baseline_crop_size
        # Keep the clean baseline identical to the original Swin model. The
        # DWConv V branch is an adaptation module introduced in the LoRA stage.
        stage_args.parallel_dwconv = False
    else:
        stage_args.max_epoch = args.lora_epochs
        stage_args.val_epoch = args.lora_val_epoch
        stage_args.val_start = args.lora_val_start
        stage_args.crop_size = args.lora_crop_size
    stage_args.save_dir = os.path.join(args.save_dir, args.model_name, stage)
    stage_args.lr = args.lora_lr if stage == 'lora' else args.lr
    if stage == 'lora' and baseline_checkpoint:
        stage_args.baseline_checkpoint = baseline_checkpoint
        stage_args.resume = ''
    seed_everything(stage_args.seed)
    trainer = RegTrainer(stage_args)
    trainer.setup()
    trainer.train()
    return trainer.best_model_path


def run_pipeline(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    if args.stage == 'both':
        best_baseline = run_stage(args, 'baseline')
        if not best_baseline:
            raise RuntimeError('Baseline produced no best checkpoint')
        gc.collect()
        torch.cuda.empty_cache()
        best_lora = run_stage(args, 'lora', baseline_checkpoint=best_baseline)
        return best_baseline, best_lora
    return run_stage(args, args.stage)


if __name__ == '__main__':
    run_pipeline(parse_args())
