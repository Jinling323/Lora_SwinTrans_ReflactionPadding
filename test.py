"""Compare baseline and LoRA checkpoints on clean, hazy, and mix data."""

import argparse
import csv
import glob
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd
from models import swin_c_multibatch as models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline and LoRA checkpoints on three domains')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_t_trans')
    parser.add_argument('--baseline-checkpoint', default='model/swin_t_trans/baseline/0922-200006/best_model.pth',
                        help='baseline checkpoint; defaults to the latest best_model*.pth')
    parser.add_argument('--lora-checkpoint', default='model/swin_t_trans/lora/0923-103544/best_model.pth',
                        help='LoRA checkpoint; defaults to the latest best_model*.pth')
    parser.add_argument('--checkpoint-root', default='model',
                        help='root used to find default checkpoints')
    parser.add_argument('--data-root', default='SHA',
                        help='root containing clean/, hazy/, and mix/')
    parser.add_argument('--domains', nargs='+', choices=['clean', 'hazy', 'mix'],
                        default=['clean', 'hazy', 'mix'])
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output-csv', default='test_results.csv')
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--lora-rank', type=int, default=4,
                        help='required when the LoRA checkpoint is a .pth file')
    parser.add_argument('--lora-alpha', type=float, default=4.0,
                        help='must match training for a LoRA .pth checkpoint')
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def resolve_checkpoint(explicit_path, checkpoint_root, model_name, stage):
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(explicit_path))
        return explicit_path

    pattern = os.path.join(
        checkpoint_root, model_name, stage, '*', 'best_model*.pth')
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            'No default {} checkpoint matched {}; provide --{}-checkpoint'.format(
                stage, pattern, stage))
    return max(candidates, key=os.path.getmtime)


def build_model(args, checkpoint_path, expect_lora):
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if checkpoint.get('model_name', args.model_name) != args.model_name:
            raise ValueError(
                '{} model name does not match --model-name'.format(checkpoint_path))
        state = checkpoint['model_state_dict']
        rank = checkpoint.get('lora_rank', 0)
        alpha = checkpoint.get('lora_alpha', args.lora_alpha)
    else:
        state = checkpoint
        rank = args.lora_rank if expect_lora else 0
        alpha = args.lora_alpha

    contains_lora = any('.parametrizations.weight.' in key for key in state)
    if expect_lora:
        if not contains_lora:
            raise ValueError('{} does not contain LoRA weights'.format(checkpoint_path))
        if not rank:
            raise ValueError(
                'LoRA .pth checkpoint needs --lora-rank and --lora-alpha')
    elif contains_lora or rank:
        raise ValueError(
            '{} is a LoRA checkpoint, not a baseline checkpoint'.format(
                checkpoint_path))

    model = getattr(models, args.model_name)(pretrained=False)
    if expect_lora:
        model.enable_lora(rank, alpha)
    model.load_state_dict(state)
    return model


def split_large_image(inputs, maximum_size=3584):
    _, _, height, width = inputs.shape
    if height < maximum_size and width < maximum_size:
        return [inputs]

    h_splits = math.ceil(height / maximum_size)
    w_splits = math.ceil(width / maximum_size)
    h_step = height // h_splits
    w_step = width // w_splits
    return [
        inputs[
            :, :,
            i * h_step:(i + 1) * h_step if i + 1 < h_splits else height,
            j * w_step:(j + 1) * w_step if j + 1 < w_splits else width,
        ]
        for i in range(h_splits)
        for j in range(w_splits)
    ]


def evaluate(model, data_dir, args, device, description):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(data_dir))
    dataset = Crowd(data_dir, args.crop_size, args.downsample_ratio,
                    is_gray=False, method='val')
    if not dataset.im_list:
        raise ValueError('No .jpg images in {}'.format(data_dir))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    errors = []
    with torch.inference_mode():
        for inputs, count, _ in tqdm(
                loader, desc=description, dynamic_ncols=True):
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            predicted = sum(
                model(patch)[0].sum().item()
                for patch in split_large_image(inputs)
            )
            errors.append(count.item() - predicted)

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'images': len(errors),
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(np.square(errors)))),
    }


def print_results(rows):
    print('\n{:<10} {:<8} {:>8} {:>12} {:>12}'.format(
        'Model', 'Domain', 'Images', 'MAE', 'RMSE'))
    print('-' * 54)
    for row in rows:
        print('{model:<10} {domain:<8} {images:>8d} '
              '{mae:>12.3f} {rmse:>12.3f}'.format(**row))


def write_results(rows, output_path):
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    fields = ['model', 'domain', 'split', 'images', 'mae', 'rmse', 'checkpoint']
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    baseline_checkpoint = resolve_checkpoint(
        args.baseline_checkpoint, args.checkpoint_root,
        args.model_name, 'baseline')
    lora_checkpoint = resolve_checkpoint(
        args.lora_checkpoint, args.checkpoint_root,
        args.model_name, 'lora')
    print('Baseline checkpoint: {}'.format(baseline_checkpoint))
    print('LoRA checkpoint: {}'.format(lora_checkpoint))

    checkpoints = (
        ('baseline', baseline_checkpoint, False),
        ('lora', lora_checkpoint, True),
    )
    for model_label, checkpoint_path, expect_lora in checkpoints:
        model = build_model(args, checkpoint_path, expect_lora)
        model.to(device).eval()
        for domain in args.domains:
            data_dir = os.path.join(args.data_root, domain, args.split)
            metrics = evaluate(
                model, data_dir, args, device,
                '{} {}'.format(model_label.capitalize(), domain))
            rows.append({
                'model': model_label,
                'domain': domain,
                'split': args.split,
                'images': metrics['images'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'checkpoint': checkpoint_path,
            })
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print_results(rows)
    write_results(rows, args.output_csv)
    print('\nCSV saved to {}'.format(os.path.abspath(args.output_csv)))


if __name__ == '__main__':
    main()
