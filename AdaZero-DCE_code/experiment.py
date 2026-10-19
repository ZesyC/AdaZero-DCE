"""Reproducible pilot/full head experiment; eval15 never supplies oracle labels."""
import argparse
import sys
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_dce_adaptive_n import (FACTORS, brightness_features, compute_oracle_n,
                                curve_outputs, load_dce, predict_n, save_head, train_head)


def read_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255


def paired_paths(root, split, expected):
    low, high = root / split / 'low', root / split / 'high'
    a = {p.name: p for p in low.glob('*.png')}
    b = {p.name: p for p in high.glob('*.png')}
    if len(a) != expected or a.keys() != b.keys():
        raise ValueError(f'{split}: expected {expected} matching low/high PNG pairs, got {len(a)}/{len(b)}')
    return [(a[name], b[name]) for name in sorted(a)]


def check_disjoint(train, test):
    # Hash both low and high, catching renamed copies as well as shared paths.
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for pair in train for p in pair}
    test_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for pair in test for p in pair}
    if train_hashes & test_hashes:
        raise ValueError('Train/test contain identical files')


def metrics(output, gt):
    return (float(peak_signal_noise_ratio(gt, output, data_range=1)),
            float(structural_similarity(gt, output, data_range=1, channel_axis=2)))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    start = time.time()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    train = paired_paths(args.data, 'our485', 485)
    test = paired_paths(args.data, 'eval15', 15)
    check_disjoint(train, test)
    order = np.random.default_rng(42).permutation(len(train))
    chosen = [train[i] for i in order[:args.train_pairs]]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_dce(args.weights, device)
    checksum = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    manifest = {'mode': 'full' if args.train_pairs == 485 else 'pilot',
                'train_pairs': args.train_pairs, 'train_files': [a.name for a, _ in chosen],
                'eval_files': [a.name for a, _ in test], 'seeds': args.seeds,
                'epochs': args.epochs, 'weights_sha256': checksum, 'device': device,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
                'resolution': 'original, RGB float32 [0,1], no resize/crop/quantization',
                'ssim': 'skimage RGB channel average; win_size=7, uniform weights, sample covariance',
                'split_seed': 42, 'factors': FACTORS, 'dark_threshold': 0.10,
                'eval_oracles': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    features, targets, labels = [], [], []
    for i, (low_path, gt_path) in enumerate(chosen):
        low, gt = read_image(low_path), read_image(gt_path)
        for factor in FACTORS:
            img = low if factor == 1 else gt * factor
            label = compute_oracle_n(img, gt, net)
            features.append(brightness_features(img))
            targets.append(label)
            labels.append({'file': low_path.name, 'factor': factor, 'oracle_n': label})
        if (i + 1) % 8 == 0:
            print(f'Oracle train pairs: {i + 1}/{len(chosen)}', flush=True)
    features, targets = np.stack(features), np.asarray(targets, dtype=np.float32)
    np.savez(out / 'train_cache.npz', features=features, targets=targets)
    write_csv(out / 'train_oracles.csv', labels)
    oracle_seconds = time.time() - start
    heads, train_info = {}, []
    for seed in args.seeds:
        tick = time.time()
        head, mu, sd, losses = train_head(features, targets, args.epochs, seed)
        save_head(out / f'head_seed{seed}.pth', head, mu, sd)
        heads[seed] = (head, mu, sd)
        pred = head(torch.from_numpy((features - mu) / sd)).detach().numpy().ravel()
        train_info.append({'seed': seed, 'seconds': time.time() - tick,
                           'final_loss': losses[-1], 'train_mae': float(np.abs(pred - targets).mean())})
        (out / f'loss_seed{seed}.json').write_text(json.dumps(losses))
        print(f'Head seed {seed}: {train_info[-1]}', flush=True)
    rows = []
    for low_path, gt_path in test:
        low, gt = read_image(low_path), read_image(gt_path)
        if low.shape != gt.shape:
            raise ValueError('Eval low/GT dimensions differ')
        for factor in FACTORS:
            img = low if factor == 1 else gt * factor
            # GT is used for metrics only; never select n from eval metrics.
            choices = {seed: predict_n(img, *head) for seed, head in heads.items()}
            needed = set(choices.values()) | {8}
            outputs = {n: image for n, image in enumerate(curve_outputs(img, net), 1) if n in needed}
            fixed_psnr, fixed_ssim = metrics(outputs[8], gt)
            for seed, n in choices.items():
                psnr, ssim = metrics(outputs[n], gt)
                rows.append({'file': low_path.name, 'factor': factor, 'seed': seed, 'n': n,
                             'input_mean': float(img.mean()), 'fixed_psnr': fixed_psnr,
                             'fixed_ssim': fixed_ssim, 'adaptive_psnr': psnr, 'adaptive_ssim': ssim})
            if low_path == test[0][0]:
                example = np.concatenate([img, outputs[8], outputs[choices[args.seeds[0]]], gt], axis=1)
                Image.fromarray((example.clip(0, 1) * 255).round().astype('uint8')).save(out / f'example_factor{factor}.png')
        print(f'Evaluated {low_path.name}', flush=True)
    write_csv(out / 'per_image.csv', rows)
    summary = []
    for factor in FACTORS:
        for seed in args.seeds:
            selected = [r for r in rows if r['factor'] == factor and r['seed'] == seed]
            result = {'factor': factor, 'seed': seed}
            for key in ('fixed_psnr', 'fixed_ssim', 'adaptive_psnr', 'adaptive_ssim', 'n'):
                result[key] = float(np.mean([r[key] for r in selected]))
            result['delta_psnr'] = result['adaptive_psnr'] - result['fixed_psnr']
            result['delta_ssim'] = result['adaptive_ssim'] - result['fixed_ssim']
            result['dark_count'] = sum(r['input_mean'] < 0.10 for r in selected)
            summary.append(result)
    write_csv(out / 'summary.csv', summary)
    report = {'summary': summary, 'training': train_info, 'oracle_seconds': oracle_seconds,
              'total_seconds': time.time() - start,
              'head_parameters': sum(p.numel() for p in next(iter(heads.values()))[0].parameters()),
              'oracle_histogram': {str(n): int((targets == n).sum()) for n in range(1, 9)}}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    assert checksum == hashlib.sha256(args.weights.read_bytes()).hexdigest()
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Folder containing our485/ and eval15/')
    p.add_argument('--weights', type=Path, default=Path('AdaZero-DCE_code/snapshots/Epoch99.pth'))
    p.add_argument('--output', type=Path, default=Path('runs/pilot'))
    p.add_argument('--train-pairs', type=int, default=96)
    p.add_argument('--epochs', type=int, default=1200)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if not 1 <= args.train_pairs <= 485 or args.epochs < 1:
        raise SystemExit('train-pairs must be 1..485; epochs must be positive')
    run(args)
