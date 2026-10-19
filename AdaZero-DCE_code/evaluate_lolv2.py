"""Evaluate frozen LOLv1 heads on LOLv2-real Test only; no training or oracles."""
import argparse
import sys
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import metrics, read_image, write_csv
from zero_dce_adaptive_n import curve_outputs, load_dce, load_head, predict_n


def test_pairs(root):
    sides = []
    for directory, prefix in [('Low', 'low'), ('Normal', 'normal')]:
        paths = list((root / directory).glob('*.png'))
        mapping = {}
        for path in paths:
            match = re.fullmatch(prefix + r'(\d{5})\.png', path.name)
            if not match:
                raise ValueError(f'Unexpected test filename: {path.name}')
            mapping[match[1]] = path
        if len(mapping) != 100:
            raise ValueError(f'Expected exactly 100 {directory} test PNGs, got {len(mapping)}')
        sides.append(mapping)
    if sides[0].keys() != sides[1].keys():
        raise ValueError('Low and Normal test IDs do not match')
    return [(sides[0][key], sides[1][key]) for key in sorted(sides[0])]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def run(args):
    start = time.time()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    pairs = test_pairs(args.test)
    args.output.mkdir(parents=True, exist_ok=False)
    head_paths = {seed: args.heads / f'head_seed{seed}.pth' for seed in [0, 1, 2]}
    expected = json.loads(args.verification.read_text())['checkpoint_sha256']
    before = {path.name: sha256(path) for path in [args.weights, *head_paths.values()]}
    for path in head_paths.values():
        if before[path.name] != expected[path.name]:
            raise ValueError(f'Head checksum differs from full LOLv1 run: {path.name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_dce(args.weights, device)
    heads = {s: load_head(path) for s, path in head_paths.items()}
    for head, _, _ in heads.values():
        head.requires_grad_(False)
    rows, files = [], []
    for index, (low_path, gt_path) in enumerate(pairs):
        low, gt = read_image(low_path), read_image(gt_path)
        if low.shape != gt.shape:
            raise ValueError(f'Low/Normal shape mismatch: {low_path.name}')
        n_by_seed = {s: predict_n(low, *head) for s, head in heads.items()}
        needed = set(n_by_seed.values()) | {8}
        outputs = {n: image for n, image in enumerate(curve_outputs(low, net), 1) if n in needed}
        fixed_psnr, fixed_ssim = metrics(outputs[8], gt)
        for seed, n in n_by_seed.items():
            psnr, ssim = metrics(outputs[n], gt)
            rows.append({'low': low_path.name, 'normal': gt_path.name, 'seed': seed,
                         'input_mean': float(low.mean()), 'n': n,
                         'fixed_psnr': fixed_psnr, 'adaptive_psnr': psnr, 'delta_psnr': psnr-fixed_psnr,
                         'fixed_ssim': fixed_ssim, 'adaptive_ssim': ssim, 'delta_ssim': ssim-fixed_ssim})
        files.append({'low': low_path.name, 'normal': gt_path.name, 'shape': list(low.shape),
                      'low_sha256': sha256(low_path), 'normal_sha256': sha256(gt_path)})
        if index == 0:
            example = np.concatenate([low, outputs[8], outputs[n_by_seed[0]], gt], axis=1)
            Image.fromarray((example.clip(0, 1)*255).round().astype('uint8')).save(args.output/'example.png')
        if (index+1) % 10 == 0:
            print(f'Evaluated {index+1}/100 test pairs', flush=True)
    summary = []
    for seed in [0, 1, 2]:
        selected = [r for r in rows if r['seed'] == seed]
        result = {'seed': seed, 'images': len(selected)}
        for key in ['fixed_psnr', 'adaptive_psnr', 'delta_psnr', 'fixed_ssim', 'adaptive_ssim', 'delta_ssim', 'n']:
            result[key] = float(np.mean([r[key] for r in selected]))
        result['dark_count'] = sum(r['input_mean'] < .10 for r in selected)
        result['n_histogram'] = {str(n): sum(r['n'] == n for r in selected) for n in range(1, 9)}
        result['psnr_improved'] = sum(r['delta_psnr'] > 0 for r in selected)
        result['psnr_equal'] = sum(r['delta_psnr'] == 0 for r in selected)
        result['psnr_worse'] = sum(r['delta_psnr'] < 0 for r in selected)
        summary.append(result)
    after = {path.name: sha256(path) for path in [args.weights, *head_paths.values()]}
    assert before == after
    assert all(not p.requires_grad and p.grad is None for p in net.parameters())
    assert all(not p.requires_grad and p.grad is None for h, _, _ in heads.values() for p in h.parameters())
    assert len(rows) == 300
    manifest = {'dataset': 'LOLv2-real', 'split': 'Real_captured/Test', 'pairs': len(pairs),
                'seeds': [0, 1, 2], 'trained': False, 'oracle_computed': False, 'dark_threshold': .10,
                'normalization': 'unchanged mu/sd from LOLv1 full checkpoints', 'hashes_before': before,
                'hashes_after': after, 'device': device,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'torch': torch.__version__, 'numpy': np.__version__,
                'metrics': 'Same experiment.metrics as LOLv1; RGB float32 [0,1], SSIM uniform7 sample covariance',
                'preprocessing': 'original resolution, no resize/crop/GT mean correction', 'files': files}
    write_csv(args.output/'per_image.csv', rows)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = {'summary': summary, 'seconds': time.time()-start}
    (args.output/'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', type=Path, required=True, help='LOLv2 Real_captured/Test containing Low/ and Normal/')
    p.add_argument('--heads', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
