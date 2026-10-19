"""Enhance one image using the frozen backbone and a trained head."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zero_dce_adaptive_n import enhance, load_dce, load_head

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path('AdaZero-DCE_code/snapshots/Epoch99.pth'))
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise SystemExit('Output must be a new file, separate from input')
    with Image.open(args.input) as im:
        img = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    net = load_dce(args.weights, 'cuda' if torch.cuda.is_available() else 'cpu')
    output, n = enhance(img, net, *load_head(args.head))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((output.clip(0, 1) * 255).round().astype('uint8')).save(args.output)
    print(f'n={n}; saved {args.output}')
