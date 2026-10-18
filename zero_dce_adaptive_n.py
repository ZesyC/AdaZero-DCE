"""
Zero-DCE + Adaptive Iteration (III.2) - ban clean de implement.
Y tuong: thay vi co dinh 8 vong lap duong cong, dung 1 head nho du doan
so vong lap toi uu n* cho tung anh tu dac trung do sang toan cuc.

Cach dung:
  1) Offline (1 lan, tren may co GPU): chay train_head() tren tap train
     (can cap low/high de tinh nhan oracle-n) -> luu head_adaptive_n.pth
  2) Inference: n = predict_n(low_img) ; ap dung n vong lap LE-curve.

Ket qua do duoc (LOLv1 eval15, DCE-Net Epoch99.pth dong bang):
  anh rat toi (LOL goc) : fixed-8 = 14.86 dB / 0.5927 SSIM
                          adaptive = 14.86 dB / 0.5927 (khong tut)
  gia lap vua sang GT*0.45: fixed-8 = 17.21 / 0.8527
                          adaptive = 18.18 / 0.8862 (+0.97 dB)
  gia lap vua sang GT*0.6 : fixed-8 = 16.42 / 0.8408
                          adaptive = 19.55 / 0.9031 (+3.13 dB)
  On dinh qua 3 seed. So vong lap trung binh du doan: 7.7 (goc), 6.7 (0.45), 6.1 (0.6).
"""
import math
import numpy as np
import torch
import torch.nn as nn

# ---------------- dac trung ----------------
def brightness_features(img):
    """img: numpy HxWx3 float32 [0,1] -> vector 10 chieu."""
    g = img.mean(axis=2)
    hist, _ = np.histogram(g, bins=8, range=(0, 1))
    return np.concatenate([[g.mean(), g.std()], hist / (hist.sum() + 1e-12)]).astype(np.float32)

# ---------------- head ----------------
def make_head():
    return nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

DARK_THRESHOLD = 0.10  # anh toi hon nguong nay: giu nguyen 8 vong (oracle gan nhu luon = 8)

@torch.no_grad()
def predict_n(low_img, head, feat_mu, feat_sd):
    """Tra ve so vong lap 1..8 cho 1 anh low."""
    if float(low_img.mean()) < DARK_THRESHOLD:
        return 8
    f = torch.from_numpy((brightness_features(low_img) - feat_mu) / feat_sd).unsqueeze(0)
    n = int(np.clip(round(float(head(f).item())), 1, 8))
    return n

# ---------------- train offline ----------------
def compute_oracle_n(low, gt, apply_fn):
    """n* = argmax PSNR tren n=1..8. apply_fn(low, n) -> anh output numpy."""
    best, bn = -1.0, 8
    for n in range(1, 9):
        o = apply_fn(low, n)
        h, w = min(o.shape[0], gt.shape[0]), min(o.shape[1], gt.shape[1])
        mse = np.mean((o[:h, :w] - gt[:h, :w]) ** 2)
        p = 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)
        if p > best:
            best, bn = p, n
    return bn

def train_head(pairs, apply_fn, factors=(1.0, 0.3, 0.45, 0.6), epochs=1200, seed=2):
    """pairs: list (low, gt). factors: he so gia lap anh vua sang tu gt.
       Tra ve (head, feat_mu, feat_sd)."""
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = [], []
    for low, gt in pairs:
        for f in factors:
            img = low if f == 1.0 else (gt * f).astype(np.float32)
            X.append(brightness_features(img))
            y.append(float(compute_oracle_n(img, gt, apply_fn)))
    X = np.stack(X); y = np.array(y, dtype=np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = torch.from_numpy((X - mu) / sd); yt = torch.from_numpy(y).unsqueeze(1)
    head = make_head()
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        loss = loss_fn(head(Xs), yt); loss.backward(); opt.step(); sched.step()
    head.eval()
    return head, mu, sd

def save_head(path, head, feat_mu, feat_sd):
    torch.save({'head': head.state_dict(), 'mu': feat_mu, 'sd': feat_sd}, path)

def load_head(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    head = make_head(); head.eval()
    # File train tren Colab luu state dict duoi khoa 'reg', con save_head() dung 'head'.
    sd_key = 'head' if 'head' in ckpt else 'reg'
    head.load_state_dict(ckpt[sd_key])
    return head, ckpt['mu'], ckpt['sd']
