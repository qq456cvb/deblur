"""Synthetic benchmark: blur sharp images with known camera-shake kernels,
run the blind deblurring pipeline and report shift-aligned PSNR / SSIM and
kernel similarity (max normalised cross-correlation with the true kernel).

    python benchmark.py
"""
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import skimage.data
from skimage.metrics import structural_similarity

import main as deblur_lib

CASES = [('camera', 25, 1), ('coffee', 31, 2), ('astronaut', 27, 3), ('chelsea', 21, 4)]
NOISE = 0.005


def motion_kernel(size, seed):
    """Smooth camera-shake-like trajectory (curved stroke with varying speed)."""
    rng = np.random.default_rng(seed)
    n = 400
    start = rng.uniform(0, 2 * np.pi)
    curv = np.cumsum(rng.normal(scale=0.04, size=n))
    curv = curv - np.linspace(0, curv[-1], n) * 0.5
    angles = start + np.cumsum(curv) * 0.05 + np.cumsum(rng.normal(scale=0.02, size=n))
    speed = 0.5 + rng.random() * 0.5 + 0.5 * np.sin(np.linspace(0, rng.uniform(2, 6), n) + rng.uniform(0, 6)) ** 2
    pos = np.cumsum(np.stack([np.cos(angles), np.sin(angles)], 1) * speed[:, None], 0)
    pos -= (pos.max(0) + pos.min(0)) / 2
    pos *= (size / 2 - 2.5) / np.abs(pos).max()
    up = 8
    k = np.zeros((size * up, size * up))
    for x, y in pos * up + size * up / 2:
        k[int(y), int(x)] += 1
    k = cv2.resize(cv2.GaussianBlur(k, (0, 0), up * 0.6), (size, size), interpolation=cv2.INTER_AREA)
    k[k < k.max() * 0.03] = 0
    return k / k.sum()


def make_case(name, ksize, seed):
    img = getattr(skimage.data, name)()
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sharp = img.astype(np.float32) / 255.
    k = motion_kernel(ksize, seed).astype(np.float32)
    p = ksize  # blur a reflected extension so borders look like a real photo
    pad = cv2.copyMakeBorder(sharp, p, p, p, p, cv2.BORDER_REFLECT)
    blur = cv2.filter2D(pad, -1, k[::-1, ::-1], borderType=cv2.BORDER_REFLECT)[p:-p, p:-p]
    blur = blur + np.random.default_rng(seed).normal(scale=NOISE, size=blur.shape).astype(np.float32)
    return sharp, np.clip(blur, 0, 1), k


def aligned_psnr_ssim(est, gt, max_shift, crop):
    """Blind deblurring is defined up to a translation: search the best shift."""
    to_gray = lambda x: cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x
    est, gt = to_gray(est).astype(np.float64), to_gray(gt).astype(np.float64)
    H, W = gt.shape
    g = gt[crop:H - crop, crop:W - crop]
    best = max(((10 * np.log10(1 / np.mean((est[crop + dy:H - crop + dy, crop + dx:W - crop + dx] - g) ** 2)), dy, dx)
                for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)))
    _, dy, dx = best
    e = est[crop + dy:H - crop + dy, crop + dx:W - crop + dx]
    return best[0], structural_similarity(e, g, data_range=1.0)


def kernel_similarity(k_est, k_gt):
    n = max(k_est.shape[0], k_gt.shape[0])
    pad = lambda k: np.pad(k, (((n - k.shape[0]) // 2, (n - k.shape[0] + 1) // 2),) * 2)
    a, b = pad(k_est).astype(np.float32), pad(k_gt).astype(np.float32)
    return float(cv2.matchTemplate(np.pad(a, n // 2), b, cv2.TM_CCORR_NORMED).max())


def run_case(case):
    name, ksize, seed = case
    sharp, blur, k_gt = make_case(name, ksize, seed)
    t0 = time.time()
    latent, k, _ = deblur_lib.deblur(blur, ksize)
    seconds = time.time() - t0
    latent = np.round(latent * 255) / 255
    blur_psnr, blur_ssim = aligned_psnr_ssim(blur, sharp, 0, ksize)
    psnr, ssim = aligned_psnr_ssim(latent, sharp, ksize // 2, ksize)
    return name, blur_psnr, blur_ssim, psnr, ssim, kernel_similarity(k, k_gt), seconds


if __name__ == '__main__':
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run_case, CASES))
    print(f"{'image':10s} {'blurred PSNR/SSIM':>18s} {'restored PSNR/SSIM':>19s} {'kernel sim':>10s} {'time':>6s}")
    for name, bp, bs, p, s, ks, t in rows:
        print(f'{name:10s} {bp:11.2f} / {bs:.3f} {p:12.2f} / {s:.3f} {ks:10.3f} {t:5.1f}s')
    print(f"{'mean':10s} {np.mean([r[1] for r in rows]):11.2f} / {np.mean([r[2] for r in rows]):.3f} "
          f"{np.mean([r[3] for r in rows]):12.2f} / {np.mean([r[4] for r in rows]):.3f} {np.mean([r[5] for r in rows]):10.3f}")
