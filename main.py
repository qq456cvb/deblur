"""Two-Phase Kernel Estimation for Robust Motion Deblurring (Xu & Jia, ECCV 2010).

Conventions used throughout:
  * images are float64 in [0, 1];
  * blur model is a true convolution ``B = I (*) k`` with the kernel centre at
    ``(ksize // 2, ksize // 2)``;
  * all FFT work happens on a periodically-extended ("wrapped") canvas that is
    larger than the image by at least one kernel size, with the image in the
    top-left corner, so circular convolution does not create border ringing;
  * gradients are circular forward differences, ``gx = I(x+1) - I(x)``, which
    match the OTFs built by :func:`diff_otfs` exactly.
"""
import argparse
import time

import cv2
import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from scipy import signal


# --------------------------------------------------------------------------- #
# FFT / convolution helpers
# --------------------------------------------------------------------------- #
def rfft2(x):
    return sfft.rfft2(x, axes=(0, 1), workers=-1)


def irfft2(x, shape):
    return sfft.irfft2(x, s=shape, axes=(0, 1), workers=-1)


def psf2otf(psf, shape):
    """OTF of a centred PSF on a canvas of ``shape`` (rfft layout)."""
    pad = np.zeros(shape)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    pad = np.roll(pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    return rfft2(pad)


def otf2psf(otf, shape, ksize):
    """Inverse of :func:`psf2otf`, cropped to a ``ksize x ksize`` support."""
    x = np.roll(irfft2(otf, shape), (ksize // 2, ksize // 2), axis=(0, 1))
    return x[:ksize, :ksize]


def diff_otfs(shape):
    """OTFs of the forward-difference operators used by :func:`grad`."""
    dx = np.zeros(shape)
    dx[0, 0], dx[0, -1] = -1, 1
    dy = np.zeros(shape)
    dy[0, 0], dy[-1, 0] = -1, 1
    return rfft2(dx), rfft2(dy)


def grad(img):
    """Circular forward differences (consistent with :func:`diff_otfs`)."""
    return np.roll(img, -1, axis=1) - img, np.roll(img, -1, axis=0) - img


def fast_shape(h, w, ksize):
    """FFT-friendly canvas that leaves at least ``ksize`` pixels of margin."""
    return sfft.next_fast_len(h + ksize, real=True), sfft.next_fast_len(w + ksize, real=True)


def wrap_boundary(img, shape):
    """Extend ``img`` (top-left) to ``shape`` so the result tiles smoothly.

    The gap between the right (bottom) image border and the left (top) border
    of the next periodic copy is filled by blending the two borders, each
    progressively blurred towards the middle of the gap. This removes the hard
    step that zero padding would introduce into the circular blur model.
    """
    h, w = img.shape[:2]
    H, W = shape
    out = np.zeros((H, W) + img.shape[2:], np.float64)
    out[:h, :w] = img

    def fill(a, c, n, periodic):
        mode = 'wrap' if periodic else 'reflect'
        cols = []
        for j in range(n):
            t = (j + 1) / (n + 1)
            sigma = min(t, 1 - t) * n / 2.
            if sigma > 0.3:
                av = ndimage.gaussian_filter1d(a, sigma, axis=0, mode=mode)
                cv = ndimage.gaussian_filter1d(c, sigma, axis=0, mode=mode)
            else:
                av, cv = a, c
            cols.append((1 - t) * av + t * cv)
        return np.stack(cols, 1)

    if W > w:
        out[:h, w:] = fill(img[:, -1], img[:, 0], W - w, False)
    if H > h:
        # rows are periodic with period W now, so blur them with wrap mode
        out[h:, :] = np.swapaxes(fill(out[h - 1, :], out[0, :], H - h, True), 0, 1)
    return out


# --------------------------------------------------------------------------- #
# Phase-one building blocks
# --------------------------------------------------------------------------- #
def shock_filter(img, iters=5, dt=0.5, sigma=1.0):
    """Gaussian pre-smoothing followed by an upwind Osher-Rudin shock filter."""
    I = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img.copy()

    def minmod(a, b):
        return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.)

    for _ in range(iters):
        xp, xm = np.roll(I, -1, 1), np.roll(I, 1, 1)
        yp, ym = np.roll(I, -1, 0), np.roll(I, 1, 0)
        Ix, Iy = (xp - xm) / 2, (yp - ym) / 2
        Ixx, Iyy = xp - 2 * I + xm, yp - 2 * I + ym
        Ixy = (np.roll(Ix, -1, 0) - np.roll(Ix, 1, 0)) / 2
        Inn = Ix * Ix * Ixx + 2 * Ix * Iy * Ixy + Iy * Iy * Iyy
        mag = np.sqrt(minmod(xp - I, I - xm) ** 2 + minmod(yp - I, I - ym) ** 2)
        I = I - dt * np.sign(Inn) * mag
    return I


def r_map(gx, gy, h):
    """Eq. (1): ratio of summed gradient to summed gradient magnitude."""
    sx = cv2.boxFilter(gx, -1, (h, h), normalize=False, borderType=cv2.BORDER_REFLECT)
    sy = cv2.boxFilter(gy, -1, (h, h), normalize=False, borderType=cv2.BORDER_REFLECT)
    sm = cv2.boxFilter(np.hypot(gx, gy), -1, (h, h), normalize=False, borderType=cv2.BORDER_REFLECT)
    return np.hypot(sx, sy) / (sm + 0.5 / 255.)


def orientation_bins(gx, gy, n_bins=4):
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    return np.minimum((ang / (np.pi / n_bins)).astype(int), n_bins - 1)


def group_threshold(values, bins, valid, n_keep, n_bins=4):
    """Largest threshold that keeps at least ``n_keep`` pixels in every bin."""
    taus = []
    for g in range(n_bins):
        v = values[(bins == g) & valid]
        if v.size == 0:
            continue
        idx = int(min(max(n_keep, 1), v.size)) - 1
        taus.append(-np.partition(-v, idx)[idx])
    return min(taus) if taus else 0.


def kernel_system(FIx, FIy, FBx, FBy, shape, ksize):
    """Normal equations of ``sum_d ||d(Is) (*) k - d(B)||^2`` on a k-sized support.

    Returns the auto-correlation window ``R`` (so that ``A^T A k`` is the
    tiny convolution ``R (*) k``) and the right-hand side ``b = A^T d(B)``.
    Both need only one inverse FFT each, so the kernel solve itself is cheap.
    """
    auto = np.roll(irfft2(np.abs(FIx) ** 2 + np.abs(FIy) ** 2, shape), (ksize - 1, ksize - 1), axis=(0, 1))
    R = auto[:2 * ksize - 1, :2 * ksize - 1]
    b = otf2psf(np.conj(FIx) * FBx + np.conj(FIy) * FBy, shape, ksize)
    return R, b


def solve_kernel(R, b, x0, reg, iters=50, tol=1e-6):
    """Preconditioned CG for ``(R (*) k) + reg * k = b`` (``reg`` scalar or array)."""
    reg = np.broadcast_to(reg, b.shape)
    center = R.shape[0] // 2
    diag = R[center, center] + reg
    Aop = lambda x: signal.fftconvolve(R, x, mode='valid') + reg * x
    x = x0.copy()
    r = b - Aop(x)
    z = r / diag
    p = z.copy()
    rz = np.sum(r * z)
    bnorm = np.linalg.norm(b) + 1e-12
    for _ in range(iters):
        Ap = Aop(p)
        alpha = rz / (np.sum(p * Ap) + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) / bnorm < tol:
            break
        z = r / diag
        rz_new = np.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def clean_kernel(k, rel_thresh=0.05):
    """Project onto valid kernels: non-negative, small values cut, sum to one."""
    k = np.maximum(k, 0)
    if k.max() <= 0:
        k = np.zeros_like(k)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1
        return k
    k[k < rel_thresh * k.max()] = 0
    # drop tiny isolated blobs that are far from the main trajectory
    labels, n = ndimage.label(k > 0, structure=np.ones((3, 3)))
    if n > 1:
        mass = ndimage.sum(k, labels, index=np.arange(1, n + 1))
        keep = np.isin(labels, 1 + np.flatnonzero(mass >= 0.1 * mass.max()))
        k = k * keep
    return k / k.sum()


def center_kernel(k):
    """Integer-shift the kernel so its support is centred in the window.

    Centring the bounding box (rather than the centre of mass) leaves the
    largest free margin on every side for the kernel to grow into.
    """
    ys, xs = np.nonzero(k > 0)
    if ys.size == 0:
        return k
    c = k.shape[0] // 2
    sy = c - (ys.min() + ys.max()) // 2
    sx = c - (xs.min() + xs.max()) // 2
    return np.roll(k, (sy, sx), axis=(0, 1))


def resize_kernel(k, ksize):
    k = cv2.resize(k, (ksize, ksize), interpolation=cv2.INTER_LINEAR)
    k = np.maximum(k, 0)
    return k / k.sum()


def latent_update(FB, Fk, FIx, FIy, Dx, Dy, lam, shape):
    """Eq. (5): ``||I (*) k - B||^2 + lam ||grad I - grad Is||^2`` in closed form."""
    num = np.conj(Fk) * FB + lam * (np.conj(Dx) * FIx + np.conj(Dy) * FIy)
    den = np.abs(Fk) ** 2 + lam * (np.abs(Dx) ** 2 + np.abs(Dy) ** 2)
    return irfft2(num / den, shape)


# --------------------------------------------------------------------------- #
# Phase one: coarse-to-fine kernel estimation with informative edge selection
# --------------------------------------------------------------------------- #
def phase_one(img, ksize, n_levels=None, scale_step=np.sqrt(2), n_iters=20,
              gamma=2.0, lam=2e-3, shock_dt=0.5, shock_iters=5, r_window=5,
              kernel_thresh=0.05, callback=None):
    """Coarse-to-fine kernel estimation (Algorithm 1).

    Returns the finest-level kernel and the Fourier-domain data that phase two
    re-uses (blurred-image gradients plus gradient maps of the final latent).
    """
    # pyramid of kernel sizes, coarsest first
    ks_list = [ksize]
    while ks_list[-1] > 5 and (n_levels is None or len(ks_list) < n_levels):
        nxt = int(round(ks_list[-1] / scale_step))
        nxt += 1 - nxt % 2
        ks_list.append(max(nxt, 3))
    ks_list = ks_list[::-1]

    k = None
    L = None
    for level, ks in enumerate(ks_list):
        s = ks / ksize if level < len(ks_list) - 1 else 1.
        h, w = max(int(round(img.shape[0] * s)), ks), max(int(round(img.shape[1] * s)), ks)
        B = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if s < 1 else img
        shape = fast_shape(h, w, ks)
        Bp = wrap_boundary(B, shape)
        Bx, By = grad(Bp)
        FB, FBx, FBy = rfft2(Bp), rfft2(Bx), rfft2(By)
        Dx, Dy = diff_otfs(shape)

        # pixels whose neighbourhood lies fully inside the observed image
        valid = np.zeros(shape, bool)
        m = ks // 2 + 1
        valid[m:h - m, m:w - m] = True

        # edge confidence r is computed once per level from the blurred image
        r = r_map(Bx, By, r_window or ks)
        tau_r = group_threshold(r, orientation_bins(Bx, By), valid, 0.5 * np.sqrt(h * w) * ks)

        if k is None:
            k = np.zeros((ks, ks))
            k[ks // 2, ks // 2 - 1:ks // 2 + 2] = 1. / 3
            L = Bp.copy()
        else:
            k = resize_kernel(center_kernel(k), ks)
            L = wrap_boundary(cv2.resize(L, (w, h), interpolation=cv2.INTER_LINEAR), shape)

        for it in range(n_iters):
            Is = shock_filter(L, shock_iters, shock_dt)
            Ix, Iy = grad(Is)
            mag = np.hypot(Ix, Iy)
            M = (r >= tau_r) & valid
            if it == 0:
                tau_s = group_threshold(mag * M, orientation_bins(Ix, Iy), valid, 2 * ks)
            sel = M & (mag >= tau_s) & (mag > 0)
            Isx, Isy = Ix * sel, Iy * sel
            FIx, FIy = rfft2(Isx), rfft2(Isy)

            R, b = kernel_system(FIx, FIy, FBx, FBy, shape, ks)
            k = clean_kernel(solve_kernel(R, b, k, gamma), kernel_thresh)
            L = latent_update(FB, psf2otf(k, shape), FIx, FIy, Dx, Dy, lam, shape)

            tau_r /= 1.1
            tau_s /= 1.1
            if callback is not None:
                callback(level, it, k, L[:h, :w])
        L_full = L
        L = L[:h, :w]

    # gradient maps phase two can refine against (restricted to the interior)
    Lx, Ly = grad(L_full)
    Sx, Sy = grad(shock_filter(L_full, shock_iters, shock_dt))
    data = dict(FBx=FBx, FBy=FBy, shape=shape, edges=(FIx, FIy),
                shock=(rfft2(Sx * valid), rfft2(Sy * valid)),
                latent=(rfft2(Lx * valid), rfft2(Ly * valid)))
    return k, data


# --------------------------------------------------------------------------- #
# Phase two: ISD-based kernel refinement
# --------------------------------------------------------------------------- #
def isd_support(k, i, ksize):
    """Support detection rule of Sec. 3: first big jump in the sorted values."""
    v = np.sort(k.ravel())
    d = np.diff(v)
    jumps = np.flatnonzero(d > v[-1] / (2 * ksize * i))
    eps = v[jumps[0]] if jumps.size else v[-2]
    return k > eps


def phase_two(k0, data, gamma=2e-3, max_iters=10, tol=1e-3, source='latent'):
    """Iterative support detection with IRLS-weighted l1 outside the support.

    ``source`` picks the sharp gradient map the kernel is fitted against:
    ``'edges'`` (the paper's selected edges), ``'shock'`` (all shock-filtered
    gradients) or ``'latent'`` (gradients of the phase-one latent image, the
    densest constraint set and the most accurate in our tests).
    """
    FIx, FIy = data[source]
    ksize = k0.shape[0]
    R, b = kernel_system(FIx, FIy, data['FBx'], data['FBy'], data['shape'], ksize)
    # scale the sparsity weight to the data term so it is image independent
    gamma = gamma * R[ksize - 1, ksize - 1]
    k = k0.copy()
    for i in range(1, max_iters + 1):
        S = isd_support(k, i, ksize)
        # IRLS: |k_j| ~= k_j^2 / max(|k_j^old|, 1e-5) for j outside the support
        w = np.where(S, 0., gamma / np.maximum(np.abs(k), 1e-5))
        k_new = clean_kernel(solve_kernel(R, b, k, 1e-3 * gamma + w, iters=100), rel_thresh=0.02)
        change = np.linalg.norm(k_new - k) / np.linalg.norm(k)
        k = k_new
        if change <= tol:
            break
    return k


# --------------------------------------------------------------------------- #
# Final non-blind deconvolution: fast TV-l1 (half-quadratic splitting)
# --------------------------------------------------------------------------- #
def deconv_tvl1(B, k, lam=2e-2, beta0=1., beta_min=1e-2, theta_min=1e-2, step=2., inner=2):
    """Minimise ``||I (*) k - B||_1 + lam ||grad I||_1``.

    Auxiliary ``v`` (residual) and ``w`` (gradients) are updated by shrinkage
    and ``I`` in closed form. Pixels of the padding canvas carry no data term
    (their residual is absorbed by ``v``), so the unknown content outside the
    frame does not force ringing into the result.
    """
    squeeze = B.ndim == 2
    if squeeze:
        B = B[..., None]
    h, w, c = B.shape
    ks = k.shape[0]
    shape = fast_shape(h, w, ks)
    Bp = wrap_boundary(B, shape)
    mask = np.zeros(shape + (1,), bool)
    mask[:h, :w] = True

    Fk = psf2otf(k, shape)[..., None]
    Dx, Dy = diff_otfs(shape)
    Dx, Dy = Dx[..., None], Dy[..., None]
    KtK = np.abs(Fk) ** 2
    DtD = np.abs(Dx) ** 2 + np.abs(Dy) ** 2

    I = Bp.copy()
    beta, theta = beta0, 1. / lam
    while True:
        for _ in range(inner):
            res = irfft2(Fk * rfft2(I), shape) - Bp
            v = np.where(mask, np.sign(res) * np.maximum(np.abs(res) - beta, 0), res)
            gx = np.roll(I, -1, axis=1) - I
            gy = np.roll(I, -1, axis=0) - I
            mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
            shrink = np.maximum(mag - lam * theta, 0) / mag
            wx, wy = gx * shrink, gy * shrink
            ratio = beta / theta
            num = np.conj(Fk) * rfft2(Bp + v) + ratio * (np.conj(Dx) * rfft2(wx) + np.conj(Dy) * rfft2(wy))
            I = irfft2(num / (KtK + ratio * DtD), shape)
        if beta <= beta_min and theta <= theta_min:
            break
        beta = max(beta / step, beta_min)
        theta = max(theta / step, theta_min)
    I = np.clip(I[:h, :w], 0, 1)
    return I[..., 0] if squeeze else I


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def deblur(img_bgr, ksize, callback=None, phase1=None, phase2=True, isd=None, deconv=None):
    """Blind deblurring of a BGR (or gray) float image in [0, 1].

    The kernel is estimated on luminance; the final TV-l1 deconvolution is run
    on every channel. ``phase1``/``isd``/``deconv`` are keyword overrides for
    :func:`phase_one`, :func:`phase_two` and :func:`deconv_tvl1`.
    Returns ``(latent, kernel, phase_one_kernel)``.
    """
    gray = cv2.cvtColor(img_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64) \
        if img_bgr.ndim == 3 else img_bgr.astype(np.float64)
    k, data = phase_one(gray, ksize, callback=callback, **(phase1 or {}))
    k1 = k
    if phase2:
        k = phase_two(k, data, **(isd or {}))
    # centre the kernel so the restored image stays aligned with the input
    k = center_kernel(k)
    latent = deconv_tvl1(img_bgr.astype(np.float64), k, **(deconv or {}))
    return latent, k, k1


def kernel_to_image(k, size=None):
    vis = k / k.max()
    if size:
        vis = cv2.resize(vis, (size, size), interpolation=cv2.INTER_NEAREST)
    return (vis * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--input', default='toy.jpg')
    ap.add_argument('--output', default='restored.png')
    ap.add_argument('--kernel-output', default='kernel.png')
    ap.add_argument('--ksize', type=int, default=95,
                    help='kernel size at the processed scale; a bit larger than the blur extent works best')
    ap.add_argument('--scale', type=float, default=1.0, help='resize input before processing')
    ap.add_argument('--gray', action='store_true', help='restore only the luminance channel')
    ap.add_argument('--show', action='store_true', help='show progress windows (needs a display)')
    args = ap.parse_args()

    img = cv2.imread(args.input).astype(np.float32) / 255.
    if args.scale != 1:
        img = cv2.resize(img, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    if args.gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = args.ksize + 1 - args.ksize % 2

    def show_progress(level, it, k, L):
        cv2.imshow('latent', np.clip(L, 0, 1))
        cv2.imshow('kernel', kernel_to_image(k, 200))
        if cv2.waitKey(1) == 27:
            raise SystemExit
    callback = show_progress if args.show else None

    t0 = time.time()
    latent, k, _ = deblur(img, ksize, callback=callback)
    print(f'done in {time.time() - t0:.1f}s')
    cv2.imwrite(args.output, (latent * 255).round().astype(np.uint8))
    cv2.imwrite(args.kernel_output, kernel_to_image(k, 4 * ksize))
    if args.show:
        cv2.imshow('restored', latent)
        cv2.waitKey()


if __name__ == '__main__':
    main()
