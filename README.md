# Two-Phase Kernel Estimation for Robust Motion Deblurring

An unofficial Python/NumPy implementation of [Two-Phase Kernel Estimation for Robust Motion Deblurring](https://doi.org/10.1007/978-3-642-15549-9_12) (Xu & Jia, ECCV 2010) — blind motion deblurring of a single image, written from scratch with OpenCV/SciPy (no deep learning).

## What's Implemented

**Phase 1 — coarse-to-fine kernel estimation** (4-level image pyramid):

- Edge prediction by shock filtering the current latent estimate
- Gradient-confidence map `r` and informative-edge selection with per-orientation thresholds (`tau_r`, `tau_s`), gradually relaxed (`/1.1`) each iteration so more edges join as the kernel improves
- Closed-form FFT kernel estimation from selected edge gradients with Tikhonov regularization, projected back to a valid kernel (non-negative, sum to 1)
- FFT latent-image update with a spatial gradient prior

**Phase 2 — kernel refinement and final deconvolution**:

- ISD-style kernel refinement: iterative support detection that re-solves the kernel least-squares with adaptive sparsity regularization outside the detected support
- Fast TV-ℓ1 deconvolution via half-quadratic splitting to recover the final latent image

## Run

```bash
pip install opencv-python numpy scipy scikit-image
python main.py
```

Runs on the bundled `toy.jpg` (downscaled 2x, kernel size 47) and shows the evolving kernel and latent image; press `Esc` to abort. Pure NumPy/FFT — expect a few minutes.

## Results

| Blurred input | Restored latent image |
|---|---|
| ![Original Image](toy.jpg) | ![Latent Image](restored.png) |

## Known Issues

- Some ringing and border artifacts remain in the estimated latent image; it is unclear how the authors handle boundary conditions.
- The inverse FFT of the kernel solve can yield negative (even complex) values that do not sum to 1. This implementation simply projects the kernel onto the valid solution manifold; a Lagrange-multiplier formulation might be cleaner.

## Reference

```bibtex
@inproceedings{xu2010two,
  title={Two-Phase Kernel Estimation for Robust Motion Deblurring},
  author={Xu, Li and Jia, Jiaya},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={157--170},
  year={2010},
  organization={Springer}
}
```
