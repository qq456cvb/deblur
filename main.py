import cv2
import numpy as np
from scipy import signal
from skimage.util import view_as_windows as viewW


# https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1
    
    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1], -1)[:,::stepsize]


def shock_filter(img):
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    it = 100
    dt = 1e-3
    for i in range(it):
        Ix = signal.convolve2d(img, np.array([[-1, 0, 1]]) / 2, mode='same')
        Iy = signal.convolve2d(img, np.array([[-1], [0], [1]]) / 2, mode='same')
        Ixx = signal.convolve2d(Ix, np.array([[-1, 0, 1]]) / 2, mode='same')
        Ixy = signal.convolve2d(Iy, np.array([[-1, 0, 1]]) / 2, mode='same')
        Iyy = signal.convolve2d(Iy, np.array([[-1], [0], [1]]) / 2, mode='same')
        img -= dt * np.sign(Ix * Ix * Ixx + 2 * Ix * Iy * Ixy + Iy * Iy * Iyy) * np.linalg.norm(np.stack([Ix, Iy], -1), axis=-1)
    # cv2.imshow('shock_filtered', img)
    # cv2.waitKey(0)
    return img


# assume channel in the last
def img_double(img):
    res = np.zeros((img.shape[0] * 2, img.shape[1] * 2, *img.shape[2:]), img.dtype)
    res[:img.shape[0], :img.shape[1]] = img
    return res


if __name__ == '__main__':
    img = cv2.imread('toy.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    
    # image too large reduce size
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Phase One
    n_pyramids = 4
    pyramids = [img]
    ksizes = [47]
    for i in range(n_pyramids - 1):
        pyramids.append(cv2.pyrDown(pyramids[-1]))
        ksizes.append(ksizes[-1] / 2)
    
    pyramids = list(reversed(pyramids))
    ksizes = list(reversed(ksizes))
    for i, ksize in enumerate(ksizes):
        ksizes[i] = int(ksize)
        if ksizes[i] % 2 == 0:
            ksizes[i] += 1

    print(ksizes)
    
    L = pyramids[0]
    for i, B in enumerate(pyramids[:-1]):
        
        # FIXME: border should not contribute to the final optimization
        ksize = ksizes[i]
        half_ksize = ksize // 2
        Bx = signal.convolve2d(B, np.array([[-1, 0, 1]]) / 2, mode='same')
        By = signal.convolve2d(B, np.array([[-1], [0], [1]]) / 2, mode='same')
        convx = signal.convolve2d(Bx, np.ones((ksize, ksize)), mode='same')
        convy = signal.convolve2d(By, np.ones((ksize, ksize)), mode='same')
        r = np.linalg.norm(np.stack([convx, convy], -1), axis=-1) / \
            (signal.convolve2d(np.linalg.norm(np.stack([Bx, By], -1), axis=-1), np.ones((ksize, ksize)),
                            mode='same') + 0.5 / 255.)
        # cv2.imshow('r map', r)
        # cv2.waitKey(30)

        grad_angle = np.arctan2(By, Bx)
        grad_angle[grad_angle < 0] += np.pi
        angle_masks = np.stack([
            grad_angle < np.pi / 4,
            (grad_angle > np.pi / 4) |  (grad_angle < np.pi / 2),
            (grad_angle > np.pi / 2) |  (grad_angle < 3 * np.pi / 4),
            grad_angle > 3 * np.pi / 4
        ])
        angle_masks[:, 0] = angle_masks[:, -1] = angle_masks[:, :, 0] = angle_masks[:, :, -1] = False
        # print(np.sum(angle_masks[0]), np.sum(angle_masks[1]), np.sum(angle_masks[2]), np.sum(angle_masks[3]))
        # import pdb; pdb.set_trace()
        for j in range(20):
            I_tilde = shock_filter(L)
            I_tilde_grad = np.stack([signal.convolve2d(I_tilde, np.array([[-1, 0, 1]]) / 2, mode='same'),
                signal.convolve2d(I_tilde, np.array([[-1], [0], [1]]) / 2, mode='same')], -1)
            # I_tilde_grad[0] = I_tilde_grad[-1] = I_tilde_grad[:, 0] = I_tilde_grad[:, -1] = 0

            if j == 0:
                tau_r_num = int(0.5 * np.sqrt(grad_angle.shape[0] * grad_angle.shape[1]) * ksize)
                tau_s_num = 2 * ksize
                tau_r = np.min([-np.partition(-r[mask], tau_r_num)[tau_r_num] for mask in angle_masks])
                tau_s = max([np.min([-np.partition(-I_tilde_grad[mask][..., k], tau_s_num)[tau_s_num] for mask in angle_masks]) for k in range(2)])

            I_grad_s = I_tilde_grad * np.heaviside(np.heaviside(r - tau_r, 0) * np.linalg.norm(I_tilde_grad, axis=-1) - tau_s, 0)[..., None]
            
            # cv2.imshow('selected edge', np.linalg.norm(I_tilde_grad, axis=-1))
            # if cv2.waitKey() == 27:
            #     exit()
            
            # kernel estimation
            fftgradb = np.fft.fft2(img_double(np.stack([Bx, By], -1)), axes=[0, 1])
            fftedges = np.fft.fft2(img_double(I_grad_s), axes=[0, 1])
            
            fftk = (np.conjugate(fftedges[..., 0]) * fftgradb[..., 0] + np.conjugate(fftedges[..., 1]) * fftgradb[..., 1]) \
                             / (np.square(np.absolute(fftedges[..., 0])) + np.square(np.absolute(fftedges[..., 1])) + 10.)
            k = np.real(np.fft.ifft2(fftk))
            kernel = np.zeros_like(k)
            kernel[:half_ksize+1, :half_ksize+1] = k[:half_ksize+1, :half_ksize+1]
            kernel[-half_ksize:, -half_ksize:] = k[-half_ksize:, -half_ksize:]
            kernel[kernel < 0] = 0
            kernel /= kernel.sum()
            # print(kernel[:half_ksize+1, :half_ksize+1])
            # print(kernel[-half_ksize:, -half_ksize:])
            # cv2.imshow('kernel estimation', kernel)
            # cv2.waitKey()
            scharr_x, scharr_y = np.zeros_like(kernel), np.zeros_like(kernel)

            scharr_x[0, 1] = 1
            scharr_x[0, -1] = -1

            scharr_y[1, 0] = 1
            scharr_y[-1, 0] = -1

            scharr_x /= 2.
            scharr_y /= 2.

            fftdx = np.fft.fft2(scharr_x)
            fftdy = np.fft.fft2(scharr_y)
            
            fftk = np.fft.fft2(kernel)
            latent = np.real(np.fft.ifft2((np.conjugate(fftk) * np.fft.fft2(img_double(B)) + 2e-3 * (np.conjugate(fftdx) * fftedges[..., 0] + np.conjugate(fftdy) * fftedges[..., 1]))
                                  / (np.conjugate(fftk) * fftk + 2e-3 * (np.conjugate(fftdx) * fftdx + np.conjugate(fftdy) * fftdy))))
            L = np.clip(latent[:latent.shape[0] // 2, :latent.shape[1] // 2], 0, 1)
            cv2.imshow('latent image estimation', L)
            if cv2.waitKey(30) == 27:
                exit()
            tau_r /= 1.1
            tau_s /= 1.1
            
        if i < len(pyramids) - 1:
            L = cv2.resize(L, (pyramids[i+1].shape[1], pyramids[i+1].shape[0]))
        vis_kernel = np.zeros((ksize, ksize))
        vis_kernel[:half_ksize, :half_ksize] = kernel[-half_ksize:, -half_ksize:]
        vis_kernel[half_ksize:, half_ksize:] = kernel[:half_ksize+1, :half_ksize+1]
        cv2.imshow('kernel', vis_kernel / vis_kernel.max())
        cv2.waitKey(30)
    

    # Phase Two
    k0 = vis_kernel
    ksize = k0.shape[0]
    half_ksize = k0.shape[0] // 2
    def select_S(kernel, i, ksize):
        kernel_sorted = np.sort(kernel)
        largest = kernel_sorted[-1]
        diff = kernel_sorted[1:] - kernel_sorted[:-1]
        thresh = kernel_sorted[-2]
        for j in range(diff.size):
            if diff[j] > largest / (2 * ksize * (i + 1)):
                thresh = kernel_sorted[j]
                break
        return kernel > thresh

    A = np.concatenate([im2col_sliding_strided(I_tilde_grad[..., 0], (k0.shape[0], k0.shape[1])).T, 
        im2col_sliding_strided(I_tilde_grad[..., 1], (k0.shape[0], k0.shape[1])).T], 0)
    VB = np.concatenate([Bx[half_ksize:-half_ksize, half_ksize:-half_ksize].reshape(-1), By[half_ksize:-half_ksize, half_ksize:-half_ksize].reshape(-1)], 0)

    Vk = k0.reshape(-1)
    S = select_S(Vk, 0, ksize)
    while True:
        VS_bar = np.ones_like(Vk, dtype=np.bool)
        VS_bar[S] = False
        psi = max(np.sum(np.abs(Vk)), 1e-5)
        Vk_new = np.linalg.lstsq(A.T @ A + 1. * np.diag(VS_bar / psi).astype(np.float), A.T @ VB)[0]
        Vk_new[Vk_new < 0] = 0
        Vk_new /= Vk_new.sum()
        S = select_S(Vk_new, 0, ksize)
        if np.linalg.norm(Vk_new - Vk) / np.linalg.norm(Vk) <= 1e-3:
            break
        else:
            Vk = Vk_new
    ks = Vk.reshape(ksize, ksize)
    cv2.imshow('kernel s', ks / ks.max())
    cv2.waitKey(30)
    
    # Fast TV-l1 Deconvolution