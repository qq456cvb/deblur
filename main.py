import cv2
import numpy as np
from scipy import signal


LAMBDA = 2e-3


def shock_filter(img):
    img = cv2.blur(img, ksize=(3, 3))
    it = 10
    dt = 0.01
    for i in range(it):
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
        Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)
        Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
        img -= dt * np.sign(Ix * Ix * Ixx + 2 * Ix * Iy * Ixy + Iy * Iy * Iyy) * np.linalg.norm(np.stack([Ix, Iy], -1), axis=-1)
    cv2.imshow('shock_filtered', img)
    cv2.waitKey(30)
    return img


# The algorithm is not clear stated in the paper, neither the parameters, try to implement, but failed
if __name__ == '__main__':
    ksize = 25
    image = cv2.imread('lion.gif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.

    n_pyramids = 1
    pyramids = [image]
    ksizes = [ksize]
    for i in range(n_pyramids - 1):
        pyramids.append(cv2.pyrDown(pyramids[-1]))
        ksizes.append(ksize // 2)
        if ksizes[-1] % 2 == 0:
            ksizes[-1] += 1
        # cv2.imshow('test', pyramids[-1])
        # cv2.waitKey()

    tau_r = 0.1
    tau_s = 0.05
    cv2.imshow('raw', image)
    cv2.waitKey(30)
    for i in reversed(range(n_pyramids)):
        img = pyramids[i]
        # Bx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Bx = signal.convolve2d(img, np.array([[-1, 0, 1]]) / 2, mode='same')
        # By = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        By = signal.convolve2d(img, np.array([[-1], [0], [1]]) / 2, mode='same')
        convx = signal.convolve2d(Bx, np.ones((ksizes[i], ksizes[i])), mode='same')
        convy = signal.convolve2d(By, np.ones((ksizes[i], ksizes[i])), mode='same')
        r = np.linalg.norm(np.stack([convx, convy], -1), axis=-1) / \
            (signal.convolve2d(np.linalg.norm(np.stack([Bx, By], -1), axis=-1), np.ones((ksizes[i], ksizes[i])),
                               mode='same') + 0.5 / 255.)
        cv2.imshow('r map', r)
        cv2.waitKey(30)
        I = img
        B = img
        for it in range(10):

            # select edges
            # Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
            Ix = signal.convolve2d(I, np.array([[-1, 0, 1]]) / 2, mode='same')
            # Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
            Iy = signal.convolve2d(I, np.array([[-1], [0], [1]]) / 2, mode='same')

            shocked = shock_filter(I)
            # Ix_shocked = cv2.Sobel(shocked, cv2.CV_64F, 1, 0, ksize=3)
            Ix_shocked = signal.convolve2d(shocked, np.array([[-1, 0, 1]]) / 2, mode='same')
            # Iy_shocked = cv2.Sobel(shocked, cv2.CV_64F, 0, 1, ksize=3)
            Iy_shocked = signal.convolve2d(shocked, np.array([[-1], [0], [1]]) / 2, mode='same')
            # Is = shocked * np.heaviside(np.heaviside(r - tau_r, 0) * np.linalg.norm(np.stack([Ix_shocked, Iy_shocked], -1), axis=-1) - tau_s, 0)
            edges = np.stack([Ix_shocked, Iy_shocked], -1) * np.heaviside(np.heaviside(r - tau_r, 0) * np.linalg.norm(np.stack([Ix_shocked, Iy_shocked], -1), axis=-1) - tau_s, 0)[..., None]
            # edges = np.stack([cv2.Sobel(Is, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(Is, cv2.CV_64F, 0, 1, ksize=3)], -1)
            cv2.imshow('selected edges', edges[..., 0] / edges[..., 0].max())
            cv2.waitKey(30)
            fftgradb = np.fft.fft2(np.stack([Bx, By], -1), axes=[0, 1])
            fftedges = np.fft.fft2(edges, axes=[0, 1])
            # print(Ix.shape, fftedges.shape)
            # TODO: actually this is circular convolution, should pad original image to represent normal convolution
            fftk = (np.conjugate(fftedges[:, :, 0]) * fftgradb[:, :, 0] + np.conjugate(fftedges[:, :, 1]) * fftgradb[:, :, 1]) \
                             / (np.square(np.absolute(fftedges[:, :, 0])) + np.square(np.absolute(fftedges[:, :, 1])))
            k = np.real(np.fft.ifft2(fftk))
            k[ksizes[i] // 2 + 1:-ksizes[i] // 2, ksizes[i] // 2 + 1:-ksizes[i] // 2] = 0
            # how to handle sparse kernels???
            # k[k < 0] = 0
            k /= k.sum()
            fftk = np.fft.fft2(k)
            # print(k.sum())
            cv2.imshow('kernel', k / k.max())
            cv2.waitKey(30)
            # Scharr operator, due to opencv doc, when kernel size is 3, solbel is scharr
            scharr_x, scharr_y = np.zeros_like(k), np.zeros_like(k)

            # schar x
            # scharr_x[:2, :2] = np.array([[0, 1], [0, 1]])
            # scharr_x[-1, -1] = -1
            # scharr_x[-1, :2] = np.array([0, 1])
            # scharr_x[:2, -1] = np.array([-1, -1])
            #
            # scharr_y[:2, :2] = np.array([[0, 0], [1, 1]])
            # scharr_y[-1, -1] = -1
            # scharr_y[-1, :2] = np.array([-1, -1])
            # scharr_y[:2, -1] = np.array([0, 1])

            scharr_x[0, 1] = 1
            scharr_x[0, -1] = -1

            scharr_y[1, 0] = 1
            scharr_y[-1, 0] = -1

            scharr_x /= 2.
            scharr_y /= 2.

            fftdx = np.fft.fft2(scharr_x)
            # cv2.imshow('fftdx', fftdx.astype(np.float32) / fftdx.astype(np.float32).max())
            # cv2.waitKey()
            fftdy = np.fft.fft2(scharr_y)

            # artifact induced, don't know why
            latent = np.real(np.fft.ifft2((np.conjugate(fftk) * np.fft.fft2(B) + LAMBDA * (np.conjugate(fftdx) * np.fft.fft2(edges[:, :, 0]) + np.conjugate(fftdy) * np.fft.fft2(edges[:, :, 1])))
                                  / (np.conjugate(fftk) * fftk + LAMBDA * (np.conjugate(fftdx) * fftdx + np.conjugate(fftdy) * fftdy))))
            latent = np.clip(latent, 0, 1)
            cv2.imshow('latent%d' % it, latent)
            cv2.waitKey()
            tau_s /= 1.1
            tau_r /= 1.1
            I = latent
