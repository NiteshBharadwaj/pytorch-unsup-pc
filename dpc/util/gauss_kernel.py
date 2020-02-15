import numpy as np
import torch


def gauss_kernel_1d(l, sig):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    xx = torch.arange(-l // 2 + 1., l // 2 + 1)
    kernel = torch.exp(-xx**2 / (2. * sig**2))
    return kernel / kernel.sum()


# def gauss_smoothen_image(cfg, img, sigma_rel):
#     fsz = cfg.pc_gauss_kernel_size
#     kernel = gauss_kernel_1d(fsz, sigma_rel)
#     in_channels = img.shape[-1]
#     k1 = torch.repeat(kernel.reshape(1, fsz, 1, 1), (1, 1, in_channels, 1))
#     k2 = torch.repeat(kernel.reshape((fsz, 1, 1, 1)), (1, 1, in_channels, 1))
#
#     img_tmp = img
#     img_tmp = tf.nn.depthwise_conv2d(img_tmp, k1, [1, 1, 1, 1], padding="SAME")
#     img_tmp = tf.nn.depthwise_conv2d(img_tmp, k2, [1, 1, 1, 1], padding="SAME")
#     return img_tmp


def separable_kernels(kernel):
    size = kernel.shape[0]
    k1 = kernel.reshape((1, 1, size, 1, 1))
    k2 = kernel.reshape((1, size, 1, 1, 1))
    k3 = kernel.reshape((size, 1, 1, 1, 1))
    return [k1, k2, k3]


def smoothing_kernel(cfg, sigma):
    fsz = cfg.pc_gauss_kernel_size
    kernel_1d = gauss_kernel_1d(fsz, sigma)
    if cfg.vox_size_z != -1:
        vox_size_z = cfg.vox_size_z
        vox_size = cfg.vox_size
        ratio = vox_size_z / vox_size
        sigma_z = sigma * ratio
        fsz_z = int(np.floor(fsz * ratio))
        if fsz_z % 2 == 0:
            fsz_z += 1
        kernel_1d_z = gauss_kernel_1d(fsz_z, sigma_z)
        k1 = kernel_1d.reshape((1, 1, fsz, 1, 1))
        k2 = kernel_1d.reshape((1, fsz, 1, 1, 1))
        k3 = kernel_1d_z.reshape((fsz_z, 1, 1, 1, 1))
        kernel = [k1, k2, k3]
    else:
        if cfg.pc_separable_gauss_filter:
            kernel = separable_kernels(kernel_1d)
    return kernel
