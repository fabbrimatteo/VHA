# -*- coding: utf-8 -*-
# ---------------------

import matplotlib


matplotlib.use('Agg')

import PIL
import torch
import numpy as np
from typing import *
from path import Path
from torch import Tensor
from matplotlib import cm
from PIL.Image import Image
from matplotlib import figure
from torchvision.transforms import ToTensor, RandomHorizontalFlip
from torch.nn.functional import interpolate
from torch import nn

RandomHorizontalFlip = RandomHorizontalFlip


def imread(path):
    # type: (Union[Path, str]) -> Image
    """
    :return: PIL image at the required path (RGB[0,255] format)
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(figure):
    # type: (figure.Figure) -> np.ndarray
    """
    :return: np.array version of the input pyplot figure
    """
    figure.canvas.draw()
    x = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(figure):
    # type: (figure.Figure) -> torch.Tensor
    """
    :param figure: torch.Tensor version of the input pyplot figure
    :return:
    """
    x = pyplot_to_numpy(figure=figure)
    x = ToTensor()(x)
    return x


def apply_colormap_to_tensor(x, cmap='jet', range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=range[0], vmax=range[1])
    try:
        x = x.cpu().numpy()
    except:
        x = x.detatch().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return ToTensor()(x)


def visualize_3d_hmap(hmap):
    # type: (Union[np.ndarray, torch.Tensor]) -> None
    """
    Interactive visualization of 3D heatmaps.
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    """
    import cv2

    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()

    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    for d, x in enumerate(hmap):
        x = cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET)
        x = cv2.putText(x, f'{d}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 128), 2, cv2.LINE_AA)
        cv2.imshow(f'press ESC to advance in the depth dimension', x)
        cv2.waitKey()
    cv2.destroyAllWindows()


def visualize_multiple_3d_hmap(hmaps):
    # type: (List[torch.Tensor]) -> None
    """
    Interactive visualization of 3D heatmaps.
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    """
    import cv2

    hmap_gt = hmaps[0].cpu().numpy()
    hmap_pr = hmaps[1].cpu().numpy()

    hmap = np.concatenate((hmap_gt, hmap_pr), axis=2)
    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    for d, x in enumerate(hmap):
        x = cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET)
        x = cv2.putText(x, f'{d}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 128), 2, cv2.LINE_AA)
        cv2.imshow(f'press ESC to advance in the depth dimension', x)
        cv2.waitKey()
    cv2.destroyAllWindows()



def save_3d_hmap(hmap, path):
    # type: (Union[np.ndarray, torch.Tensor], str) -> None
    """
    Saves a 3D heatmap as MP4 video with JET colormap.
    :param hmap: 3D heatmap with values in [0,1] and shape (D, H, W)
    :param path: desired path for the output video
    """
    import cv2
    import imageio

    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()
    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    frames = [cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET) for x in hmap]

    for d, x in enumerate(frames):
        frames[d] = cv2.putText(frames[d], f'{d}', (10, 60), 1, 2, (255, 128, 128), 2, cv2.LINE_AA)[:, :, ::-1]

    imageio.mimsave(path, frames, macro_block_size=None)



def gkern(d, h, w, center, s=2, device='cuda'):
    # type: (int, int, int, Union[List[int], Tuple[int, int, int]], float) -> torch.Tensor
    """
    :param d: hmap depth
    :param h: hmap height
    :param w: hmap width
    :param center: center of the Gaussian | ORDER: (x, y, z)
    :param s: sigma of the Gaussian
    :return: heatmap (shape torch.Size([d, h, w])) with a gaussian centered in `center`
    """
    x = torch.arange(0, w, 1).float().to(device)
    y = torch.arange(0, h, 1).float().to(device)
    y = y.unsqueeze(1)
    z = torch.arange(0, d, 1).float().to(device)
    z = z.unsqueeze(1).unsqueeze(1)

    x0 = center[0]
    y0 = center[1]
    z0 = center[2]

    return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / s ** 2)


def torch_to_list(y):
    # type: (torch.Tensor) -> List[List[int]]
    if len(y.shape) == 3:
        y = y[0]
    return [list(point3d.cpu().numpy()) for point3d in y]


def get_local_maxima_3d(hmap3d, threshold, device='cuda'):
    # type: (torch.Tensor, float, str) -> torch.Tensor
    """
    :param hmap3d: 3D heatmap with shape (D, H, W)
    :param threshold: peaks with values < of that threshold will not be returned
    :param device: device where you want to run the operation
    :return: torch Tensor with 3D coordinates of the gaussian peaks (local maxima)
        * tensor shape: (#peaks, D, H, W)
    """
    d = torch.device(device)

    m_f = torch.zeros(hmap3d.shape).to(d)
    m_f[1:, :, :] = hmap3d[:-1, :, :]
    m_b = torch.zeros(hmap3d.shape).to(d)
    m_b[:-1, :, :] = hmap3d[1:, :, :]

    m_u = torch.zeros(hmap3d.shape).to(d)
    m_u[:, 1:, :] = hmap3d[:, :-1, :]
    m_d = torch.zeros(hmap3d.shape).to(d)
    m_d[:, :-1, :] = hmap3d[:, 1:, :]

    m_r = torch.zeros(hmap3d.shape).to(d)
    m_r[:, :, 1:] = hmap3d[:, :, :-1]
    m_l = torch.zeros(hmap3d.shape).to(d)
    m_l[:, :, :-1] = hmap3d[:, :, 1:]

    p = torch.zeros(hmap3d.shape).to(d)
    p[hmap3d >= m_f] = 1
    p[hmap3d >= m_b] += 1
    p[hmap3d >= m_u] += 1
    p[hmap3d >= m_d] += 1
    p[hmap3d >= m_r] += 1
    p[hmap3d >= m_l] += 1

    p[hmap3d >= threshold] += 1
    p[p != 7] = 0

    return torch.tensor(torch.nonzero(p).cpu())


def get_multi_local_maxima_3d(hmaps3d, threshold, device='cuda'):
    # type: (torch.Tensor, float, str) -> List[Tuple[int, int, int, int]]
    """
    :param hmaps3d: 3D heatmaps with shape (N_joints, D, H, W)
    :param threshold: peaks with values < of that threshold will not be returned
    :param device: device where you want to run the operation
    :return: ...
    """
    d = torch.device(device)

    peaks = []

    for jtype, hmap3d in enumerate(hmaps3d):
        m_f = torch.zeros(hmap3d.shape).to(d)
        m_f[1:, :, :] = hmap3d[:-1, :, :]
        m_b = torch.zeros(hmap3d.shape).to(d)
        m_b[:-1, :, :] = hmap3d[1:, :, :]

        m_u = torch.zeros(hmap3d.shape).to(d)
        m_u[:, 1:, :] = hmap3d[:, :-1, :]
        m_d = torch.zeros(hmap3d.shape).to(d)
        m_d[:, :-1, :] = hmap3d[:, 1:, :]

        m_r = torch.zeros(hmap3d.shape).to(d)
        m_r[:, :, 1:] = hmap3d[:, :, :-1]
        m_l = torch.zeros(hmap3d.shape).to(d)
        m_l[:, :, :-1] = hmap3d[:, :, 1:]

        p = torch.zeros(hmap3d.shape).to(d)
        p[hmap3d >= m_f] = 1
        p[hmap3d >= m_b] += 1
        p[hmap3d >= m_u] += 1
        p[hmap3d >= m_d] += 1
        p[hmap3d >= m_r] += 1
        p[hmap3d >= m_l] += 1

        p[hmap3d >= threshold] += 1
        p[p != 7] = 0

        tmp = torch.tensor(torch.nonzero(p).cpu())
        tmp = [[jtype, z, y, x] for z, y, x in torch_to_list(tmp)]
        peaks += tmp

    return peaks


def argmax(hmaps3d):
    # type: (torch.Tensor) -> Tuple[List[List[int]], List[float]]
    """
    :param hmaps3d: 3D heatmaps with shape (N_joints, D, H, W)
    :return: ...
    """
    peaks = []
    confidences = []
    for jtype, hmap3d in enumerate(hmaps3d):
        confidence = hmap3d.max()
        tmp = list((hmap3d == confidence).nonzero()[0].cpu().numpy())
        tmp = [jtype] + tmp
        peaks += [tmp]
        confidences.append(confidence.item())

    return peaks, confidences
