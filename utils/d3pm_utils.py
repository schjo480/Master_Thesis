import torch
import numpy as np
import PIL.Image
from torch.nn import functional as F
from functools import partial

RGB_TO_YUV = np.array([[0.29900, -0.16874, 0.50000],
                       [0.58700, -0.33126, -0.41869],
                       [0.11400, 0.50000, -0.08131]])

def normalize_data(x, mode=None):
    x = torch.tensor(x, dtype=torch.float32)  # Ensure input is a PyTorch tensor
    if mode is None or mode == 'rgb':
        return x / 127.5 - 1.
    elif mode == 'rgb_unit_var':
        return 2. * normalize_data(x, mode='rgb')
    elif mode == 'yuv':
        x = (x / 127.5 - 1.).mm(torch.tensor(RGB_TO_YUV).T)
        return x
    else:
        raise NotImplementedError(mode)

def log_min_exp(a, b, epsilon=1.e-6):
    y = a + torch.log1p(-torch.exp(b - a) + epsilon)
    return y

def sample_categorical(logits, uniform_noise):
    logits = torch.tensor(logits, dtype=torch.float32)
    uniform_noise = torch.tensor(uniform_noise, dtype=torch.float32)
    uniform_noise = torch.clamp(uniform_noise, min=torch.finfo(uniform_noise.dtype).tiny, max=1.)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    sample = torch.argmax(logits + gumbel_noise, dim=-1)
    return F.one_hot(sample, num_classes=logits.shape[-1])

def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    logits1, logits2 = logits1, logits2
    probs1 = F.softmax(logits1 + eps, dim=-1)
    log_probs1 = F.log_softmax(logits1 + eps, dim=-1)
    log_probs2 = F.log_softmax(logits2 + eps, dim=-1)
    kl = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)
    return kl

def categorical_kl_probs(probs1, probs2, eps=1.e-6):
    probs1, probs2 = probs1, probs2
    log_probs1 = torch.log(probs1 + eps)
    log_probs2 = torch.log(probs2 + eps)
    kl = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)
    return kl

def categorical_log_likelihood(x, logits):
    logits = logits
    log_probs = F.log_softmax(logits, dim=-1)
    x = x
    x_onehot = F.one_hot(x, num_classes=logits.shape[-1])
    return (log_probs * x_onehot.float()).sum(dim=-1)

def meanflat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))

def np_tile_imgs(imgs, pad_pixels=1, pad_val=255, num_col=0):
    imgs = np.asarray(imgs)
    n, h, w, c = imgs.shape
    if num_col <= 0:
        num_col = int(np.ceil(np.sqrt(n)))
    num_row = int(np.ceil(n / num_col))
    imgs = np.pad(imgs, pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
                  mode='constant', constant_values=pad_val)
    h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
    imgs = imgs.reshape(num_row, num_col, h, w, c).transpose(0, 2, 1, 3, 4).reshape(num_row * h, num_col * w, c)
    return imgs

def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
    imgs = np_tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col)
    PIL.Image.fromarray(imgs).save(filename)
