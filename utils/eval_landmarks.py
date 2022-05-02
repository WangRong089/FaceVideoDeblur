import numpy as np 
import cv2
import pickle
import torch
from torch.autograd import Variable

def generate_gt(size, landmark_list, sigma):
    '''
    return N * H * W
    '''
    heatmap_list = [
        _generate_one_heatmap(size, l, sigma) for l in landmark_list
    ]
    return np.stack(heatmap_list, axis=0)

def _generate_one_heatmap(size, landmark, sigma):
    w, h = size
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap

def generate_gt_tensor(size, landmark_list, sigma):
    '''
    return N * H * W
    '''
    landmark_list = (landmark_list * 255).long()
    heatmap_list = [
        _generate_one_heatmap_tensor(size, l, sigma) for l in landmark_list
    ]
    return np.stack(heatmap_list, axis=0)

def _generate_one_heatmap_tensor(size, landmark, sigma):
    w, h = size
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - landmark[0].item())**2 + (yy - landmark[1].item())**2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap

def heatmap2landmark(heatmap):
    h = heatmap.transpose(1, 0)
    n ,_,_, d = h.size()
    m = h.view(n, -1).argmax(1)
    # indices has range [0, 1], whose pixel indices are [0, 255]
    indices = torch.cat(((m / d).view(-1, 1), (m % d).view(-1, 1)), dim=1).float() / 255
    return indices