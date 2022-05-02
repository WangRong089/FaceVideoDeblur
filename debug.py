import os
# import face_alignment
import pickle
import shutil
import cv2
import numpy as np

from utils.eval_landmarks import generate_gt
import torch
from models.srfbn_hg_arch import merge_heatmap_5, FeedbackBlockCustom, FeedbackBlockHeatmapAttention
import torchvision

torch.cuda.empty_cache()


def visualize_landmarks(img='./datasets/300VW_STFAN_P/train/002_v0/GT/00024.jpg'):
    landmarks = open('./datasets/300VW_STFAN_P/landmarks_clean.pkl', 'rb')
    l = pickle.load(landmarks)['002_v0'][24]
    image = cv2.imread(img)
    image = np.copy(np.flipud(image))
    w, h, c = image.shape
    l[:,1] = w - l[:,1]
    print(len(l))
    for i in range(68):
        image = cv2.circle(image, tuple(l[i]), radius=0, color=(0, 0, 255), thickness=-1)
    cv2.imshow('image',image)
    cv2.waitKey(0)

def visualize_heatmap(img='./datasets/300VW_STFAN_P/train/002_v0/GT/00024.jpg'):
    landmarks = open('./datasets/300VW_STFAN_P/landmarks_clean.pkl', 'rb')
    l = pickle.load(landmarks)['002_v0'][24]
    image = cv2.imread(img)
    result = generate_gt((256, 256), l, 1)
    ht = torch.from_numpy(np.ascontiguousarray(result))
    new_heatmap = merge_heatmap_5(ht.unsqueeze(0), False)
    for i in range(5):
        print(new_heatmap[0, i, :, :].unsqueeze(2).shape)
        cv2.imshow('image', new_heatmap[0,i,:,:].unsqueeze(2).numpy())
        cv2.waitKey(0)

def verify_feedback_block(img='./datasets/300VW_STFAN_P/train/002_v0/GT/00024.jpg'):
    landmarks = open('./datasets/300VW_STFAN_P/landmarks_clean.pkl', 'rb')
    l = pickle.load(landmarks)['002_v0'][24]
    t = torchvision.transforms.ToTensor()

    ht = generate_gt((256, 256), l, 1)
    ht = torch.from_numpy(np.ascontiguousarray(ht))
    new_heatmap = merge_heatmap_5(ht.unsqueeze(0), False).to(torch.device('cuda'))
    new_heatmap = new_heatmap.to(torch.float32)

    
    image = t(cv2.imread(img))
    image = image.unsqueeze(0).to(torch.device('cuda'))


    print(image.shape, new_heatmap.shape)

    with torch.no_grad():
        c = FeedbackBlockCustom(num_features=48, num_groups=6, upscale_factor=4, act_type='prelu', norm_type=None, num_features_in=3).to(torch.device('cuda'))
        s = FeedbackBlockHeatmapAttention(num_features=48, num_groups=6, upscale_factor=4, act_type='prelu', norm_type=None, num_heatmap=5, num_fusion_block=7).to(torch.device('cuda'))
        output = c(image)
        result = s(output, new_heatmap)
        print(result.shape)


def verify_match_key():
    vids = os.listdir('./datasets/300VW_STFAN_P/train')
    landmarks = pickle.load(open('./datasets/300VW_STFAN_P/landmarks_clean.pkl', 'rb'))
    for vid in vids:
        if vid in landmarks.keys():
            print(vid)


verify_match_key()
    