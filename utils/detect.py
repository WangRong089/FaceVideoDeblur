import os
import face_alignment
import pickle
import shutil
import cv2
import numpy as np

"""
Generate landmarks for ground truth frames, clean outliers and select filtered subset
"""
def detect_GT_landmarks(dataset='300VW_STFAN'):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    preds = fa.get_landmarks_from_directory('./datasets/300VW_STFAN')

    with open(f'../datasets/{dataset}/landmarks.pkl', 'wb') as g:
        pickle.dump(preds, g)


def filter_landmarks(dataset='300VW_STFAN'):
    result = {}
    undetected = set([])
    vids = set([])

    with open(f'../datasets/{dataset}/landmarks.pkl', 'rb') as f:
        landmarks = pickle.load(f)

        frames = []
        for i in landmarks.keys():
            path = i.split('\\')
            vid, phase, frame = path[2], path[3], int(path[4][:-4])
            if phase == 'input':
                continue
            if landmarks[i] is None or (landmarks[i][0] < 0).any():
                undetected.add(vid)
            if not landmarks[i] is None:
                frames.append(landmarks[i][0])
            if frame == 74:
                if not vid in undetected:
                    assert len(frames) == 75
                    result[vid] = frames
                frames = []
            vids.add(vid)

    with open(f'../datasets/{dataset}/landmarks_clean.pkl', 'wb') as g:
        pickle.dump(result, g)

    return result

def filter_dataset(dataset='300VW_STFAN_P'):
    with open(f'../datasets/{dataset}/landmarks_clean.pkl', 'rb') as f:
        landmarks = pickle.load(f)
        keys = landmarks.keys()

        vids = os.listdir(f'../datasets/{dataset}/train')

        for vid in vids:
            if vid not in keys:
                shutil.rmtree(f'../datasets/{dataset}/train/{vid}')




