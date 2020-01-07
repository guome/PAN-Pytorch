# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 14:18
# @Author  : zhoujun
import os
import cv2
import torch
import time
import subprocess
import numpy as np

from .pypse import pse_py
from .kmeans import km

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def decode(preds, scale=1, threshold=0.7311, min_area=5, bbox_type='rectangle'):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    from .pse import pse_cpp, get_points, get_num
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    score = preds[0].astype(np.float32)
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel
    similarity_vectors = preds[2:].transpose((1, 2, 0))

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < min_area:
            continue
        label_values.append(label_idx)

    dis_threshold = 0.8
    pred = pse_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, dis_threshold)
    pred = pred.reshape(text.shape)

    bbox_list = []
    label_points = get_points(pred, score, label_num)
    for label_value, label_point in label_points.items():
        if label_value not in label_values:
            continue
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < 100 / (scale * scale):
            continue

        if score_i < 0.93:
            continue
        
        if bbox_type == 'rectangle':
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
        elif bbox_type == 'contour':
            binary = (pred == label_value).astype(np.uint8)
            _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bbox = contours[0]

        bbox_list.append(bbox)
    return pred, bbox_list
