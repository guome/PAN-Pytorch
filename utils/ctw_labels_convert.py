# coding: utf-8
# @Author: oliver
# @Date:   2019-11-25 20:52:44


import os
import sys
import numpy as np
from shapely.geometry import *

labels_path = 'origin_labels'
output_dir = 'gt_labels'
labels_list = os.listdir(labels_path)
for file in labels_list:
    file_name = os.path.join(labels_path, file)
    with open(file_name, encoding='utf-8', mode='r') as f:
        boxes = []
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            x1 = np.int(params[0])
            y1 = np.int(params[1])

            box = [np.int(params[i]) for i in range(4, 32)]
            box = np.asarray(np.asarray(box) + ([x1 * 1.0, y1 * 1.0] * 14)).astype(np.int)
            box = [[int(box[j]), int(box[j+1])] for j in range(0,len(box),2)]
            try:
                pgt = Polygon(box)
            except Exception as e:
                print('Not a valid polygon.', pgt)
                continue

            if not pgt.is_valid: 
                print('GT polygon has intersecting sides.', pts)
                continue
            
            pRing = LinearRing(box)
            if pRing.is_ccw:
                box.reverse()
            boxes.append(np.array(box).reshape(-1))
        boxes = np.asarray(boxes)
    saved_path = os.path.join(output_dir, file)
    np.savetxt(saved_path, boxes, fmt='%d', delimiter=',')
                
