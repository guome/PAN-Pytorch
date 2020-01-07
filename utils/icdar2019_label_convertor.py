# coding: utf-8
# @Author: oliver
# @Date:   2019-11-08 11:31:36

import os
import sys
import cv2
import json
import numpy as np
from tqdm import tqdm

label_path = 'train_full_labels.json'
images_path = 'images'
output_path = 'gt_labels'

with open(label_path, 'rb') as file_reader:
	data = file_reader.read().decode('utf-8')
data = json.loads(data, encoding='utf-8')
for (img_name, gt_list) in tqdm(data.items()):
	img_path = os.path.join(images_path, img_name + '.jpg')
	if not os.path.exists(img_path):
		print('{} not exists.'.format(img_path))
		exit(0)
	file_writer = open(os.path.join(output_path, img_name + '.txt'), 'wb')
	for gt in gt_list:
		bbox = np.array(gt['points']).reshape(-1).tolist()
		assert len(bbox) % 2 == 0
		bbox = list(map(str, bbox))
		text = '###' if gt['illegibility'] else gt['transcription']
		line = '\t'.join(bbox)
		line += '\t{}\n'.format(text.encode('utf-8'))
		file_writer.write(line)
	file_writer.close()
