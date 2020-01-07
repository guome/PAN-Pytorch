# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun
import os
import glob
import pathlib

data_path = '/home/insight/datasets/text_detect/icdar2019_lsvt/train'

f_w = open(os.path.join(data_path, 'train.txt'), 'w', encoding='utf8')
ext_list = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
file_list = []
for ext in ext_list:
    file_list.extend(glob.glob(data_path + '/images/*{}'.format(ext), recursive=True))
for img_path in file_list:
    d = pathlib.Path(img_path)
    #label_path = os.path.join(data_path, 'gt_labels', ('gt_' + str(d.stem) + '.txt'))
    label_path = os.path.join(data_path, 'gt_labels', (str(d.stem) + '.txt'))
    if os.path.exists(img_path) and os.path.exists(label_path):
        print(img_path, label_path)
    else:
        print('不存在', img_path, label_path)
    f_w.write('{}\t{}\n'.format(img_path, label_path))
f_w.close()
