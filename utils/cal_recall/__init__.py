# -*- coding: utf-8 -*-
# @Time    : 1/16/19 6:40 AM
# @Author  : zhoujun
from .straight_text import straight_text_metrics
from .curve_text import curve_text_metrics

def cal_recall_precision_f1(gt_path, result_path, text_type='curve', show_result=False):
	if text_type == 'curve':
		return curve_text_metrics(gt_path, result_path, show_result)
	elif text_type == 'straight':
		return straight_text_metrics(gt_path, result_path, show_result)
	else:
		raise NotImplementedError('invalid text type!')

