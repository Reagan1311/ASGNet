from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

import numpy as np
import cv2
import torch


def place_seed_points(mask, down_stride=8, max_num_sp=5, avg_sp_area=100):
	'''
	:param mask: the RoI region to do clustering, torch tensor: H x W
	:param down_stride: downsampled stride for RoI region
	:param max_num_sp: the maximum number of superpixels
	:return: segments: the coordinates of the initial seed, max_num_sp x 2
	'''

	segments_x = np.zeros(max_num_sp, dtype=np.int64)
	segments_y = np.zeros(max_num_sp, dtype=np.int64)

	m_np = mask.cpu().numpy()
	down_h = int((m_np.shape[0] - 1) / down_stride + 1)
	down_w = int((m_np.shape[1] - 1) / down_stride + 1)
	down_size = (down_h, down_w)
	m_np_down = cv2.resize(m_np, dsize=down_size, interpolation=cv2.INTER_NEAREST)

	nz = np.nonzero(m_np_down)
	# After transform, there may be no nonzero in the label
	if len(nz[0]) != 0:

		p = [np.min(nz[0]), np.min(nz[1])]
		pend = [np.max(nz[0]), np.max(nz[1])]

		# cropping to bounding box around ROI
		m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]

		# num_sp is adaptive, based on the area of support mask
		mask_area = (m_np_roi == 1).sum()
		num_sp = int(min((np.array(mask_area) / avg_sp_area).round(), max_num_sp))

	else:
		num_sp = 0

	if (num_sp != 0) and (num_sp != 1):
		for i in range(num_sp):

			# n seeds are placed as far as possible from every other seed and the edge.

			# STEP 1:  conduct Distance Transform and choose the maximum point
			dtrans = distance_transform_edt(m_np_roi)
			dtrans = gaussian_filter(dtrans, sigma=0.1)

			coords1 = np.nonzero(dtrans == np.max(dtrans))
			segments_x[i] = coords1[0][0]
			segments_y[i] = coords1[1][0]

			# STEP 2:  set the point to False and repeat Step 1
			m_np_roi[segments_x[i], segments_y[i]] = False
			segments_x[i] += p[0]
			segments_y[i] += p[1]
	
	segments = np.concatenate([segments_x[..., np.newaxis], segments_y[..., np.newaxis]], axis=1)  # max_num_sp x 2
	segments = torch.from_numpy(segments)

	return segments