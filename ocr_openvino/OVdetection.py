import os


from openvino.inference_engine import IENetwork, IECore

import logging as log
import sys
import cv2
import yaml

import numpy as np
from metrics import *
import time

np.set_printoptions(threshold = np.inf)
class OVdetection:
	def __init__(self,model_path,  device, cpu_extension,config_path):
		self.model_path = model_path
		self.cpu_extension = cpu_extension
		self.device = device
		self.config = self.load_config(config_path)
		self.input_blob = None
		self.out_blob1 = None
		self.out_blob2 = None
		self.ie = None
		self.net = None
		self.n = None
		self.c = None
		self.h = None
		self.w = None

	def load_config(self,path):
		""" Load saved configuration from yaml file. """

		with open(path,'r') as read_file:

			config = yaml.load(read_file)
		return config

	def load_model(self):
		model_xml = self.model_path
		model_bin = os.path.splitext(model_xml)[0] + ".bin"

		# Plugin initialization for specified device and load extensions library if specified
		log.info("Creating Inference Engine")
		ie = IECore()
		if self.cpu_extension and 'CPU' in self.device:
			ie.add_extension(self.cpu_extension, "CPU")
		# Read IR
		log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
		net = IENetwork(model=model_xml, weights=model_bin)

		if "CPU" in self.device:
			supported_layers = ie.query_network(net, "CPU")
			not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
			if len(not_supported_layers) != 0:
				log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
						  format(args.device, ', '.join(not_supported_layers)))
				log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
						  "or --cpu_extension command line argument")
				sys.exit(1)

		log.info("Preparing input blobs")
		self.input_blob = next(iter(net.inputs))
		out_blob = iter(net.outputs)
		self.out_blob1 = next(out_blob)
		self.out_blob2 = next(out_blob)
		
		self.net = net
		n,c,h,w = self.net.inputs[self.input_blob].shape
		#net.reshape({self.input_blob:(n,c,760,1280)})
		self.ie = ie
		self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)


		
		
	def infer(self, inputdata):
		
		self.net.batch_size = len(inputdata)
		n,c,h,w = self.net.inputs[self.input_blob].shape
		images = np.ndarray(shape=(n,c,h,w))

		ori_images = []
		for i in range(n):
			#ori_image = cv2.imread(inputdata[i])
			image = inputdata[i]
			ori_image = image
			if image.shape[:-1] != (h, w):
				#log.warning("Image {} is resized from {} to {}".format(i, image.shape[:-1], (h, w)))
				image = cv2.resize(image, (w, h))
				image = image.astype(np.float32)
			
			image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
			images[i] = image
			ori_images.append(ori_image)
		log.info("Batch size is {}".format(n))
		# Start sync inference
		log.info("Starting inference in synchronous mode")
		
		 # Loading model to the plugin
		log.info("Loading model to the plugin")
		
		start = time.time()
		res = self.exec_net.infer(inputs={self.input_blob: images})
		end = time.time()
		print("detect:{}s".format(end-start))

		# Processing output blob
		log.info("Processing output blob")
		res_link = res[self.out_blob1]
		res_seg= res[self.out_blob2]
		segm_logits = []
		link_logits = []
		for i in range(n):
			link_logits_i = res_link[i].transpose((1,2,0)).reshape((96, 160, 8, 2))  #96*160*8*2
			segm_logits_i = res_seg[i].transpose((1,2,0))
			link_logits.append(link_logits_i)
			segm_logits.append(segm_logits_i)
		
		segm_scores = softmax(segm_logits)
		link_scores = softmax(link_logits)
		#print(segm_logits)
		#images = ori_images#ori_images.transpose((0,3,2,1)) #from CHW to HWC
		s = time.time()
		bboxes = self.to_boxes(ori_images, segm_scores[:, :, :,1],  link_scores[:, :, :, :,1], self.config)
		e = time.time()
		print("bbox",e-s)
		return bboxes

	def min_area_rect(self,contour):
		""" Returns minimum area rectangle. """

		(center_x, cencter_y), (width, height), theta = cv2.minAreaRect(contour)
		return [center_x, cencter_y, width, height, theta], width * height
	def mask_to_bboxes(self,mask, config, image_shape):
		""" Converts mask to bounding boxes. """

		image_h, image_w = image_shape[0:2]

		min_area = config['min_area']
		min_height = config['min_height']

		bboxes = []
		max_bbox_idx = mask.max()
		mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

		for bbox_idx in range(1, max_bbox_idx + 1):
			bbox_mask = (mask == bbox_idx).astype(np.uint8)
			cnts = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
			if len(cnts) == 0:
				continue
			cnt = cnts[0]
			rect, rect_area = self.min_area_rect(cnt)

			box_width, box_height = rect[2:-1]
			if min(box_width, box_height) < min_height:
				continue

			if rect_area < min_area:
				continue

			xys = self.rect_to_xys(rect, image_shape)
			bboxes.append(xys)

		return bboxes
	def rect_to_xys(self,rect, image_shape):
	    """ Converts rotated rectangle to points. """

	    height, width = image_shape[0:2]

	    def get_valid_x(x_coord):
	        return np.clip(x_coord, 0, width - 1)

	    def get_valid_y(y_coord):
	        return np.clip(y_coord, 0, height - 1)

	    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
	    points = cv2.boxPoints(rect)
	    points = np.int0(points)
	    for i_xy, (x_coord, y_coord) in enumerate(points):
	        x_coord = get_valid_x(x_coord)
	        y_coord = get_valid_y(y_coord)
	        points[i_xy, :] = [x_coord, y_coord]
	    points = np.reshape(points, -1)
	    return points


	def get_neighbours(self,x_coord, y_coord):
		""" Returns 8-point neighbourhood of given point. """
		return [(x_coord - 1, y_coord - 1), (x_coord, y_coord - 1), (x_coord + 1, y_coord - 1), \
			(x_coord - 1, y_coord), (x_coord + 1, y_coord), \
			(x_coord - 1, y_coord + 1), (x_coord, y_coord + 1), (x_coord + 1, y_coord + 1)]
	def is_valid_coord(self,x_coord, y_coord, width, height):
		""" Returns true if given point inside image frame. """

		return 0 <= x_coord < width and 0 <= y_coord < height


	def decode_image(self,segm_scores, link_scores, segm_conf_threshold, link_conf_threshold):
		""" Convert softmax scores to mask. """
		segm_mask = segm_scores >= segm_conf_threshold
		link_mask = link_scores >= link_conf_threshold
		points = list(zip(*np.where(segm_mask)))
		height, width = np.shape(segm_mask)
		group_mask = dict.fromkeys(points, -1)

		def find_parent(point):
			return group_mask[point]

		def set_parent(point, parent):
			group_mask[point] = parent

		def is_root(point):
			return find_parent(point) == -1

		def find_root(point):
			root = point
			update_parent = False
			while not is_root(root):
				root = find_parent(root)
				update_parent = True

			if update_parent:
				set_parent(point, root)

			return root

		def join(point1, point2):
			root1 = find_root(point1)
			root2 = find_root(point2)

			if root1 != root2:
				set_parent(root1, root2)

		def get_all():
			root_map = {}

			def get_index(root):
				if root not in root_map:
					root_map[root] = len(root_map) + 1
				return root_map[root]

			mask = np.zeros_like(segm_mask, dtype=np.int32)
			for point in points:
				point_root = find_root(point)
				bbox_idx = get_index(point_root)
				mask[point] = bbox_idx
			return mask

		for point in points:
			y_coord, x_coord = point
			neighbours = self.get_neighbours(x_coord, y_coord)
			for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
				if self.is_valid_coord(neighbour_x, neighbour_y, width, height):

					link_value = link_mask[y_coord, x_coord, n_idx]
					segm_value = segm_mask[neighbour_y, neighbour_x]
					if link_value and segm_value:
						join(point, (neighbour_y, neighbour_x))

		mask = get_all()
		return mask

	def decode_batch(self,segm_scores, link_scores, config):
		""" Returns boxes mask for each input image in batch."""

		batch_size = segm_scores.shape[0]
		batch_mask = []
		for image_idx in range(batch_size):
			image_pos_pixel_scores = segm_scores[image_idx, :, :]
			image_pos_link_scores = link_scores[image_idx, :, :, :]
			mask = self.decode_image(image_pos_pixel_scores, image_pos_link_scores,
								config['segm_conf_thr'], config['link_conf_thr'])
			batch_mask.append(mask)
		return np.asarray(batch_mask, np.int32)

	def to_boxes(self,image_data, segm_pos_scores, link_pos_scores, conf):
		""" Returns boxes for each image in batch. """
		bboxes = []
		for item, seg_item, link_item in zip(image_data,segm_pos_scores,link_pos_scores):
			seg_item = np.expand_dims(seg_item, axis=0)
			link_item = np.expand_dims(link_item, axis=0)
			#print(item.shape,seg_item.shape,link_item.shape)
			mask = self.decode_batch(seg_item, link_item, conf)[0, ...]
			item_box = self.mask_to_bboxes(mask, conf, item.shape)
			bboxes.append(item_box)

		# print(image_data.shape,segm_pos_scores.shape,link_pos_scores.shape)
		# mask = self.decode_batch(segm_pos_scores, link_pos_scores, conf)[0, ...]
		# bboxes = self.mask_to_bboxes(mask, conf, image_data.shape)

		return bboxes


