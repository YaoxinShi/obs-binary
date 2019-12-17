#coding=utf-8
import cv2
from OVdetection import OVdetection

import os
import argparse



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-md", "--md", help="Required. Path to an .xml file with a trained model.",
				 default="model/FP16/ka_1_0_x08/text_detection.xml",
				  type=str) 
ap.add_argument("-mrh", "--mrh", help="Required. Path to an .xml file with a trained model.",
				 default="model/FP32/recognition.xml",
				  type=str)   


ap.add_argument("-l", "--cpu_extension",
					help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
						"kernels implementations.", type=str, default="cpu_extension_avx2.dll")

ap.add_argument("-d", "--device",
					help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
						"acceptable. The demo will look for a suitable plugin for device specified. "
						"Default value is CPU", default="GPU", type=str)
ap.add_argument("-c", "--config",
					help="Optional. depend on different model", default="config/detection.yml", type=str)
args = vars(ap.parse_args())


from glob import glob
import numpy as np
from boxprocess import *
from augmentation import *
import os
import time
if __name__=='__main__':
	
	detect_model = OVdetection(args["md"],args["device"],args["cpu_extension"],args["config"])
	detect_model.load_model()
	#recogh_model = OVrecognition(args["mrh"],args["device"],args["cpu_extension"])
	#recogh_model.load_model()
	

	totaltime = 0
	detection = 0
	recognition = 0
	#res_txt = open('containernumber_result_new.txt', 'w')
	#default: only  one image 
	imlist = glob('./testinput/*.png')
	start = time.time()
	
	for idx,imfn in enumerate(imlist):
		
		im = cv2.imread(imfn)
		_range = np.max(abs(im))
		im = im / _range
		#print(im.shape)
		#print(im[0][0:10])
		imname = os.path.basename(imfn)
		detect_start = time.time()

		bboxes = detect_model.infer([im])
		for i , line in enumerate(bboxes[0]):
			pts = np.array([[line[0],line[1]],[line[2],line[3]],[line[4],line[5]],[line[6],line[7]]], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(im,[pts],True,(0,0,255), 2)

		im = cv2.resize(im, (960, 540))
		cv2.imshow('hi', im)
		cv2.waitKey()  

	#end = time.time()
	#totaltime = end - start
	#print("totaltime:{}s".format(totaltime))


