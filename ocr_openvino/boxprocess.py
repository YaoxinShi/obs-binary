abnormal_ch_space = 155
abnormal_x = 250
abnormal_y = 250
abnormal_distance =300
abnormal_line = 20
abnormal_line_space = 300
abnormal_line_w_min = 50
ch_min_h = 25
ch_min_w = 25
ch_max_h = 10
def sort_box(bboxes):
	import pandas as pd
	import os
	import numpy as np
	res = []
	direction = -1
	count_v = 0
	count_h = 0

	listx = []
	listy = []
	listx2 = []
	listy2 = []
	boxlist = []
	listvalue = []

	for box in bboxes:
		xmin = np.min(np.array(box[::2]))
		xmax = np.max(np.array(box[::2]))
		ymin = np.min(np.array(box[1::2]))
		ymax = np.max(np.array(box[1::2]))
		
		listx.append(int(xmin))
		listy.append(int(ymin))

		listx2.append(int(xmax))
		listy2.append(int(ymax))

		boxlist.append(box)

		w = xmax - xmin
		h = ymax - ymin
		if w > h:
			count_h += 1
		else:
			count_v +=1
	
	if count_h > count_v:
		direction = 0
		df = pd.DataFrame({'xmin':listx,'ymin':listy,'xmax':listx2,'ymax':listy2,"bboxes":boxlist})
	else:
		direction = 1
		df = pd.DataFrame({'ymin':listx,'xmin':listy,'ymax':listx2,'xmax':listy2,"bboxes":boxlist})
		
	#check wheher in same line
	df = df.sort_values(['ymax','xmin'])
	temp1 =0 
	dfidx = 0
	listline = []
	line_threshold = 25
	for index,row in df.iterrows():
		y1 = row["ymin"]
		
		if dfidx != 0 and abs(temp1 - y1) > line_threshold:
			 newlistline = sorted(listline, key = lambda k:k['xmin'])
			 newlistline = filter0(newlistline)
			 res.extend(newlistline)
			 listline = []
		temp1 = y1
		dfidx+=1
		dictbox = {"xmin":row["xmin"],"ymin":row["ymin"],"xmax":row["xmax"],"ymax":row["ymax"],"bboxes":row["bboxes"]}
		listline.append(dictbox)
	newlistline = sorted(listline, key = lambda k:k['xmin'])
	newlistline = filter0(newlistline)
	res.extend(newlistline)
	res = filter3(res)
	res = specific_box(res)
	return res,direction


#if box is  abnormal, remove

def filter0(listline):
	newlistline = []
	for idx,item in enumerate(listline):
		w = item["xmax"] - item["xmin"]
		h = item["ymax"] - item["ymin"]
		#print(h,w)
		#abnoraml_flag = 0
		if h < ch_min_h or  w < ch_min_w or (w > ch_min_h and w < h):
			continue
		
		newlistline.append(item)
	return newlistline


#if one line is far from others, remove
def filter3(reslist):
	import numpy as np
	import math
	new_reslist = []
	maxlen = 0
	maxkey = ""
	for key,line in enumerate(reslist):
		w = line["xmax"] - line["xmin"]
		if maxlen < w:
			maxlen =  w
			maxkey = key
	center_y = reslist[maxkey]["ymax"]
	for key,line in enumerate(reslist):
		if key == maxkey:
			new_reslist.append(line)
			continue
		linedistance = abs(line["ymax"] - center_y)
		#print(linedistance)
		if linedistance > abnormal_line_space:
			continue
		new_reslist.append(line)
	return new_reslist


def specific_box(reslist):
	if len(reslist) == 2:
		xmin_0 = reslist[0]["xmin"]
		xmax_0 = reslist[0]["xmax"]
		ymin_0 = reslist[0]["ymin"]
		ymax_0 = reslist[0]["ymax"]
		xmin_1 = reslist[1]["xmin"]
		xmax_1 = reslist[1]["xmax"]
		ymin_1 = reslist[1]["ymin"]
		ymax_1 = reslist[1]["ymax"]
		w_0 =  ymax_0 - ymin_0
		w_1 = ymax_1 - ymin_1
		
		if w_0 < w_1*2/3 or (ymin_0 < ymin_1 and xmin_0 > xmax_1):
			reslist = reslist[::-1]
			#print("reverse")
	return reslist


def showbox(reslist,fn,im):
	import numpy as np
	import os
	import cv2
	newim_root = "output"
	for item in reslist:
		box = item
		pts = np.array(box).reshape((-1,1,2))
		cv2.polylines(im,[pts],True,(0,255,0),2)
		
	cv2.imwrite(os.path.join(newim_root,fn),im)