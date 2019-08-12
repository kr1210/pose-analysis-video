import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.animation as animation
import cv2
import time
import numpy as np
import sys
import pose_match
import parse_openpose_json
import numpy as np
import logging
import prepocessing
import matplotlib.pyplot as plt
import time



fps_time = 0

w = 432
h = 368
#image_path = 'frame30.jpg'
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))


parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
parser.add_argument('--view', type=str, default='')
parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')
parser.add_argument('--Video', type=str, default='')
args = parser.parse_args()


#def get_score_actual(points, det_view):

	


def get_score(input_points, model_points, det_view,model_image,input_image):
	#plt.ion
	input_features = input_points
	catch_later= det_view
	torso_value = 0
	#test_pic = cv2.imread("test.jpg")
	#model_features = get_points(test_pic)
	match_result, torso_value = pose_match.single_person(model_points,input_features, True)
	print("->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", torso_value)
	
	return torso_value

def plot(model_points, input_features, model_image, input_image,torso_value):
	im = pose_match.plot_single_person(model_points, input_features, model_image, input_image,torso_value)
	#plt.show()
	return im





def get_points(image):
	#img = cv2.imread(image)
	points = []
	idx=0
	humans=[]
	pos_data=[]
	img = cv2.resize(image,(432,368))
	if idx%1==0:
		humans = e.inference(image,resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
		for human in humans[:1]:
			pos_temp_data=[]
			for part_idx in range(18):
				if part_idx not in human.body_parts.keys():
					pos_temp_data.extend([0,0])
					continue
				body_part = human.body_parts[part_idx]
				part_idx,x,y,score=body_part.part_idx,body_part.x, body_part.y,body_part.score
				pos_temp_data.extend([x , y])
			pos_data.append(pos_temp_data)
			
			points = np.array(pos_temp_data, dtype=np.float16)
			#print(points)
			points = points.reshape(18,2)
			points = np.multiply(points, np.array([432,368]))			
	idx = idx + 1
	return points


model = cv2.imread("test_real.png")
model_points = get_points(model)
view_point = args.view
cap = cv2.VideoCapture(args.Video)
score_updated = 0
counter = 1
ims = []
if cap.isOpened() is False:
	print("error in opening video")
while cap.isOpened():
	ret_val, image = cap.read()
	if ret_val == True:
		detected_points = get_points(image)
		score = get_score(detected_points,model_points,image,model,image)
		if (score >90):
			score_updated = ((score_updated + score)*100)/counter
			counter = counter + 1
		graph = plot(model_points, detected_points, model, image,score_updated)
		ims.append(graph)
		print(len(ims))
		time.sleep(0.5)
	else:
		break
cap.release()
print("start save")
ani = animation.ArtistAnimation(graph, ims, interval=50, blit=True,repeat_delay=1000)
print("saving")
ani.save('dynamic_images.mp4')
















