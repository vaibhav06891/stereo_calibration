import os
import numpy as np
import cv2
import pickle
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from stereo_helpers import *

def main():

	nx = 9 # the number of inside corners in x
	ny = 6 # the number of inside corners in y

	# read and show the test_image
	img = cv2.imread('/home/vdedhia/Desktop/code/stereo_data/left_images/left50.jpg')
	img_size = (img.shape[1], img.shape[0])

	# extract object points and image points for calibration
	objpts, imgpts_left, imgpts_right = prepare_stereo_objpoints(nx,
														 ny, visualize=True)

	'''
		If the calibration for left camera is not already available,
		intrinsic calibration is done using the data that is process
		for stereo calibration
	'''
	if not(os.path.exists("camera_left.json")):
		print("Starting calibration for left camera")
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts,
	  			 				imgpts_left, img_size, None, None)
		data_left = {}
		data_left["K"] = mtx.tolist()
		data_left["dist"] = dist.tolist()
		with open('camera_left.json', 'w') as outfile:
			json.dump(data_left, outfile)

	with open("camera_left.json") as f:
		data_left = json.load(f)
	KLeft = np.asarray(data_left["K"], np.float64)
	DLeft = np.asarray(data_left["dist"], np.float64)	

	'''
		If the calibration for left camera is not already available,
		intrinsic calibration is done using the data that is process
		for stereo calibration
	'''
	if not(os.path.exists("camera_right.json")):
		print("Starting calibration for right camera")
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, 
								imgpts_right, img_size, None, None)
		data_right = {}
		data_right["K"] = mtx.tolist()
		data_right["dist"] = dist.tolist()
		with open('camera_right.json', 'w') as outfile:
			json.dump(data_right, outfile)

	with open("camera_right.json")  as f:
		data_right = json.load(f)
	KRight = np.asarray(data_right["K"], np.float64)
	DRight = np.asarray(data_right["dist"], np.float64)

	flags = cv2.CALIB_FIX_INTRINSIC
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)

	retval, CM1, DC1, CM2, DC2, R, T, E, F = cv2.stereoCalibrate(objpts, imgpts_left, imgpts_right, KLeft, 
												DLeft, KRight, DRight, img_size,criteria,flags)
	
	print("Stereo Reconstruction Error:" + str(retval))

	data ={}
	data['Rotation'] = R.tolist()
	data['Translation'] = T.tolist()
	data['Essential_Matrix'] = E.tolist()
	data['Fundamental_Matrix'] = F.tolist()
	with open('camera_stereo.json', 'w') as outfile:
		json.dump(data, outfile)
	
if __name__ == "__main__":
	main()
