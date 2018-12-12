import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def prepare_stereo_objpoints(nx, ny, visualize=False):
  '''
     Prepare object points
  '''
  objp = np.zeros((ny*nx,3), np.float32)
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d points in real world space
  imgpoints_left  = [] # 2d points in left image plane.
  imgpoints_right = [] # 2d points in right image plane.

  # Make a list of calibration images
  left_images  = glob.glob('/home/vdedhia/Desktop/code/stereo_data/left_images/left*.jpg')
  right_images = glob.glob('/home/vdedhia/Desktop/code/stereo_data/right_images/right*.jpg') 
  left_images.sort()
  right_images.sort()


  # Step through the list and search for chessboard corners
  for i in range(0, len(left_images)):
    img_left  = cv2.imread(left_images[i])
    img_right = cv2.imread(right_images[i])

    gray_left  = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret1, corners1 = cv2.findChessboardCorners(gray_left, (nx,ny), None)
    ret2, corners2 = cv2.findChessboardCorners(gray_right, (nx,ny), None)

    if ret1 == True and ret2 == True:
        objpoints.append(objp)
        imgpoints_left.append(corners1)
        imgpoints_right.append(corners2)

        # Draw and display the corners
        if visualize:
          cv2.drawChessboardCorners(img_left,  (nx,ny), corners1, ret1)
          cv2.drawChessboardCorners(img_right, (nx,ny), corners2, ret2)
          combined_imgs = np.hstack((img_left, img_right))
          cv2.imshow("images", combined_imgs)
          #cv2.waitKey(100)

  cv2.destroyAllWindows()
  
  return objpoints, imgpoints_left, imgpoints_right