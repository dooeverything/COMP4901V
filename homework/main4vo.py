'''
It is a main function that can be used for you to conduct the tasks in Visual Odometry.
You run this main function to generate the expected outputs and results described in the Instruction.pdf, 
by calling functions implemented in submission.py and helper.py
You are free to write it in your own style.
'''

# Insert your package here
import cv2 as cv
import numpy as np
from submission import *

from skimage.measure import ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt

if __name__ == "__main__":
	'''
	Replace pass with your implementation
	'''
	# img1 = cv.imread('data/image1.jpg')
	# img2 = cv.imread('data/image2.jpg')

	# print(img1.shape)

	# # Load Intrinsic Matrix 
	# K = {}
	# with open('data/Intrinsic4Recon.npz', 'r') as f:
	# 	for line in f:
	# 		for value in line.split():
	# 			if value == 'K1:' or value == 'K2:':
	# 				key = value[:2]
	# 				K[key] = []
	# 			else:
	# 				K[key].append(float(value))

	# K1 = np.array([K['K1']]).reshape((3,3))
	# K2 = np.array([K['K2']]).reshape((3,3))


	# Rt = essentialDecomposition(img1, img2, K1, K2)

	# print("Decompose essential:")
	# print(Rt)

	gt_pose = []
	xs = []
	ys = []
	with open('data/GTPoses.npz', 'r') as f:
		for line in f:
			t = np.fromstring(line, dtype=np.float64, sep=' ')
			t = t.reshape((3,4))
			x = t[0][3]
			y = t[2][3]
			xs.append(x)
			ys.append(y)
			# print(t.shape)
			gt_pose.append(t)
    
	traj = visualOdometry('data/monocular video sequence/data', gt_pose)
	path_pred = [traj[:][0], traj[:][3]]
	path_gt = [xs, ys]

	print(path_pred)

	raise NotImplementedError("Stop at main 4 vo")

	fig, ax = plt.subplots(figsize=(12, 9))
	# ax.plot(xs, ys)


	# path

	for name, path, color in zip(opt_names, all_paths, colors):
		# ax.quiver(path[0, :-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color=color, lw=3)
		ax.plot([], [], color=color, label=name, lw=3)

	# for i in range(1,len(traj)):
	# 	print(f"x: {traj[i][0]} y: {traj[i][2]}")
	

	plt.show()
