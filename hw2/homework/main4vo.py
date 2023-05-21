'''
It is a main function that can be used for you to conduct the tasks in Visual Odometry.
You run this main function to generate the expected outputs and results described in the Instruction.pdf, 
by calling functions implemented in submission.py and helper.py
You are free to write it in your own style.
'''

# Insert your package here
import numpy as np
import cv2 as cv
from submission import visualOdometry

if __name__ == "__main__":
	gt_pose = []
	xs = np.zeros((1591,1))
	ys = np.zeros((1591,1))
	i = 0

	with open('data/GTPoses.npz', 'r') as f:
		for line in f:
			t = np.fromstring(line, dtype=np.float64, sep=' ')
			t = t.reshape((3, 4))
			xs[i][0] = t[0][3]	
			ys[i][0] = t[2][3]
			gt_pose.append(t)
			i += 1

	path_gt = np.hstack((xs, ys))
	traj = visualOdometry('data/monocular video sequence/data', gt_pose)
