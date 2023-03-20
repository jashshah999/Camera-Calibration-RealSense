import cv2 as cv
import numpy as np
import time
import os
import pyrealsense2
from realsense_depth import *
import argparse



if not os.path.exists("images"):
	os.makedirs("images")
else:
	print("[INFO] erasing images in the folder?")
	if ((input("[Y/N] ").capitalize()) == "Y"):
		for file in os.listdir("images"):
			os.remove("images/" + file)
	else:
		print("[INFO] exiting...")

cv.namedWindow("camera capture preview")

i = 0
dc = DepthCamera()
while True:

	ret, _,frame = dc.get_frame()

	sho_frame = frame.copy()
	cv.putText(sho_frame, "Press 'space' to capture", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	cv.putText(sho_frame, "Press 'q' to exit", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	cv.imshow("preview", sho_frame)
	
	key = cv.waitKey(1)
	if key == 32:  # space
		cv.imwrite("images/img" + str(i) + ".png", frame)
		i += 1
		print(f"[INFO] Image {i} saved")
	elif key == ord("q"):
		print("[INFO] exiting...")
		break

# cap.release()
cv.destroyAllWindows()
