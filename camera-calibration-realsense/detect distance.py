import cv2
import pyrealsense2
import argparse
from realsense_depth import *
from math import atan2, cos, sin, sqrt, tan, pi

#--------------------------ARUCO ARGS------------------------------------------# 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
    default="DICT_7X7_50",
    help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#

#--------------------ARUCO DEFINITIONS FOR OPENCV------------------------------#
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11  
}
#------------------------------------------------------------------------------#
#--------------------------FUNCTION DEFINITIONS--------------------------------#

def nothing(x):

    # any operation
    pass

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
  textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
 
  return angle

#-----------------------------Camera initialization------------------------#
dc = DepthCamera()
#--------------------------------------------------------------------------#
#-----------------------------Trackbars initialization---------------------#
cv2.namedWindow("Color frame")
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 99, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 20, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 115, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("threshold", "Trackbars", 0, 100, nothing)
#--------------------------------------------------------------------------#

#-----------------------------FONT ----------------------------------------#
font = cv2.FONT_HERSHEY_COMPLEX
#--------------------------------------------------------------------------#

#-------------------------ARUCO VERIFICATION ------------------------------#

if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
        args["type"]))
    sys.exit(0)
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
#--------------------------------------------------------------------------#

#-------------------------------START LOOP --------------------------------#

while True:
    ret, depth_frame, color_frame = dc.get_frame() #Get frame 
    frame = color_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to gray
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HS



#-----------------------------SET TRACKBARS ------------------------------#

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
    # threshold = cv2.getTrackbarPos("threshold", "Trackbars")
    
#----------------------------------------------------------------------------#
#--------------------------SET THRESHOLDS AND MASK---------------------------#

    lower_thresh = np.array([l_h, l_s, l_v])
    upper_thresh = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

#----------------------------------------------------------------------------#
#---------------------------DETECT CONTOURS/ARUCO CORNERS--------------------#

    if int(cv2.__version__[0]) > 3:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:    
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    (aruco_corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

#----------------------------------------------------------------------------#
#--------------------ITERATE OVER ARUCO_CORNERS AND CONTOURS-----------------#
    centers = []
    if len(aruco_corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(aruco_corners, ids):
            aruco_corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = aruco_corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX_aruco = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY_aruco = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX_aruco, cY_aruco), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            centers.append((cX_aruco, cY_aruco))
            for i in range(0, len(centers)-1):
                cv2.line(frame, centers[i], centers[i+1], (255, 0, 0), 2)
                theta_arucos= np.rad2deg(np.arctan2(centers[i][1]-centers[i+1][1],(centers[i][0] - centers[i+1][0])))
                print(centers[0][0])

            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True) #approximates a polygon with required precision
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                ### Set area threshold here: This has not been tuned for the optimal area #TODO
                if area > 1:
                    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cX, cY), 7, (255, 0, 255), -1)
                    ###Set edge threshold here: We will use 4 because we want to detect a quadrilateral
                    if len(approx) == 4:
                        cv2.putText(frame, "Marker", (x, y), font, 1, (0, 0, 0)) #text to show if marker is available
                        cv2.circle(frame, (approx[0][0]),7,(255, 0, 255),-1) #pointers at each corner
                        cv2.circle(frame, (approx[1][0]),7,(255, 0, 255),-1)
                        cv2.circle(frame, (approx[2][0]),7,(255, 0, 255),-1)
                        cv2.circle(frame, (approx[3][0]),7,(255, 0, 255),-1)
        #------------------------------------------FOLLOWING CODE IS FOR DISPLAYING ANGLES-----------------------------------#            
                        # cv2.line(frame, (approx[2][0]), (approx[2][0][0]+2000,approx[2][0][1]),(0,0,255),1)
                        # cv2.line(frame, (approx[2][0]), (approx[2][0][0],approx[2][0][1]-2000),(0,0,255),1) 
                        # for i in range(1,9): #This for loop is used to create the fixed angles (to show)
                        #     cv2.line(frame, (approx[2][0]), (approx[2][0][0]+2000,approx[2][0][1]-int(tan(i*10*3.1415/180)*2000)),(0,255,255),1)
                        # cv2.line(frame, (cX,cY),(cX_aruco,cY_aruco),(255,0,0),2 )
                        theta = np.rad2deg(np.arctan2(cY-cY_aruco,(cX-cX_aruco)))
        #---------------------------------------------------------------------------------------------------------------------#
                        
                        rect = cv2.minAreaRect(cnt)
                        center = (int(rect[0][0]),int(rect[0][1])) 
                        width = int(rect[1][0])
                        height = int(rect[1][1])
                        angle = 90 - int(rect[2])
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        label = "  Rotation Angle: " + str(np.abs(theta_arucos-theta)) + " degrees"
                        textbox = cv2.rectangle(frame, (center[0]-35, center[1]-25), 
                            (center[0] + 295, center[1] + 10), (255,255,255), -1)
                        cv2.putText(frame, label, (center[0]-50, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    
    # cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break