# Camera-Calibration-RealSense
This repo provides an end-to-end intrinsics + extrinsics calibration solution for Intel RealSense (Tested on D435i)

SETUP: <br>
Step 1: Create a conda environment : https://docs.conda.io/en/latest/miniconda.html <br>
Step 2: Install opencv :- 
```
pip install opencv-python
pip install opencv-python-contrib
```
Step 3: Install pyrealsense2:- 

```
pip install pyrealsense2

```
RUNNING THE CODE: <br>

Step 1: Print out the Checkerboard_19_by_13.pdf provided in the repo<br>
Step 2: Measure the length of each side of the square to confirm the length to be provided while calibrating the camera <br>
Step 3: Connect your Intel RealSense Camera and place the Checkerboard in the field of view of the camera <br>
Step 4: Run the "capture_images_realsense.py" file :- 
```
python3 capture_images_realsense.py
```
Step 5: Press "Y" to indicate that the images folder needs to be cleared so you can add your own images. <br> 
Step 6: Move the checkerboard around at various orientations and translations. Press spacebar every time you want to capture a picture at that pose. The more you click, the better! <br>
Step 7: Now you must have images inside your /images folder. <br>
Step 8: Next, run "calibrate_realsense.py" :
```
python3 calibrate_realsense.py -H 19 -V 13 -S 25
```
Here the S parameter will be the value you calculated in Step 2. Make sure your H is always more than your V (place the sheet horizontally)<br>
Step 9: This should save your camera intrinsic values to camera_calibration.json and print the camera extrinsic values to the terminal



