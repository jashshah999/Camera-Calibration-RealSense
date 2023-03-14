# Camera-Calibration-RealSense
This repo provides an end-to-end intrinsics + extrinsics calibration solution for Intel RealSense (Tested on D435i)

SETUP: 
```
Step 1: Create a conda environment : https://docs.conda.io/en/latest/miniconda.html
```

Step 1: Print out the Checkerboard_19_by_13.pdf provided in the repo
Step 2: Measure the length of each side of the square to confirm the length to be provided while calibrating the camera
Step 3: Connect your Intel RealSense Camera and place the Checkerboard in the field of view of the camera
Step 4: Run the "capture_images_realsense.py" file :- "python3 capture_images_realsense.py"
Step 5: Press "Y" to indicate that the images folder needs to be cleared so you can add your own images. 
Step 6: Move the checkerboard around at various orientations and translations. Press spacebar every time you want to capture a picture at that pose. The more you click, the better!
Step 7: Now you must have images inside your /images folder. 
Step 8: Next, run "calibrate_realsense.py" : "python3 
