## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is repository for write up report for Project 4 of Udacity Nano Degree.
In this readme file, overview of this repository is shown.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

 The images for camera calibration are stored in the folder called `camera_cal`. 
 The images in `test_images` are for testing your pipeline on single frames.  

### Structure of this repository

The following resources can be found in this github repository:
**Files**
* main.py
* functionset.py
* project_video.mp4
* project_video.mp4
* writeup_report.md

**Directory**
* camera_cal
* test_images
* output_images
* examples

## Details About Files In This Repository

### `main.py`

 main.py has 2 steps. 

Step1:  
 Creation of converted image files for reporting. Original files were saved into "test_images" directory.
Output files will be saved into "output_images" direcotry.
Output files include image from each image processing step, image with lane, radius and position calculated based on image.

Step2:  
Creation of Converted mp4 file.
Output file includes colored lane, radius and position calculated based on image.

### `functionset.py`

This file includes functions used for main.py.
There are 2 main functions.
"image_converter"(from line 327 to 395 ) and "pipeline_video" (from line 397 to 435 ).
Each function is used for each of step1 and step2 in main.py respectively.

### `project_video.mp4`

 Input file for step2 in main.py.

### `project_video_w_pipeline.mp4`
 Output file for step2 in main.py

### `writeup_report.md`
 writeup report for this project.

### `Directory:  camera_cal`
 Images for camera calibration.
 With testing all images on step1 in main,py, calibration8.jpg showed the best result.
 Then, calibration8.jpg is used for camera calibration in functionset.py.

### `test_images`
 Input files for step1 in main.py.
 
### `output_images`
 Output files for step1 in main.py.