## Writeup report for Advanced lane finding project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/straight_lines1.png "example1"
[image2]: ./output_images/test2.png "example2"
[image3]: ./output_images/sobelx_example.png "sobelx_binary output"
[image4]: ./output_images/s_channel_example.png "s compositon output"
[video1]: ./project_video_w_pipeline.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Rubric points for this project are shown in following link

 [Link](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md)

 Required data is shonw below.

### Camera Calibration

 In following section, procedure in each step is shown.
 Output of each steps are shown in Fig.1.

#### Fig.1 
##### (a) Conversion of test1.png in test_images directory
![alt text][image1]

#### (b) Conversion of test2.png in test_images directory
![alt text][image2]


#### 1. Compute the camera calibration matrix and distortion

 Function of "create_cmr_mtx" is defined in functionset.py. ( from line 7 to 36 )  
 This function is used at line 343 and 408 in functionset.py with parameter of 9 and 6.  
   
 Firstly, corners' coordinate in calibration image was calculated with findChessboardCorners function from OpenCV.  
 Based on corners' coordinate and object points, camera calibration and distortion matrix were calculated.  
 Undistortion calculation was done based on these matrixes with undist_image function. (from line 38 to 48) 

 In Fig1, examples of undistorted image is shown in upper center.
 
 For this calibration, calibration8.jpg was selected from "camera_cal" directory, because it showed best straight line with testing all other ipg.  
 The best results mean that it showed the biggest radius with test of straight_lines1.jpg and straight_line2.jpg. 

#### 2. Convert to binary image and apply perspective transform

 Function of "create_binary_img" was defined to convert original BGR image to binary image including perspective transform.(from line 50 to 90).  
 
 For binary image conversion, combination of color and gradient thresholds were applied.  
 Firstly, BGR image was converted to HLS at line 59.
 Then, gradient threshould was applied for L channel. (From line 64 to 67). 
 For S channel, threshould was directory applied. (From line 69 to 71).  
 Example of each output is shown in Fig2. 
 
 ### Fig.2 Conversion based each threshould
##### (a) Gradient threshold with L channel  
![alt text][image3]  
##### (b) Threshould with S channel  
![alt text][image4]  

After marge of both binary images, perspective transformation was applied (From line 78 to 88)
Coordinats for perspective transfrom are shonw in table1.  
These coordinates were defined to keep symmetricity for vertial axle. And detail values were tuned to show parallel line in test of straight_lines1.jpg and straight_line2.jpg.
In Fig1, examples of undistorted image is shown in upper right.

### Table.1 Perspective trans form parameter  
|         | Original image | After conversion |
|:-------------:|:-------------:| :-----:|
| Position1 | [594, 450] | [300, 0] |
| Position1 | [200, 720] | [300, 720] |
| Position1 | [1079, 720] | [979, 720] |
| Position1 | [685, 450] | [979, 0] |


#### 4. Lane position detection

 "find_window_centroids" function in functionset.py was defined to detect lane position. (From line 108 to 208)  
 Firstly, find the two starting positions for the left and right lane by using np.sum to get the vertical image slice and then np.convolve the vertical image slice with the window template. (From line 124 to 130)
 This position was set as initial position.  
 Then, lane position in each layer was calculated with convolution. (From line 135 to 199)  
 If difference of center position in the next layer is more than multiplication between tol_lane_gap and window_width, position will be exclude for output and None will be added.

#### 5. Polynomial fitting and parameter calculation
  
 For polynomial curve fitting from lane positions, "polyfit_funt" function was defined. (From line 256 to 270) 
 As shown in 4, lane position list includes None data. They were excluded in this function.

 For calculation of lane radius and position of vehicle, "image_pal_calculatn" function was defined. (From laine 274 to 325)
 Vehicle position was calculated as mid point between end of lane curves, calculated with polyfit_funt. (From line 288 295)
 Radius was calculated based on the following formula.

 In Fig1. right bottom image includs all parameters.   

#### 6. Video with pip line

 Pip line for video was defined as "pipeline_video" function. (From line 397 to 435)
 Output video data is here [link to my video result](./project_video.mp4)

