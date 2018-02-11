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
[image2]: ./output_images/test10.png "example2"
[image3]: ./output_images/sobelx_example.png "sobelx_binary output"
[image4]: ./output_images/s_channel_example.png "s compositon output"
[video1]: ./project_video_w_pipeline.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

### Rubric points for this project are shown in following link

 [Link](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md)

 Required data is shonw below.

 In following section, procedure in each step is shown.
 Fig.1 shows examples of outputs from each procedure.

#### Fig.1 Example of output
##### (a) Conversion of straight_lines1.jpg in test_images directory
![alt text][image1]
##### (b) Conversion of test10.jpg in test_images directory
![alt text][image2]

### 1. Compute the camera calibration matrix and distortion

 Function of "create_cmr_mtx" is defined in functionset.py. ( from line 7 to 36 in functionset.py)  
 This function is used at line 369 and 434 in functionset.py with parameter of 9 and 6.  
   
 Firstly, corners' coordinate in calibration image was calculated with findChessboardCorners function from OpenCV.  
 Based on corners' coordinate and object points, camera calibration and distortion matrix were calculated.  
 Undistortion calculation was done based on these matrixes with undist_image function. (from line 38 to 48 in functionset.py) 

 In Fig1, examples of undistorted image are shown in upper center. Original image is in upper left.
 
 For this calibration, calibration8.jpg was selected from "camera_cal" directory, because it showed best straight line with testing all other ipg.  
 The best result means that it showed the biggest radius with test of straight_lines1.jpg and straight_line2.jpg. 

### 2. Convert to binary image and apply perspective transform

 Function of "create_binary_img" was defined to convert original BGR image to binary image including perspective transform.(from line 50 to 103 in functionset.py).  
 
 For binary image conversion, combination of color and gradient thresholds were applied.  
 Firstly, BGR image was converted to HLS at line 59.
 Secondary, pixels showing low value in L channel were excluded from S channel. This is to reduce the effect of shadow. Example is b) of Fig.1. (from line 64 to 67 in functionset.py)
 Then, histogram equalization was applied fro S channel to reduce effect of color change of pavement.(from line 69 to 72 in functionset.py)
 For S channel, threshould was directory applied. (From line 74 to 77 in functionset.py).  
 For gradient threshould was applied for L channel. (From line 79 to 90 in functionset.py). 

 Example of each output is shown in Fig2. 
 
#### Fig.2 Conversion based each threshould
##### (a) Gradient threshold with L channel
![alt text][image3]
##### (b) Threshould with S channel
![alt text][image4]

After marge of both binary images, perspective transformation was applied (From line 88 to 94 in functionset.py)
Coordinats for perspective transfrom are shonw in table1.  
These coordinates were defined to keep symmetricity for vertial axle. And detail values were tuned to show parallel line in test of straight_lines1.jpg and straight_line2.jpg.
In Fig1, examples of image afrer perspective transform are shown in upper right.

### Table.1 Perspective trans form parameter  
|         | Original image | After conversion |
|:-------------:|:-------------:| :-----:|
| Position1 | [594, 450] | [300, 0] |
| Position1 | [200, 720] | [300, 720] |
| Position1 | [1079, 720] | [979, 720] |
| Position1 | [685, 450] | [979, 0] |


#### 4. Lane position detection
 "find_window_centroids" function in functionset.py was defined to detect lane position. (From line 113 to 221 in functionset.py)  
 Firstly, find the two starting positions for the left and right lane by using np.sum to get the vertical image slice and then np.convolve the vertical image slice with the window template. (From line 146 to 159 in functionset.py)
 This position was set as initial position.  
 Then, lane position in each layer was calculated with convolution. (From line 163 to 212 in functionset.py)  
 If difference of center position in the next layer is more than multiplication between tol_lane_gap and window_width, position will be exclude for output and None will be added.

 In Fig1, examples of detected position are shown in bottom left.


#### 5. Polynomial fitting and parameter calculation
 For polynomial curve fitting from lane positions, "polyfit_funt" function was defined. (From line 275 to 289  in functionset.py) 
 As shown in 4, lane position list includes None data. They were excluded in this function.
 In Fig1, examples of polynominal curve are shown in bottom center.  

 For calculation of lane radius and position of vehicle, "image_pal_calculatn" function was defined. (From line 291 to 345)
 Vehicle position was calculated as mid point between end of lane curves, calculated with polyfit_funt. (From line 305 to 309)
 Radius was calculated based on coefficient of polynomial fitting. (From line 342 to 343)
 [Detail about appoximation of radius.](https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

 In Fig1. right bottom image includs all parameters.   

#### 6. Video with pipline

 Pipline for video was defined as "pipeline_video" function. (From line 397 to 435)
 Output video data is here [link to my video result](./project_video_w_pipeline.avi)  
 To see video, please download it to your local computer.

#### 7. Discussion



 
