import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_cmr_mtx(img,nx,ny):
    """ Img: image data of chess board to calculate camera matrix
    nx, ny: number of row and column of chess board
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Create Camera matrix in accordance with CV2 page
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        objpoints.append(objp)
        imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("here")
        print("total error:", total_erro / len(objpoints))

        return [mtx,dist]
    else:
        print("error: Conner can not be found")
        return None

def create_cmr_mtx(img,nx,ny):
    '''
    Calculate Camera calibration matrix and distortion coefficients
    :param img: image to
    :param nx: number of column of chess board
    :param ny: number of row of chess board
    :return: Camera calibration matrix and distortion coefficient
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Create Camera matrix in accordance with CV2 page
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        objpoints.append(objp)
        imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return [mtx,dist]

    else:
        print("error: Conner can not be found")
        return None

def undist_image(img, undist_data):
    '''
    :param img: numpy matrix of image data .
    :param undist_data: camera matrix for undistortion
    :return: img: undistorted image
    '''
    rtn_img = img
    temp_image = []
    for ind in range(3):
        rtn_img[:,:,ind] = cv2.undistort(img[:,:,ind],undist_data[0], undist_data[1], None, undist_data[0])
    return rtn_img


def create_binary_img(img,pts1,pts2):
    '''
    :param img:
    :return:
    '''
    #
    # Apply Histogram Equalization
    # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    #

    s_thresh = (180, 255)
    sx_thresh = (0, 30)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    temp_img = []
    l_channel = img[:,:,1]
    s_channel = img[:,:,2]

    hist_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = hist_eq.apply(l_channel)
    s_channel = hist_eq.apply(s_channel)

    # Convert from l channel data to binary data based on Sobel
    #

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # s_channel = clahe.apply(s_channel)
    # l_channel = clahe.apply(l_channel)
    gray = clahe.apply(gray)

    sobelx_g = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx_g = np.absolute(sobelx_g)
    scaled_sobel_g = np.uint8(255 * abs_sobelx_g / np.max(abs_sobelx_g))
    binary_output_g =  np.zeros_like(sobelx_g)
    s_thresh_g = (30, 50)
    binary_output_g[(scaled_sobel_g >= s_thresh_g[0]) & (scaled_sobel_g <= s_thresh_g[1])] = 1

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobelx_s = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx_s = np.absolute(sobelx_s)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_s = np.uint8(255 * abs_sobelx_s / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    #
    # Convert from S channel to binary data
    #
    s_binary = np.zeros_like(s_channel)
    s_binary[:,:] = 1
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 0

    # plt.imshow(s_binary)
    # plt.show()


    #
    # Combine both sobel and S channel data
    #
    output_binary = np.zeros_like(s_binary)
    output_binary[(s_binary==1) & (sxbinary==1)] = 1

    #
    # Combine both sobel and S channel data
    #
    temp_output_binary = np.zeros_like(s_binary)
    temp_output_binary[output_binary==0] = 1

    #
    # Perspective transform of binary file
    #
    M = cv2.getPerspectiveTransform(pts1, pts2)
    temp_output_binary = cv2.warpPerspective(temp_output_binary, M, (1280, 720))

    #
    # Lane finding
    #

    # histogram = np.sum(temp_output_binary[temp_output_binary.shape[0] / 2:, :], axis=0)

    # print(temp_output_binary.mean())

    return temp_output_binary

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin, tol_lane_gap):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    l_center_1_bf = l_center
    r_center_1_bf = r_center
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        hight_center = image.shape[0] - (level-0.5) * window_height

        l_key=True
        r_key=True

        if level == 0:
            pass
        else:
            if l_center > l_center_1_bf+window_width*tol_lane_gap or l_center < l_center_1_bf-window_width*tol_lane_gap:
                l_center = l_center_1_bf
                l_key = False
            else:
                pass
            if r_center > r_center_1_bf+window_width*tol_lane_gap or r_center < r_center_1_bf-window_width*tol_lane_gap:
                r_center = r_center_1_bf
                r_key = False
            else:
                pass

            l_center_1_bf = l_center
            r_center_1_bf = r_center

        if l_key and r_key:
            window_centroids.append((l_center, r_center))
        else:
            window_centroids.append((None, None))

    return window_centroids

def lane_writer(warped, window_width = 50,window_height = 80,margin = 100, tol_lane_gap=2):

    # target_img = cv2.cvtColor(target_img, cv2.COLOR_HLS2BGR)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin, tol_lane_gap)

    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            if window_centroids[level][0] == None:
                pass
            else:
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
                r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((warped, warped, warped)) * 255  # making the original road pixels 3 color channels
        output_img = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
        window_center_l = []
        window_center_r = []
        for i, data_pos in enumerate(window_centroids):
            if data_pos[0] == None:
                pass
            else:
                window_center_l.append([data_pos[0],warped.shape[0]-(0.5+i)*window_height])
                window_center_r.append([data_pos[1],warped.shape[0]-(0.5+i)*window_height])
    # for data in wi
    # If no window centers found, just display orginal road image
    else:
        output_img = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    return output_img, window_center_l,window_center_r

def polyfit_funt(lane_info, xm_per_pix, ym_per_pix):
    x_cord = []
    y_cord = []
    for i in lane_info:
        x_cord.append(i[0])
        y_cord.append(i[1])

    fit_data = np.polyfit(np.array(y_cord) * ym_per_pix, np.array(x_cord) * xm_per_pix, 2)

    return fit_data

def image_pal_calculatn(img, left_lane, right_lane, lane_thickness = 5):
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    xm_per_pix = 3.7/620
    ym_per_pix = 4*4/350

    # Calculate position

    left_fit_cr = polyfit_funt(left_lane,1,1)
    right_fit_cr = polyfit_funt(right_lane,1,1)

    left_end = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval**1 + left_fit_cr[2]
    right_end = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval**1 + right_fit_cr[2]

    # calculate mid point between left and right
    current_position = (np.mean((left_end,right_end))-1279/2)*xm_per_pix

    #Create lane image
    sxbinary_road = np.zeros_like(cv2.split(img)[0])
    sxbinary_left = np.zeros_like(cv2.split(img)[0])
    sxbinary_right = np.zeros_like(cv2.split(img)[0])
    for count_y,slice_zeors_x in enumerate(sxbinary_road):
        left_end = left_fit_cr[0] * count_y ** 2 + left_fit_cr[1] * count_y ** 1 + left_fit_cr[2]
        right_end = right_fit_cr[0] * count_y ** 2 + right_fit_cr[1] * count_y ** 1 + right_fit_cr[2]
        for count_x, dumy_data in enumerate(slice_zeors_x):
            if count_x < left_end + lane_thickness or count_x > right_end - lane_thickness:
                sxbinary_road[count_y,count_x] = 0
            else:
                sxbinary_road[count_y, count_x] = 1

        for count_x, dumy_data in enumerate(slice_zeors_x):
            if count_x >= left_end - lane_thickness and count_x <= left_end + lane_thickness:
                sxbinary_left[count_y, count_x] = 1
            else:
                sxbinary_left[count_y, count_x] = 0

        for count_x, dumy_data in enumerate(slice_zeors_x):
            if count_x >= right_end - lane_thickness and count_x <= right_end + lane_thickness:
                sxbinary_right[count_y, count_x] = 1
            else:
                sxbinary_right[count_y, count_x] = 0

    lane_img = cv2.merge((sxbinary_left,sxbinary_road,sxbinary_right))
    lane_img = lane_img *120

    left_fit_cr = polyfit_funt(left_lane,xm_per_pix,ym_per_pix)
    right_fit_cr = polyfit_funt(right_lane,xm_per_pix,ym_per_pix)

    left_radius  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius  = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    return lane_img,current_position,left_radius,right_radius
