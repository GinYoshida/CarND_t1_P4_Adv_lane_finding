import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    :param img: image data to do perspective transform
    :param pts1: coordinate of original image
    :param pts2: coordinate of converted image
    :return: Transformed image
    '''
    s_thresh = (185, 255)# Threshold to do binary conversion with s compositon
    sx_thresh = (0, 20) # Threshold to do binary conversion with sobel image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    temp_img = []
    l_channel = img[:,:,1]
    s_channel = img[:,:,2]

    shadow_filter = np.zeros_like(l_channel)
    shadow_filter[(l_channel <= 20) & (l_channel >= 0)] = 1
    s_channel[(shadow_filter==1)]=0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    s_channel = clahe.apply(s_channel)

    sobelx_s = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx_s = np.absolute(sobelx_s)
    scaled_sobel_s = np.uint8(255 * abs_sobelx_s / np.max(abs_sobelx_s))
    sxbinary_s = np.zeros_like(scaled_sobel_s)
    sxbinary_s[(scaled_sobel_s >= 50) & (scaled_sobel_s <= 190)] = 1

    # Convert from l channel data to binary data based on Sobel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.ones_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 0

    # Convert from S channel to binary data
    s_binary = np.zeros_like(s_channel)
    s_binary[:, :] = 0
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= 255)] = 1

    # equ = cv2.equalizeHist(s_channel)


    # Combine both sobel and S channel data
    output_binary = np.zeros_like(sxbinary)
    # output_binary[(((sxbinary==1) | (s_binary==1)) & (sxbinary_s==1)) | (s_binary==1)] = 1
    output_binary[(sxbinary==1) | (s_binary==1)] = 1


    res = np.hstack((sxbinary*100, s_binary*100, sxbinary_s*100, output_binary*100))
    # plt.imshow(res)
    # plt.show()


    res = np.hstack((s_channel, sxbinary_s*100))
    # plt.imshow(res)
    # plt.show()



    # Combine both sobel and S channel data
    # temp_output_binary = np.zeros_like(sxbinary)
    # temp_output_binary[output_binary==0] = 1

    # Perspective transform of binary file
    M = cv2.getPerspectiveTransform(pts1, pts2)
    temp_output_binary = cv2.warpPerspective(output_binary, M, (1280, 720))

    return temp_output_binary

def window_mask(width, height, img_ref, center, level):
    '''
    Return masked image to draw lane position.
    :param width: width of masking area
    :param height: hight of masking area
    :param img_ref:reference image to get shape of image
    :param center: position of masked window
    :param level: Level to indicate which holizontal layer will be masked
    :return: Maked image
    '''
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin, tol_lane_gap):
    '''
    Calculate positon of right and left lane.
    :param image: original image
    :param window_width: width of masking area
    :param window_height: height of masking area
    :param margin: margin to cut off unnecessary area from original image
    :param tol_lane_gap: value to calculate how much difference is allowed to set for next layer from original layer.
    If difference of center position in the next later is more than multiplication between tol_lane_gap and window_width,
    position will be exclude for output and None will be added.
    :return: List of center position of left and right in each horizontal layer.
    '''

    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)
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
        # temp_windows_width = (int)(window_width * (level / (image.shape[0] / window_height) + 0.5))
        temp_windows_width = window_width
        window = np.ones(temp_windows_width)
        window = np.ones(temp_windows_width)
        # Create our window template that we will use for convolutions
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = temp_windows_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        hight_center = image.shape[0] - (level-0.5) * window_height

        # Values to check whether calculated center position is within tolerance or not.
        l_key=True
        r_key=True
        # tolerance check.
        if level == 0:
            pass
        else:
            if l_center > l_center_1_bf+temp_windows_width*tol_lane_gap or l_center < l_center_1_bf-temp_windows_width*tol_lane_gap:
                l_key = False
            else:
                pass
            if r_center > r_center_1_bf+temp_windows_width*tol_lane_gap or r_center < r_center_1_bf-temp_windows_width*tol_lane_gap:
                r_key = False
            else:
                pass

        if conv_signal[l_min_index:l_max_index].max() <= 100:
            l_key = False
        else:
            pass
        if conv_signal[r_min_index:r_max_index].max() <= 100:
            r_key = False
        else:
            pass

            #New position calculation
            window_center_l = []
            window_center_r = []
            for i, data_pos in enumerate(window_centroids):
                if data_pos[0] == None:
                    pass
                else:
                    window_center_l.append([data_pos[0], image.shape[0] - (0.5 + i) * window_height])
                    window_center_r.append([data_pos[1], image.shape[0] - (0.5 + i) * window_height])
            if len(window_center_l) <=2 or len(window_center_r) <=2:
                l_center_1_bf = l_center
                r_center_1_bf = r_center
            else:
                l_fit_ex = polyfit_funt(window_center_l,1,1)
                r_fit_ex = polyfit_funt(window_center_l,1,1)

                level_cal = (level + 0.5)*window_height
                l_center_1_bf = l_fit_ex[0] * level_cal ** 2 + l_fit_ex[1] * level_cal ** 1 + l_fit_ex[2]
                r_center_1_bf = r_fit_ex[0] * level_cal ** 2 + r_fit_ex[1] * level_cal ** 1 + r_fit_ex[2]

        if l_key and r_key:
            window_centroids.append((l_center, r_center))
            l_center_1_bf = l_center
            r_center_1_bf = r_center
        else:
            window_centroids.append((None, None))

    return window_centroids

def lane_writer(warped, window_width = 50,window_height = 80,margin = 100, tol_lane_gap=2):
    '''
    Draw images to show the position of lane.
    :param warped: Binay image after perspective transformation
    :param window_width: width of masking area
    :param window_height: height of masking area
    :param margin: margin to cut off unnecessary area from original image
    :param tol_lane_gap: value to calculate how much difference is allowed to set for next layer from original layer.
    If difference of center position in the next later is more than multiplication between tol_lane_gap and window_width,
    position will be exclude for output and None will be added.
    :return:Image including right and left lane, list of left and right lane center positions.
    '''

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

    # If no window centers found, just display orginal road image
    else:
        output_img = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    return output_img, window_center_l,window_center_r

def polyfit_funt(lane_info, xm_per_pix, ym_per_pix):
    '''
    Calculate Polynomial curve fitting information in 3rd order.
    :param lane_info: coordination of lane calculated from base image
    :param xm_per_pix: coefficient to calculate meter from pixel in x direction
    :param ym_per_pix: coefficient to calculate meter from pixel in y direction
    :return: return coefficients of polynomial curve
    '''
    x_cord = []
    y_cord = []
    for i in lane_info:
        x_cord.append(i[0])
        y_cord.append(i[1])

    fit_data = np.polyfit(np.array(y_cord) * ym_per_pix, np.array(x_cord) * xm_per_pix, 2)

    return fit_data

def image_pal_calculatn(img, left_lane, right_lane, lane_thickness = 5):
    '''
    covert image file with colored lane.
    :param img: image to be converted
    :param left_lane: coordination of left lane calculated from base image
    :param right_lane: coordination of right lane calculated from base image
    :param lane_thickness: lane thickness in converted image
    :return:
    '''
    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    xm_per_pix = 3.7/620
    ym_per_pix = 4*4/350

    # Calculate position
    left_fit_cr = polyfit_funt(left_lane,1,1)
    right_fit_cr = polyfit_funt(right_lane,1,1)
    left_end = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval**1 + left_fit_cr[2]
    right_end = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval**1 + right_fit_cr[2]

    left_end_in_dir = left_fit_cr[2]
    right_end_in_dir  = right_fit_cr[2]

    # calculate mid point between left and right
    current_position = (np.mean((left_end,right_end))-1279/2)*xm_per_pix
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

    return lane_img,current_position,left_radius,right_radius,left_end_in_dir,right_end_in_dir

def image_converter(input_file_name):
    '''
    To convert images with colored lane, position of vehicle, radias fo lane.
    Also, add output in each ste of conversion.
    Create image from each steps.
    Exported image will be saved into "output_images" directory.
    :param input_file_name
    :return: None
    '''
    file_path = './test_images/' + input_file_name + '.jpg'
    print(file_path)
    target_img = cv2.imread(file_path)
    original = target_img.copy()

    # camera caliburation data
    cal_img = cv2.imread('./camera_cal/calibration8.jpg')
    undist_data = create_cmr_mtx(cal_img, 9, 6)
    undist_img = undist_image(target_img, undist_data)

    # Convert binary image and apply perspective transform
    pts1 = np.float32([[594, 450], [200, 720], [1079, 720], [685, 450]])
    pts2 = np.float32([[300, 0], [300, 720], [979, 720], [979, 0]])
    binary_img = create_binary_img(undist_img, pts1, pts2)

    # Draw lane image
    binary_img2, left_lane, right_lane = lane_writer(binary_img, window_width=50, window_height=100, margin=100,
                                                          tol_lane_gap=1.5)
    color_img, current_position, left_radius, right_radius,dummy_1,dummy2 =\
        image_pal_calculatn(binary_img2, left_lane, right_lane, 10)
    if current_position < 0:
        text_in_img1 = "Position: {0:.3f}".format(abs(current_position)) + "m on right"
    else:
        text_in_img1 = "Current position: {0:.3f}".format(abs(current_position)) + "m on left"
    text_in_img2 = "Left radius: {0:.1f}".format(left_radius) + "m" \
                   + "Right radius: {0:.1f}".format(right_radius) + "m, "

    # Marge lane image and original image
    M = cv2.getPerspectiveTransform(pts2, pts1)
    rev_color_img = cv2.warpPerspective(color_img, M, (1280, 720))
    result = cv2.addWeighted(undist_img, 1, rev_color_img, 0.9, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(result, text_in_img1, (20, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(result, text_in_img2, (20, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

    # plot the training and validation loss for each epoch
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original image')
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted image')
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(binary_img)
    ax3.set_title('Binaly img')
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(binary_img2)
    ax4.set_title('Binaly with after lane finding')
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(color_img)
    ax5.set_title('Lane meta data')
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(result)
    ax6.set_title('Converted image')
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1,wspace=0.2, hspace=0.2)
    output_path = './output_images/' + input_file_name + '.png'
    plt.savefig(output_path)

    return None

def pipeline_video(input_img):
    '''
    To convert images with colored lane, position of vehicle, radius of lane.
    :param input_file_name
    :return: images with colored lane, position of vehicle, radius of lane.
    '''
    target_img = input_img
    original = target_img.copy()

    # camera caliburation data
    cal_img = cv2.imread('./camera_cal/calibration8.jpg')
    undist_data = create_cmr_mtx(cal_img, 9, 6)
    undist_img = undist_image(target_img, undist_data)

    # Convert binary image and apply perspective transform
    pts1 = np.float32([[594, 450], [200, 720], [1079, 720], [685, 450]])
    pts2 = np.float32([[300, 0], [300, 720], [979, 720], [979, 0]])
    binary_img = create_binary_img(undist_img, pts1, pts2)

    # Draw lane image
    binary_img2, left_lane, right_lane = lane_writer(binary_img, window_width=50, window_height=100, margin=100,
                                                          tol_lane_gap=1.5)
    color_img, current_position, left_radius, right_radius, left_end_in_dir,right_end_in_dir =\
        image_pal_calculatn(binary_img2, left_lane, right_lane, 10)
    if current_position < 0:
        text_in_img1 = "Position: {0:.3f}".format(abs(current_position)) + "m on right"
    else:
        text_in_img1 = "Current position: {0:.3f}".format(abs(current_position)) + "m on left"
    text_in_img2 = "Left radius: {0:.1f}".format(left_radius) + "m" \
                   + "Right radius: {0:.1f}".format(right_radius) + "m, "
    # Marge lane image and original image
    M = cv2.getPerspectiveTransform(pts2, pts1)
    rev_color_img = cv2.warpPerspective(color_img, M, (1280, 720))
    result = cv2.addWeighted(undist_img, 1, rev_color_img, 0.9, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(result, text_in_img1, (20, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(result, text_in_img2, (20, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

    return undist_img, rev_color_img,current_position,left_radius,right_radius,left_end_in_dir,right_end_in_dir

def video_creation(original_video_name, output_video_name, end_sec = 1, start_sec = 0, flg_whole_vide = False):

    video = cv2.VideoCapture(original_video_name)
    total_num_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('project_video_w_pipeline.mp4', fourcc, fps, (1280, 720))

    start_frame = start_sec * fps
    end_frame = end_sec * fps
    if flg_whole_vide == True:
        ent_frame = total_num_frame
    else:
        pass

    #variable to judge irregularity.
    time_record = []
    time_record_position = []
    time_record_r_rigth = []
    time_record_r_left = []
    time_record_width_in_dir = []
    previous_image = None
    previous_position = None
    previous_l_radius = None
    previous_r_radius = None
    previous_width = None

    for num_frame in range(0,(int)(end_frame)):
        if num_frame < start_frame:
            ret, frame = video.read() #pass until start flame
        else:
            print((int)(num_frame-start_frame),"/",(int)(end_frame - start_frame))
            #Key to judge irregularity of lane data
            left_key = True
            right_key = True
            width_key = True
            ret, frame = video.read()

            if ret == True:
                undist_img, temp_img, current_position,left_radius,right_radius,left_end_in_dir,right_end_in_dir \
                    = pipeline_video(frame)
                width = right_end_in_dir - left_end_in_dir

                print(num_frame)
                try:
                    r_rec = np.array(time_record_r_rigth[-5:-1])
                    upper_bd = np.mean(r_rec) + 3*np.std(r_rec)
                    lower_bd = np.mean(r_rec) - 3*np.std(r_rec)
                    if right_radius >= upper_bd or right_radius <= lower_bd:
                        right_key = False
                    else:
                        pass

                    l_rec = np.array(time_record_r_left[-5:-1])
                    upper_bd = np.mean(l_rec) + 3*np.std(l_rec)
                    lower_bd = np.mean(l_rec) - 3*np.std(l_rec)
                    if left_radius >= upper_bd or left_radius <= lower_bd:
                        left_key = False
                    else:
                        pass

                    width_rec = np.array(time_record_width_in_dir[-5:-1])
                    upper_bd = np.mean(width_rec) + 4*np.std(width_rec)
                    lower_bd = np.mean(width_rec) - 4*np.std(width_rec)

                    if width >= upper_bd or width <= lower_bd or width < 0:
                        width_key = False
                        print("width: NOK")
                    else:
                        pass
                except:
                    pass

                if num_frame - start_frame < 3:
                    print("here")
                    width_key = True
                    left_key = True
                    right_key = True

                if right_key == True and left_key == True and width_key == True:
                   result = cv2.addWeighted(undist_img, 1, temp_img, 0.9, 0)

                   if current_position < 0:
                       text_in_img1 = "Position: {0:.3f}".format(abs(current_position)) + "m on right"
                   else:
                       text_in_img1 = "Current position: {0:.3f}".format(abs(current_position)) + "m on left"
                   text_in_img2 = "Left radius: {0:.1f}".format(left_radius) + "m" \
                                  + "Right radius: {0:.1f}".format(right_radius) + "m, "
                   font = cv2.FONT_HERSHEY_DUPLEX
                   cv2.putText(result, text_in_img1, (20, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)
                   cv2.putText(result, text_in_img2, (20, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

                   previous_image = temp_img
                   previous_position = current_position
                   previous_l_radius = left_radius
                   previous_r_radius = right_radius
                   previous_width = width

                   time_record_position.append(current_position)
                   time_record_r_left.append(left_radius)
                   time_record_r_rigth.append(right_radius)
                   time_record_width_in_dir.append(width)
                else:
                    result = cv2.addWeighted(undist_img, 1, previous_image, 0.9, 0)

                    if current_position < 0:
                        text_in_img1 = "Position: {0:.3f}".format(abs(previous_position)) + "m on right"
                    else:
                        text_in_img1 = "Current position: {0:.3f}".format(abs(previous_position)) + "m on left"
                    text_in_img2 = "Left radius: {0:.1f}".format(previous_l_radius) + "m" \
                                   + "Right radius: {0:.1f}".format(previous_r_radius) + "m, "
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(result, text_in_img1, (20, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)
                    cv2.putText(result, text_in_img2, (20, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_4)

                    time_record_position.append(current_position)
                    time_record_r_left.append(left_radius)
                    time_record_r_rigth.append(right_radius)
                    time_record_width_in_dir.append(width)

                time_record.append(num_frame/fps)
                out.write(result)
            else:
                break
    # Release everything if job is finished
    video.release()
    out.release()
    cv2.destroyAllWindows()
