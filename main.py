import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functionset as myfc

cal_img_paths = [['./camera_cal/calibration1.jpg',9 ,5],
                 ['./camera_cal/calibration2.jpg',9 ,6],
                 ['./camera_cal/calibration3.jpg',9 ,6],
                 ['./camera_cal/calibration6.jpg',9, 6],
                 ['./camera_cal/calibration7.jpg',9, 6],
                 ['./camera_cal/calibration8.jpg',9, 6],
                 ['./camera_cal/calibration9.jpg',9, 6],
                 ['./camera_cal/calibration10.jpg',9, 6],
                 ['./camera_cal/calibration12.jpg',9, 6],
                 ['./camera_cal/calibration13.jpg', 9, 6],
                 ['./camera_cal/calibration14.jpg', 9, 6],
                 ['./camera_cal/calibration15.jpg', 9, 6],
                 ['./camera_cal/calibration16.jpg', 9, 6],
                 ['./camera_cal/calibration17.jpg', 9, 6],
                 ['./camera_cal/calibration18.jpg', 9, 6],
                 ['./camera_cal/calibration19.jpg', 9, 6],
                 ['./camera_cal/calibration20.jpg', 9, 6]]

def main():
    cal_path= cal_img_paths[5]
    print(cal_path)
    cal_img = cv2.imread(cal_path[0])
    undist_data = myfc.create_cmr_mtx(cal_img, cal_path[1], cal_path[2])

    target_img = cv2.imread('./test_images/test5.jpg')
    original = target_img.copy()
    undist_img = myfc.undist_image(target_img,undist_data)

    pts1 = np.float32([[594, 450], [200, 720], [1079, 720], [685, 450]])
    pts2 = np.float32([[300, 0], [300, 720], [979, 720], [979, 0]])

    binary_img = myfc.create_binary_img(undist_img,pts1,pts2)
    binary_img2, left_lane, right_lane = myfc.lane_writer(binary_img,window_width = 60,window_height = 100,margin = 100, tol_lane_gap=1.5)
    color_img, current_position, left_radius, right_radius = myfc.image_pal_calculatn(binary_img2, left_lane, right_lane,10)
    if current_position <0:
        text_in_img1 = "Position: {0:.3f}".format(abs(current_position)) + "m on right"
    else:
        text_in_img1 = "Current position: {0:.3f}".format(abs(current_position)) + "m on left"

    text_in_img2 = "Left radius: {0:.1f}".format(left_radius) + "m" \
                   + "Right radius: {0:.1f}".format(right_radius) + "m, "

    M = cv2.getPerspectiveTransform(pts2,pts1)
    rev_color_img = cv2.warpPerspective(color_img, M, (1280, 720))

    result = cv2.addWeighted(undist_img, 1, rev_color_img, 0.9, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(result, text_in_img1, (20,50), font, 1.5, (255,255,255), 2, cv2.LINE_4)
    cv2.putText(result, text_in_img2, (20,100), font, 1.5, (255,255,255), 2, cv2.LINE_4)

    ### plot the training and validation loss for each epoch
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))
    ax1.set_title('Original image')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(undist_img,cv2.COLOR_BGR2RGB))
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

    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1,
                        wspace=0.2,hspace=0.2)
    plt.show()

    plt.imshow(result)
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    import os
    os.chdir("C:/Users/hitoshi/AppData/Local/Programs/Python/Python35/Scripts/Udacity/CarND_t1_P4_Adv_lane_finding")
    main()