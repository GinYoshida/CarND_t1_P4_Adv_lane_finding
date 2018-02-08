import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import functionset as myfc

def main():
    '''Main function to execute following 2 procedure
     Step1: Creation of converted image files for reporting. Original files were saved into "test_images" directory.
     Output files will be saved into "output_images" direcotry.
     Output files include image from each image processing step, image with lane,
    radius and position calculated based on image.

    Step2: Creation of Converted mp4 file.
     Output file includes colored lane, radius and position calculated based on image.
    '''

    # Part for step1
    # define target images. The image need be saved into "test_images" directory.
    # for list of images, extention is not necessary.
    # images = ['straight_lines1','straight_lines2','test1','test2','test3','test4','test5','test6','test7',
    #           'test8','test9','test10','test11']
    #
    # images = ['test10','test9','test8','test11']
# # , 'test9', 'test10']

    # for file_path in images:
    #    myfc.image_converter(file_path)

    # Part for step2
    myfc.video_creation("project_video.mp4","project_video_w_pipeline.mp4",42,40,False)

if __name__ == "__main__":
    # execute only if run as a script
    import os
    os.chdir("C:/Users/hitoshi/AppData/Local/Programs/Python/" +
             "Python35/Scripts/Udacity/CarND_t1_P4_Adv_lane_finding")
    main()
