import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import functionset as myfc

def main():
    # images = ['test1','test2','test3','test4','test5','test6',
    #           'straight_lines1','straight_lines2']
    # for file_path in images:
    #     myfc.image_converter(file_path)

    myfc.image_converter('harder_challenge_video_Moment_1')

    # clip1 = VideoFileClip("challenge_video.mp4").subclip(0, 5)
    # # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    # white_clip = clip1.fl_image(myfc.pipeline_video)
    # white_clip.write_videofile("challenge_video_w_pipeline.mp4", audio=False)


if __name__ == "__main__":
    # execute only if run as a script
    import os
    os.chdir("C:/Users/hitoshi/AppData/Local/Programs/Python/Python35/Scripts/Udacity/CarND_t1_P4_Adv_lane_finding")
    main()