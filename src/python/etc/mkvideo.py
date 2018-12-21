#!/usr/local/bin/python3
import glob

import cv2
import numpy as np

# # Construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
# ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
# args = vars(ap.parse_args())
# Arguments
# dir_path = '.'
# ext = args['extension']
# output = args['output']
from utils.workdir import cd_work

cd_work()
src_dir = 'resource/ext/samples/iros2018_course_final_simple_17gates/'
ext = '.jpg'
output = 'out/synthetic.avi'
files = list(sorted(glob.glob(src_dir + '*' + ext)))

# Determine the width and height from the first image
frame = cv2.imread(files[0])
cv2.imshow('video', frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 10.0, (width, height))

for f in files:

    frame = cv2.imread(f)

    out.write(frame.astype(np.uint8))  # Write out frame to video

    cv2.imshow('video', frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
