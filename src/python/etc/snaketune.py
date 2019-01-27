import cv2

from utils.image.Backend import imread
from utils.workdir import cd_work

cd_work()
img = imread('resource/ext/samples/daylight_course1/00031.jpg', 'bgr')

mat = img.array
mat_yuv = cv2.cvtColor(mat, cv2.COLOR_BGR2YUV)
mat_bin = cv2.inRange(mat_yuv, (100, 80, 140), (230, 135, 185))

cv2.imshow("Orig", mat)
cv2.imshow("Binary", mat_bin)
cv2.waitKey(0)
