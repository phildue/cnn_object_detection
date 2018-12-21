# https://www.packtpub.com/mapt/book/application_development/9781785283932/2/ch02lvl1sec21/motion-blur
import random

import cv2
import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.imageprocessing.transform.TransformMotionBlur import TransformMotionBlur
from utils.labels.ImgLabel import ImgLabel


class RandomMotionBlur(ImgTransform):

    def __init__(self, sigma_min=0.1, sigma_max=1.5, kernel_size=15):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.kernel_size = kernel_size
        self.kernels = {
            'horizontal': TransformMotionBlur.kernel_horizontal,
            'vertical': TransformMotionBlur.kernel_vertical,
        }
        self.direction = list(self.kernels.keys())[0]

    def transform(self, img: Image, label: ImgLabel):
        img_t = img.copy()
        label_t = label.copy()
        direction = random.choice(list(self.kernels.keys()))
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        kernel = self.kernels[direction](self.kernel_size, sigma)
        img_t.array = cv2.filter2D(img_t.array, -1, kernel)

        return img_t, label_t
