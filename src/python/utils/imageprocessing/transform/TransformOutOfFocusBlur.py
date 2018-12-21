import cv2
import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformOutOfFocusBlur(ImgTransform):

    def __init__(self, kernel_size, sigmaX, sigmaY):
        self.sigmaY = sigmaY
        self.sigmaX = sigmaX
        self.kernel_size = kernel_size

    def transform(self, img: Image, label: ImgLabel = ImgLabel([])):
        label_t = label.copy()
        mat = cv2.GaussianBlur(src=img.array, ksize=self.kernel_size, sigmaX=self.sigmaX, sigmaY=self.sigmaY)
        return Image(mat, img.format), label_t
