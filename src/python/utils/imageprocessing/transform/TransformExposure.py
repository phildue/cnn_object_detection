import cv2
import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformExposure(ImgTransform):

    def __init__(self, contrast, delta_exposure):
        self.delta_exposure = delta_exposure
        self.contrast = contrast

    def transform(self, img: Image, label: ImgLabel = ImgLabel([])):
        label_t = label.copy()
        mat = img.array.copy()

        for i in range(3):
            exposure = np.log(255 / mat[:, :, i] - 1) / (-self.contrast)
            reexposure = exposure + self.delta_exposure
            reexposed = 255 / (1 + np.exp(-self.contrast * reexposure))
            mat[:, :, i] = reexposed

        return Image(mat, img.format), label_t
