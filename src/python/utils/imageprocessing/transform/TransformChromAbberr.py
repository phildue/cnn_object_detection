import cv2
import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformChromAbberr(ImgTransform):

    def __init__(self, scale: (float, float, float), t_x: (float, float, float), t_y: (float, float, float)):

        self.transmat = np.zeros((2, 3, 3))
        for i in range(3):
            self.transmat[:, :, i] = np.array([[scale[i], 0, t_x[i]],
                                               [0, scale[i], t_y[i]]])
        self.t_y = t_y
        self.t_x = t_x
        self.scale = scale

    def transform(self, img: Image, label: ImgLabel = ImgLabel([])):
        label_t = label.copy()
        mat = np.zeros(img.shape, dtype=np.uint8)
        for i in range(3):
            if img.format is not 'bgr':
                raise ValueError("Chromatic:: Wrong Color format!")

            mat[:, :, i] = cv2.warpAffine(src=img.array[:, :, i], M=self.transmat[:, :, i],
                                          dsize=(mat.shape[1], mat.shape[0]))


        return Image(mat, img.format), label_t
