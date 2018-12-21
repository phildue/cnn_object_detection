import copy

import numpy as np

from utils.imageprocessing.Backend import color_shift
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomColorShift(ImgTransform):

    def __init__(self, red=(0.0, 1.0), green=(0.0, 1.0), blue=(0.0, 1.0)):
        self.blue = blue
        self.green = green
        self.red = red

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug, label_aug = color_shift(img_aug,
                                         np.array([np.random.uniform(self.blue[0], self.blue[1]),
                                                   np.random.uniform(self.green[0], self.green[1]),
                                                   np.random.uniform(self.red[0], self.red[1])]),
                                         label_aug)

        return img_aug, label_aug
