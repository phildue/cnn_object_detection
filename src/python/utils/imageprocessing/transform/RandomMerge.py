import random

import numpy as np

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomMerge(ImgTransform):

    def __init__(self, pixel_frac=0.005, kernel_size=(9, 9)):
        self.pixel_frac = pixel_frac
        self.offset = ((np.array(kernel_size) - 1) / 2).astype(np.int)

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = label.copy()

        rows, cols, _ = img.shape
        n_pixels = int(np.floor(self.pixel_frac * (rows - 2 * self.offset[1]) * (cols - 2 * self.offset[0])))
        y = np.array([random.randint(self.offset[0], rows - self.offset[0]) for _ in range(n_pixels)]).reshape((-1, 1))
        x = np.array([random.randint(self.offset[1], cols - self.offset[1]) for _ in range(n_pixels)]).reshape((-1, 1))

        idx = np.concatenate([x, y], -1)
        for n in range(n_pixels):
            i, j = idx[n]
            img_aug.array[j - self.offset[0]:j + self.offset[0], i - self.offset[1]:i + self.offset[1]] = \
                np.mean(
                    np.mean(img_aug.array[j - self.offset[0]:j + self.offset[0], i - self.offset[1]:i + self.offset[1]],
                            0), 0)

        return img_aug, label_aug
