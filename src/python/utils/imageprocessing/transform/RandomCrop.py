import copy
import random

from utils.imageprocessing.Backend import crop
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomCrop(ImgTransform):
    total_fails = 0

    def __init__(self, c_min=0.7, c_max=0.9):
        self.c_min = c_min
        self.c_max = c_max

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        if len(label.objects) >= 0:
            for i in range(100):
                img_aug, label_aug = self._crop(img, label)

                if len(label_aug.objects) > 0:
                    return img_aug, label_aug
            self.total_fails += 1
            print("Augmenter Crop Total Fails: " + str(self.total_fails))
        return img_aug, label_aug

    def _crop(self, img: Image, label: ImgLabel):

        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        w_crop_min = int(random.uniform(0, 1.0 - self.c_max) * img.shape[1])
        h_crop_min = int(random.uniform(0, 1.0 - self.c_max) * img.shape[0])

        w_crop_max = int(random.uniform(self.c_min, self.c_max) * img.shape[1])
        h_crop_max = int(random.uniform(self.c_min, self.c_max) * img.shape[0])

        img_aug, label_aug = crop(img_aug, label=label_aug, min_xy=(w_crop_min, h_crop_min),
                                  max_xy=(w_crop_max, h_crop_max))
        return img_aug, label_aug
