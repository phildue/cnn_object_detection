import copy
import random

from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class RandomEnsemble(ImgTransform):
    def __init__(self, augmenters: [(float, ImgTransform)]):
        self.augmenters = [a[1] for a in augmenters]
        self.probs = [a[0] for a in augmenters]

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        for i, a in enumerate(self.augmenters):
            if random.uniform(0, 1) <= self.probs[i]:
                img_aug, label_aug = a.transform(img_aug, label_aug)

        return img_aug, label_aug
