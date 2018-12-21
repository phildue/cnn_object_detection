import copy

from utils.imageprocessing.Backend import resize
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformSubsample(ImgTransform):
    def __init__(self, factor=0.5):
        self.factor = factor

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        label_aug = copy.deepcopy(label)

        img_aug, label_aug = resize(img_aug, label=label_aug, scale_y=self.factor, scale_x=self.factor)
        img_aug, label_aug = resize(img_aug, label=label_aug, scale_y=1 / self.factor, scale_x=1 / self.factor)

        return img_aug, label_aug
