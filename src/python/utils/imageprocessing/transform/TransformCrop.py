from utils.imageprocessing.Backend import crop
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformCrop(ImgTransform):

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def transform(self, img: Image, label: ImgLabel):
        """
        Crop the center part
        :param img:
        :param label:
        :return:
        """

        return crop(img, (self.x_min, self.y_min), (self.x_max, self.y_max), label)
