from utils.imageprocessing.Backend import convert_color, COLOR_BGR2YUV, COLOR_YUV2YUYV
from utils.imageprocessing.Image import Image
from utils.imageprocessing.transform.ImgTransform import ImgTransform
from utils.labels.ImgLabel import ImgLabel


class TransformRaw(ImgTransform):

    def transform(self, img: Image, label: ImgLabel):
        img_aug = img.copy()
        img_yuv = convert_color(img_aug, COLOR_BGR2YUV)
        img_aug = convert_color(img_yuv, COLOR_YUV2YUYV)
        return img_aug, label.copy()
