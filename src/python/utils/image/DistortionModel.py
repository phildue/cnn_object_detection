from abc import ABC, abstractmethod

from utils.image.Image import Image
from utils.labels.ImgLabel import ImgLabel


class DistortionModel(ABC):
    @abstractmethod
    def distort(self, img: Image, label: ImgLabel):
        pass

    @abstractmethod
    def undistort(self, img: Image, label: ImgLabel):
        pass
