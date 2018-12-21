from abc import ABC, abstractmethod

from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class ImgTransform(ABC):
    @abstractmethod
    def transform(self, img: Image, label: ImgLabel = ImgLabel([])):
        pass
