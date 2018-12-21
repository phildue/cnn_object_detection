import glob

from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing import Image
from utils.imageprocessing.Backend import resize, imread
from utils.labels.ImgLabel import ImgLabel
import numpy as np
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon


class YoloParser(AbstractDatasetParser):

    def __init__(self, directory: str, color_format, image_format='jpg', start_idx=0, img_norm=(416, 416)):
        super().__init__(directory, color_format, start_idx, image_format)
        self.img_norm = img_norm

    def read(self, n=0) -> ([Image], [ImgLabel]):
        files = sorted(glob.glob(self.directory + '/*.' + self.image_format))
        samples = []
        labels = []
        for i, file in enumerate(files):
            if n > 0 and 0 < n < i:
                break
            image = imread(file, self.color_format)

            if image is None:
                continue

            label = YoloParser.read_label(file.replace(self.image_format, 'txt'), self.img_norm)
            samples.append(image)
            labels.append(label)
        return samples, labels

    @staticmethod
    def read_label(filepath: str, img_norm=(416, 416)) -> [ImgLabel]:
        objs = []
        with open(filepath) as f:
            for line in f:
                fields = line.split(' ')
                class_id = int(fields[0])
                cx = float(fields[1])
                cy = float(fields[2])
                w = float(fields[3])
                h = float(fields[4])
                x_min = cx - w / 2
                x_max = cx + w / 2
                y_min = cy - h / 2
                y_max = cy + h / 2
                bounding_box = np.array([[x_min, y_min],
                                         [x_max, y_max]])
                bounding_box *= np.array(img_norm)
                bounding_box[:, 1] = img_norm[0] - bounding_box[:, 1]
                obj = ObjectLabel(ObjectLabel.id_to_name(class_id),1.0, Polygon.from_quad_t_centroid(bounding_box))
                objs.append(obj)

        return ImgLabel(objs)

    def write_label(self, label: ImgLabel, path: str):
        # [category number] [object center in X] [object center in Y] [object width in X] [object width in Y]
        with open(path + '.txt', "w+") as f:
            for obj in label.objects:
                cy = self.img_norm[0] - obj.poly.cy
                f.write('{} {} {} {} {}\n'.format(obj.class_id-1, obj.poly.cx / self.img_norm[1], cy / self.img_norm[0],
                                                  obj.poly.width / self.img_norm[1], obj.poly.height / self.img_norm[0]))

    def write(self, images: [Image], labels: [ImgLabel]):
        for i, l in enumerate(labels):
            filename = '{0:s}/{1:05d}'.format(self.directory, self.idx)
            img, label = resize(images[i], self.img_norm, label=l)

            if self.color_format == img.format:
                self.write_img(img, filename + '.' + self.image_format)
                self.write_label(label, filename)
                self.idx += 1
            else:
                raise ValueError('Color Format!')
