import random

from pycocotools.coco import COCO
from utils.image.imageprocessing import imread
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon
import numpy as np


class CocoGenerator:
    def __init__(self, path, super_categories=None, categories=None):

        self.categories = categories
        self.super_categories = super_categories
        self.path = path

    @staticmethod
    def convert_label(coco_label, y_max):
        object_labels = []
        for l in coco_label:
            bbox = np.array([l['bbox']])

            bbox[0, 0] += bbox[0, 2] / 2
            bbox[0, 1] += bbox[0, 3] / 2
            bbox[0, 1] = y_max - bbox[0, 1]

            obj_label = ObjectLabel((l['category_id']), 1.0,
                                    Polygon.from_quad_t_centroid(bbox))
            object_labels.append(obj_label)
        return ImgLabel(object_labels)

    def generate_valid(self, batch_size: int, n: int):
        return self._generate(batch_size, 'val', n, True)

    def generate_train(self, batch_size: int, n: int):
        return self._generate(batch_size, 'train', n, True)

    def _generate(self, batch_size: int, set_type: str, n: int, loop: bool):
        if set_type not in ['val', 'train']:
            raise ValueError('Unknown Type: ' + set_type)

        ann_file_path = '{}/annotations/instances_{}2017.json'.format(self.path, set_type)
        coco = COCO(ann_file_path)

        cat_ids = coco.getCatIds(catNms=self.categories)
        img_ids = coco.getImgIds(catIds=cat_ids)
        if n > -1:
            img_ids = img_ids[:n]

        random.shuffle(img_ids)
        while loop:
            for idx_batch in range(0, len(img_ids), batch_size):
                img_ids_batch = img_ids[idx_batch:idx_batch + batch_size]
                imgs_meta = coco.loadImgs(img_ids_batch)

                imgs = []
                labels = []
                for i, img_meta in enumerate(imgs_meta):
                    ann_ids = coco.getAnnIds(imgIds=img_ids_batch[i], catIds=cat_ids, iscrowd=None)
                    anns = coco.loadAnns(ann_ids)
                    img = imread('%s/images/%s2017/%s' % (self.path, set_type, img_meta['file_name']), 'bgr')
                    label = self.convert_label(anns, img.shape[0])
                    imgs.append(img)
                    labels.append(label)

                yield imgs, labels

            random.shuffle(img_ids)
