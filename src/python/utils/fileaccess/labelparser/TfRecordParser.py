import os
import tensorflow as tf
from object_detection.utils import dataset_util

from utils.fileaccess.labelparser.AbstractDatasetParser import AbstractDatasetParser
from utils.imageprocessing.Backend import convert_color, COLOR_BGR2RGB
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
import numpy as np


class TfRecordParser(AbstractDatasetParser):

    def __init__(self, directory: str, color_format, start_idx=0, image_format='jpg'):

        super().__init__(directory, color_format, start_idx, image_format)
        flags = tf.app.flags
        flags.DEFINE_string('output_path', self.directory, 'Path to output TFRecord')
        self.FLAGS = flags.FLAGS

    def read(self, n=0, filename='set.record') -> ([Image], [ImgLabel]):
        record_iterator = tf.python_io.tf_record_iterator(path=self.directory + filename)
        images = []
        labels = []
        for string_record in record_iterator:

            if len(images) > n: break

            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = int(example.features.feature['image/height'].int64_list.value[0])

            width = int(example.features.feature['image/width'].int64_list.value[0])

            img_string = (example.features.feature['image/encoded'].bytes_list.value[0])

            # xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
            # xmaxs = example.features.feature['image/object/bbox/xmax'].float_list
            # ymins = example.features.feature['image/object/bbox/ymin'].float_list
            # ymaxs = example.features.feature['image/object/bbox/ymax'].float_list
            # names = example.features.feature['image/object/class/text'].bytes_list

            objects = []
            # for i, name in enumerate(names):
            #     object_label = ObjectLabel(name, tf.np.array([(xmins[i], ymins[i]), (xmaxs[i], ymaxs[i])]))
            #     objects.append(object_label)
            labels.append(ImgLabel(objects))

            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, -1))
            img = Image(reconstructed_img, 'rgb')
            img = convert_color(img, COLOR_BGR2RGB)
            img.color_format = 'bgr'
            images.append(img)
        return images, labels

    @staticmethod
    def read_label(filepath: str) -> [ImgLabel]:
        pass

    def write(self, images: [Image], labels: [ImgLabel], filename='/set.record'):
        writer = tf.python_io.TFRecordWriter(self.FLAGS.output_path + filename)
        i = 0
        for i in range(len(images)):
            filename = tf.compat.as_bytes('{}/{:05d}'.format(self.directory, self.idx))
            height, width = images[i].shape[:2]
            with tf.gfile.GFile(os.path.abspath(images[i].path), 'rb') as fid:
                encoded_image_data = fid.read()
            image_format = 'jpeg'.encode('utf8')  # tf.compat.as_bytes(self.image_format)

            xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
            xmaxs = []  # List of normalized right x coordinates in bounding box
            # (1 per box)
            ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
            ymaxs = []  # List of normalized bottom y coordinates in bounding box
            # (1 per box)
            classes_text = []  # List of string class name of bounding box (1 per box)
            classes = []  # List of integer class id of bounding box (1 per box)
            for obj in labels[i].objects:
                xmins.append(obj.x_min)
                xmaxs.append(obj.x_max)
                ymins.append(obj.y_min)
                ymaxs.append(obj.y_max)
                classes_text.append(tf.compat.as_bytes(obj.class_name))
                classes.append(obj.class_id)

            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_image_data),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            writer.write(tf_example.SerializeToString())
        print("{} samples written to {}".format(i, self.directory))
        writer.close()
