from utils.fileaccess.labelparser.PklParser import PklParser
from utils.fileaccess.labelparser.XmlParser import XmlParser
from utils.fileaccess.labelparser.YoloParser import YoloParser


class DatasetParser:

    @staticmethod
    def get_parser(directory: str, label_format, color_format, start_idx=0, image_format='jpg', img_norm=(416, 416)):
        if label_format is 'xml':
            return XmlParser(directory, color_format, start_idx, image_format)
        elif label_format is 'tf_record':
            from utils.fileaccess.labelparser.TfRecordParser import TfRecordParser
            return TfRecordParser(directory, color_format, start_idx, image_format)
        elif label_format is 'pkl':
            return PklParser(directory, color_format, start_idx, image_format)
        elif label_format is 'yolo':
            return YoloParser(directory=directory, color_format=color_format, start_idx=start_idx,
                              image_format=image_format, img_norm=img_norm)
        else:
            print("Label format not known!!")

    @staticmethod
    def read_label(label_format, filepath):
        if label_format is 'xml':
            return XmlParser.read_label(filepath)
        elif label_format is 'tf_record':
            from utils.fileaccess.labelparser.TfRecordParser import TfRecordParser
            return TfRecordParser.read_label(filepath)
        elif label_format is 'pkl':
            return PklParser.read_label(filepath)
        elif label_format is 'yolo':
            return YoloParser.read_label(filepath)
        else:
            print("Label format not known!!")
