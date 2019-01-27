import glob
import xml.etree.ElementTree as ET

import numpy as np
from utils.image import Image
from utils.image.imageprocessing import imwrite, imread
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Polygon import Polygon
from utils.labels.Pose import Pose


class XmlParser:

    def __init__(self, directory: str, color_format, start_idx=0, image_format='jpg'):
        self.color_format = color_format
        self.image_format = image_format
        self.directory = directory
        self.idx = start_idx

    def write(self, images: [Image], labels: [ImgLabel]):
        for i, l in enumerate(labels):
            filename = '{}/{:05d}'.format(self.directory, self.idx)
            self.write_img(images[i], filename + '.' + self.image_format)
            self.write_label(l, filename)
            self.idx += 1

    def write_img(self, image: Image, path: str):
        imwrite(image, path)

    def write_label(self, label: ImgLabel, path: str):
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = path.replace('xml', self.image_format)
        # TODO extend this with color format and other informations about the dataset
        for obj in label.objects:
            obj_root = ET.SubElement(root, 'object')
            ET.SubElement(obj_root, 'name').text = '{0:s}'.format(obj.name)
            bnd_box = ET.SubElement(obj_root, 'bndbox')
            x1 = obj.poly.x_min
            x2 = obj.poly.x_max
            y1 = obj.poly.y_min
            y2 = obj.poly.y_max
            xmin = min((x1, x2))
            xmax = max((x1, x2))
            ymin = min((y1, y2))
            ymax = max((y1, y2))
            ET.SubElement(bnd_box, 'xmin').text = '{0:d}'.format(int(xmin))
            ET.SubElement(bnd_box, 'xmax').text = '{0:d}'.format(int(xmax))
            ET.SubElement(bnd_box, 'ymin').text = '{0:d}'.format(int(ymin))
            ET.SubElement(bnd_box, 'ymax').text = '{0:d}'.format(int(ymax))

            if obj.pose is not None:
                pose_root = ET.SubElement(obj_root, 'pose')
                ET.SubElement(pose_root, 'north').text = '{0:03f}'.format(obj.pose.north)
                ET.SubElement(pose_root, 'east').text = '{0:03f}'.format(obj.pose.east)
                ET.SubElement(pose_root, 'up').text = '{0:03f}'.format(obj.pose.up)
                ET.SubElement(pose_root, 'yaw').text = '{0:03f}'.format(obj.pose.yaw)
                ET.SubElement(pose_root, 'pitch').text = '{0:03f}'.format(obj.pose.pitch)
                ET.SubElement(pose_root, 'roll').text = '{0:03f}'.format(obj.pose.roll)

            corner_root = ET.SubElement(obj_root, 'corners')
            ET.SubElement(corner_root, 'top_left').text = '{},{}'.format(
                int(obj.poly.points[3, 0]),
                int(obj.poly.points[3, 1]))
            ET.SubElement(corner_root, 'top_right').text = '{},{}'.format(
                int(obj.poly.points[2, 0]),
                int(obj.poly.points[2, 1]))
            ET.SubElement(corner_root, 'bottom_left').text = '{},{}'.format(
                int(obj.poly.points[0, 0]),
                int(obj.poly.points[0, 1]))
            ET.SubElement(corner_root, 'bottom_right').text = '{},{}'.format(
                int(obj.poly.points[1, 0]),
                int(obj.poly.points[1, 1]))

        tree = ET.ElementTree(root)
        tree.write(path + '.xml')

    def read(self, n=0) -> ([Image], [ImgLabel]):
        files = sorted(glob.glob(self.directory + '/*.xml'))
        samples = []
        labels = []
        for i, file in enumerate(files):
            if n > 0 and 0 < n < i: break
            label = XmlParser.read_label(file)
            if label is None: continue
            image = imread(file.replace('xml', self.image_format), self.color_format)
            samples.append(image)
            labels.append(label)
        return samples, labels

    @staticmethod
    def _parse_gate_corners(gate_corners_xml: str) -> Polygon:
        top_left = tuple([int(e) for e in gate_corners_xml.find('top_left').text.split(',')])
        top_right = tuple([int(e) for e in gate_corners_xml.find('top_right').text.split(',')])
        bottom_left = tuple([int(e) for e in gate_corners_xml.find('bottom_left').text.split(',')])
        bottom_right = tuple([int(e) for e in gate_corners_xml.find('bottom_right').text.split(',')])

        gate_corners = Polygon(np.array([[bottom_left[0], bottom_left[1]],
                                         [bottom_right[0], bottom_right[1]],
                                         [top_right[0], top_right[1]],
                                         [top_left[0], top_left[1]]]))
        return gate_corners

    @staticmethod
    def _parse_pose(pose_xml: str) -> Pose:
        north = float(pose_xml.find('north').text)
        east = float(pose_xml.find('east').text)
        up = float(pose_xml.find('down').text)
        yaw = float(pose_xml.find('yaw').text)
        pitch = float(pose_xml.find('pitch').text)
        roll = float(pose_xml.find('roll').text)

        pose = Pose(north, east, up, roll, pitch, yaw)
        return pose

    @staticmethod
    def _parse_bndbox(bndbox_xml: str) -> Polygon:
        x1 = int(np.round(float(bndbox_xml.find('xmin').text)))
        y1 = int(np.round(float(bndbox_xml.find('ymin').text)))
        x2 = int(np.round(float(bndbox_xml.find('xmax').text)))
        y2 = int(np.round(float(bndbox_xml.find('ymax').text)))
        x_min = min((x1, x2))
        x_max = max((x1, x2))
        y_min = min((y1, y2))
        y_max = max((y1, y2))
        poly = Polygon.from_quad_t_minmax(np.array([[x_min, y_min, x_max, y_max]]))
        return poly

    @staticmethod
    def read_label(path: str) -> [ImgLabel]:
        with open(path, 'rb') as f:
            try:
                tree = ET.parse(f)
                objects = []
                for element in tree.findall('object'):
                    name = element.find('name').text
                    try:
                        confidence_field = element.find('conf').text
                        confidence = float(confidence_field)
                    except AttributeError:
                        confidence = 1.0
                    try:
                        pose_xml = element.find('pose')
                        pose = XmlParser._parse_pose(pose_xml)
                    except AttributeError:
                        pose = None

                    try:
                        gate_corners_xml = element.find('gate_corners')
                        if gate_corners_xml is None:
                            gate_corners_xml = element.find('corners')
                        poly = XmlParser._parse_gate_corners(gate_corners_xml)

                    except AttributeError:
                        box = element.find('bndbox')
                        poly = XmlParser._parse_bndbox(box)



                    label = ObjectLabel(name, confidence,poly, pose)
                    objects.append(label)

                return ImgLabel(objects)
            except ET.ParseError:
                print('Error parsing: ' + path)
