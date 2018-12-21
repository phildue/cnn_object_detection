import copy

from utils.labels.Polygon import Polygon
from utils.labels.Pose import Pose


class ObjectLabel:
    classes = []

    def __init__(self, name: str, confidence: float, poly: Polygon, pose: Pose = None):
        self.name = name
        self.confidence = confidence
        self.poly = poly
        self.pose = pose

    @staticmethod
    def name_to_id(name: str) -> int:
        try:
            return ObjectLabel.classes.index(name) + 1
        except ValueError:
            ObjectLabel.classes.append(name)
            print("Added class: gate")
        return ObjectLabel.classes.index(name) + 1

    @staticmethod
    def id_to_name(id: int) -> str:
        try:
            return ObjectLabel.classes[id - 1]
        except IndexError:
            return "Unknown"

    def __repr__(self):
        return '{0:s}: \t{1:s}'.format(self.name, str(self.poly))

    def copy(self):
        return copy.deepcopy(self)

    @property
    def class_id(self):
        return ObjectLabel.name_to_id(self.name)

