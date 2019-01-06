import copy

from utils.labels.Polygon import Polygon
from utils.labels.Pose import Pose


class ObjectLabel:
    class_names = []

    def __init__(self, class_id: int, confidence: float, poly: Polygon, pose: Pose = None):
        self.class_id = class_id
        self.confidence = confidence
        self.poly = poly
        self.pose = pose

    def __repr__(self):
        return '{0:s}: \t{1:s}'.format(self.name, str(self.poly))

    def copy(self):
        return copy.deepcopy(self)

    @property
    def name(self) -> str:
        if len(self.class_names) == 0:
            return str(self.class_id)
        else:
            try:
                return self.class_names[self.class_id]
            except IndexError:
                print('Unknown Class Id')
                return "Unknown"
