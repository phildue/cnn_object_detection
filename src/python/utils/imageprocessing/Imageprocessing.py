import numpy as np

from utils.imageprocessing.Backend import draw_gate_corners, draw_bounding_box, annotate_text, imshow, imwrite
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.Pose import Pose

LEGEND_BOX = 0
LEGEND_CORNERS = 1
LEGEND_TEXT = 2
LEGEND_POSITION = 3

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)


def annotate_gate(img: Image, label: ImgLabel, bounding_box=False) -> Image:
    img_ann = img.copy()

    for obj in label.objects:
        if obj is not None:
            img_ann = draw_gate_corners(img_ann, obj)
            if bounding_box:
                img_ann = draw_bounding_box(img_ann, (int(obj.x_max), int(obj.y_max)), (int(obj.x_min), int(obj.y_min)))
    return img_ann


def annotate_position(img: Image, pos: Pose, x, y, color=(0, 255, 0)):
    img_ann = img.copy()
    lookup = [pos.north, pos.east, pos.up, np.degrees(pos.roll), np.degrees(pos.pitch), np.degrees(pos.yaw)]
    for i in range(len(lookup)):
        img_ann = annotate_text("{0:0.2f}".format(lookup[i]), img_ann,
                                (int(x), int(y - (i + 1) * 20)), color)
    return img_ann


def annotate_label(img: Image, label: ImgLabel, color=None, legend=LEGEND_POSITION, thickness=None) -> Image:
    img_ann = img.copy()

    if color is None:
        color = (0, 255, 0)

    for obj in label.objects:
        if obj is None: continue
        confidence = obj.confidence

        if legend >= LEGEND_CORNERS:
            img_ann = draw_gate_corners(img_ann, obj)
        if legend >= LEGEND_POSITION:
            if obj.pose:
                img_ann = annotate_position(img_ann, obj.pose, obj.poly.x_min, obj.poly.y_max + 5, color)
        if legend >= LEGEND_BOX:
            if thickness is None:
                thickness_obj = int(np.ceil(confidence * 4))
            else:
                thickness_obj = thickness

            img_ann = draw_bounding_box(img_ann, (int(obj.poly.x_max), int(obj.poly.y_max)), (int(obj.poly.x_min), int(obj.poly.y_min)),
                                        color=color, thickness=thickness_obj)

        if legend >= LEGEND_TEXT:
            img_ann = annotate_text(obj.name + ' - ' + str(np.round(confidence, 2)) + ' - ' + str(np.round(obj.poly.area/(img.shape[0]*img.shape[1]),2)), img_ann,
                                    (int(obj.poly.x_min), int(obj.poly.y_max + 5)),
                                    color)
    return img_ann


def annotate_labels(img: Image, labels: [ImgLabel], colors=None, legend=LEGEND_POSITION, thickness=None):
    img_ann = img.copy()
    for i, l in enumerate(labels):
        if colors is not None:
            color = colors[i]
        else:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0),
                      (255, 255, 255), (0, 0, 0)]
            color = colors[i % len(colors)]
        try:
            img_ann = annotate_label(img_ann, l, color, legend, thickness)
        except OverflowError:
            print("Label: Overflow error")

    return img_ann


def show(img: Image, name: str = "Img", t=0, labels=None, colors=None, legend=LEGEND_POSITION, thickness=None):
    img_ann = img.copy()

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]
        img_ann = annotate_labels(img_ann, labels, colors, legend, thickness)

    imshow(img_ann, name, t)

def save_labeled(img: Image, filename: str = "Img", labels=None, colors=None, legend=LEGEND_POSITION):
    img_ann = img.copy()
    if isinstance(labels, list):
        img_ann = annotate_labels(img_ann, labels, colors, legend)
    elif isinstance(labels, ImgLabel):
        img_ann = annotate_label(img_ann, labels, colors, legend)

    imwrite(img_ann, filename + '.jpg')
