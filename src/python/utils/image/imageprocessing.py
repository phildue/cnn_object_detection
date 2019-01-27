import copy

import cv2
import numpy as np
from PIL.Image import frombytes, FLIP_TOP_BOTTOM
from utils.image.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.labels.ObjectLabel import ObjectLabel
from utils.labels.Pose import Pose
from utils.labels.utils import resize_label

LEGEND_BOX = 0
LEGEND_CORNERS = 1
LEGEND_TEXT = 2
LEGEND_POSITION = 3

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

COLOR_BGR2YUV = cv2.COLOR_BGR2YUV
COLOR_YUV2BGR = cv2.COLOR_YUV2BGR
COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
COLOR_RGBA2BGR = cv2.COLOR_RGBA2BGR
COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
COLOR_RGB2BGR = cv2.COLOR_RGB2BGR
COLOR_YUV2YUYV = 99
COLOR_YUYV2YUV = 100


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

            img_ann = draw_bounding_box(img_ann, (int(obj.poly.x_max), int(obj.poly.y_max)),
                                        (int(obj.poly.x_min), int(obj.poly.y_min)),
                                        color=color, thickness=thickness_obj)

        if legend >= LEGEND_TEXT:
            img_ann = annotate_text(obj.name + ' - ' + str(np.round(confidence, 2)) + ' - ' + str(
                np.round(obj.poly.area / (img.shape[0] * img.shape[1]), 2)), img_ann,
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


def imread(path, color_format) -> Image:
    return Image(cv2.imread(path), color_format, path)


def imwrite(img: Image, path):
    cv2.imwrite(path, img.array)


def parse_screenshot(pixels, width, height) -> Image:
    image = frombytes("RGB", (width, height), pixels)
    image = image.transpose(FLIP_TOP_BOTTOM)
    return Image(__pil2cv(image), 'bgr')


def __pil2cv(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()


def replace_background(img: Image, background: Image, background_color=(0, 0, 0)) -> Image:
    img_cv = img.array
    background_cv = background.array
    background_cv = cv2.resize(background_cv, (img_cv.shape[1], img_cv.shape[0]))
    merged = background_cv.copy()
    merged[img_cv != background_color] = img_cv[img_cv != background_color]
    return Image(merged, img.format)


def draw_gate_corners(img: Image, label: ObjectLabel) -> Image:
    annotated_img = img.array.copy()
    corners = label.poly.points.copy()

    corners[:, 1] = img.shape[0] - corners[:, 1]

    for i in range(corners.shape[0]):
        if 0 < corners[i, 0] < img.shape[1] and 0 < corners[i, 1] < img.shape[0]:
            cv2.circle(annotated_img, tuple(corners[i, :].astype(int)), 3,
                       (0, 0, 255), -1)

    return Image(annotated_img, img.format)


def flip_y(point: tuple, ymax) -> tuple:
    return point[0], ymax - point[1]


def draw_bounding_box(img: Image, p1: tuple, p2: tuple, color=(0, 255, 0), thickness=2) -> Image:
    img_bb = img.array.copy()
    cv2.rectangle(img_bb, flip_y(p1, img.shape[0]), flip_y(p2, img.shape[0]), color, thickness=thickness)

    return Image(img_bb, img.format)


def imshow(img: Image, name: str = "Img", t=0):
    cv2.imshow(name, img.array)
    cv2.waitKey(t)


def annotate_bounding_box(img: Image, label: ImgLabel, color=(0, 255, 0)) -> Image:
    img_ann = img.copy()

    for obj in [obj for obj in label.objects if obj is not None]:
        img_ann = draw_bounding_box(img_ann, (obj.x_min, obj.y_min), (obj.x_max, obj.y_max), color)
        cv2.putText(img_ann.array, obj.class_id,
                    flip_y((obj.x_max, obj.y_max), img_ann.shape[0]), 0,
                    1e-3 * img_ann.array.shape[0], color, 0)
    return img_ann


def annotate_text(text: str, img: Image, xy: tuple = (10, 10), color=(0, 255, 0), thickness=0):
    img_ann = img.copy()
    xy_flip = flip_y(xy, img_ann.shape[0])
    cv2.putText(img_ann.array, text,
                xy_flip, 0,
                1e-3 * img_ann.array.shape[0], color, thickness=thickness)
    return img_ann


def resize(img: Image, shape: tuple = None, scale_x=1.0, scale_y=1.0, label: ImgLabel = None) -> (Image, ImgLabel):
    if shape is None:
        shape_cv = (0, 0)
    else:
        shape_cv = shape[1], shape[0]

    img_resized = cv2.resize(img.array.copy(), shape_cv, fx=scale_x, fy=scale_y)

    if label is not None:
        label_resized = resize_label(label, img.array.shape, shape, scale_x, scale_y)
        return Image(img_resized, img.format), label_resized
    else:
        return Image(img_resized, img.format)


def pad_equal_letterbox(img: Image, label: ImgLabel = None):
    """
    Pad image such that aspect ratio is equal
    :param img: image
    :param label: label
    :return: padded image, label
    """
    h, w = img.shape[:2]

    target_size = max(img.shape)

    diff = np.max(target_size - h, 0), np.max(target_size - w, 0)
    return pad_letterbox(img, diff, label)


def pad_letterbox(img: Image, diff: tuple, label: ImgLabel = None):
    diff_top = int(np.ceil(diff[0] / 2))
    diff_bottom = int(np.floor(diff[0] / 2))
    diff_right = int(np.ceil(diff[1] / 2))
    diff_left = int(np.floor(diff[1] / 2))

    mat_letterbox = cv2.copyMakeBorder(img.array, top=diff_top, bottom=diff_bottom,
                                       right=diff_right, left=diff_left,
                                       borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))

    img_letterbox = Image(mat_letterbox, img.format)
    if label:
        for obj in label.objects:
            obj.poly.points[:, 1] += diff_bottom
            obj.poly.points[:, 0] += diff_left
        return img_letterbox, label
    else:
        return img_letterbox


def flip(img: Image, label: ImgLabel = None, flip_code=1) -> (Image, ImgLabel):
    flipped = cv2.flip(img.array.copy(), flip_code)

    if label is not None:
        objs_flipped = []
        for obj in label.objects:
            mat = obj.points
            mat[:, 0] = img.array.shape[1] - mat[:, 0]
            obj_flipped = obj.copy()
            objs_flipped.append(obj_flipped)
        label_flipped = ImgLabel(objs_flipped)
    else:
        label_flipped = None

    return Image(flipped, img.format), label_flipped


def translate(img: Image, shift_x, shift_y, label: ImgLabel) -> (Image, ImgLabel):
    content = img.array.copy()

    shift_y = 0 if shift_y < 0 else shift_y
    shift_y = img.shape[0] if shift_y > img.shape[0] else shift_y

    shift_x = 0 if shift_x < 0 else shift_x
    shift_x = img.shape[1] if shift_x > img.shape[1] else shift_x

    content = content[shift_y: (shift_y + img.shape[0]), shift_x: (shift_x + img.shape[1])]
    if label is not None:
        label_translated = copy.deepcopy(label)
        for obj in label_translated.objects:
            mat = obj.gate_corners.mat
            mat[:, 0] -= shift_x
            obj.points = mat
    else:
        label_translated = None
    return Image(content, img.format), label_translated


def color_shift(img, t, label=None) -> (Image, ImgLabel):
    content = img.array.astype(np.float64)
    content *= (1. + t)
    # content /= (255. * 2.)
    return Image(content.astype(np.uint8), img.format), copy.deepcopy(label)


def normalize(img: Image, min_range: float = 0.0, max_range: float = 1.0):
    return Image(
        cv2.normalize(img.array.astype('float'), None, alpha=min_range, beta=max_range, norm_type=cv2.NORM_MINMAX),
        img.format)


def blur(img: Image, size: (int, int), iterations=10):
    mat = img.array
    for i in range(iterations):
        mat = cv2.GaussianBlur(mat, size, 0.5)
    return Image(mat, img.format)


def noisy(img: Image, var=0.1, iterations=10):
    row, col, ch = img.array.shape
    mean = 0
    sigma = var ** 0.5
    noised = img.array
    for i in range(iterations):
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noised[:, :, 0][(noised[:, :, 0] < 255 - sigma * 4) & (noised[:, :, 0] > sigma * 4)] += gauss.astype(np.uint8)[
            (noised[:, :, 0] < 255 - sigma * 4) & (noised[:, :, 0] > sigma * 4)]
        noised[:, :, 1][(noised[:, :, 1] < 255 - sigma * 4) & (noised[:, :, 1] > sigma * 4)] += gauss.astype(np.uint8)[
            (noised[:, :, 1] < 255 - sigma * 4) & (noised[:, :, 1] > sigma * 4)]
        noised[:, :, 2][(noised[:, :, 2] < 255 - sigma * 4) & (noised[:, :, 2] > sigma * 4)] += gauss.astype(np.uint8)[
            (noised[:, :, 2] < 255 - sigma * 4) & (noised[:, :, 2] > sigma * 4)]
    return Image(noised, img.format)


def noisy_color(img: Image, var=0.1, iterations=10):
    row, col, ch = img.array.shape
    mean = 0
    sigma = var ** 0.5
    noised = img.array
    for i in range(iterations):
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noised[:, :, 0][(noised[:, :, 0] < 255 - sigma * 4) & (noised[:, :, 0] > sigma * 4)] += \
            gauss.astype(np.uint8)[
                (noised[:, :, 0] < 255 - sigma * 4) & (noised[:, :, 0] > sigma * 4), 0]
        noised[:, :, 1][(noised[:, :, 1] < 255 - sigma * 4) & (noised[:, :, 1] > sigma * 4)] += \
            gauss.astype(np.uint8)[
                (noised[:, :, 1] < 255 - sigma * 4) & (noised[:, :, 1] > sigma * 4), 1]
        noised[:, :, 2][(noised[:, :, 2] < 255 - sigma * 4) & (noised[:, :, 2] > sigma * 4)] += \
            gauss.astype(np.uint8)[
                (noised[:, :, 2] < 255 - sigma * 4) & (noised[:, :, 2] > sigma * 4), 2]
    return Image(noised, img.format)


def convert_color(img: Image, code):
    if code is COLOR_BGR2GRAY:
        if img.format is not 'bgr':
            raise ValueError('Wrong Format')
        img_array = cv2.cvtColor(img.array, code)
        img_array = np.expand_dims(img_array, -1)
        img_array = np.tile(img_array, (1, 1, 3))
        new_format = 'bgr'
    elif code is COLOR_BGR2YUV:
        if img.format is not 'bgr':
            raise ValueError('Wrong Format')
        img_array = cv2.cvtColor(img.array, code)
        new_format = 'yuv'
    elif code is COLOR_RGBA2BGR:
        if img.format is not 'rgba':
            raise ValueError('Wrong Format')
        img_array = cv2.cvtColor(img.array, code)
    elif code is COLOR_YUV2BGR:
        if img.format is not 'yuv':
            raise ValueError('Wrong Format')
        img_array = cv2.cvtColor(img.array, code)
        new_format = 'bgr'
    elif code is COLOR_YUV2YUYV:
        if img.format is not 'yuv':
            raise ValueError('Wrong Format')
        img_array = yuv2yuyv(img).array
        new_format = 'yuyv'
    elif code is COLOR_YUYV2YUV:
        if img.format is not 'yuyv':
            raise ValueError('Wrong Format')
        img_array = yuyv2yuv(img).array
        new_format = 'yuv'
    else:
        raise ValueError('Unknown Conversion')
    return Image(img_array, new_format)


def yuv2yuyv(img: Image):
    mat = img.array
    mat_yuyv = np.zeros((mat.shape[0], mat.shape[1], 2))
    mat_yuyv[:, :, 0] = mat[:, :, 0]
    for i in range(mat.shape[1]):
        if i % 2 == 0:
            mat_yuyv[:, i, 1] = mat[:, i, 1]
        else:
            mat_yuyv[:, i, 1] = mat[:, i, 2]
    return Image(mat_yuyv, 'yuyv', img.path)


def yuyv2yuv(img: Image):
    mat = img.array
    mat_yuv = np.zeros((mat.shape[0], mat.shape[1], 3), dtype=np.uint8)
    mat_yuv[:, :, 0] = mat[:, :, 0]
    for i in range(mat.shape[1] - 1):
        if i % 2 == 0:
            mat_yuv[:, i:i + 2, 1] = mat[:, i, 1:]
        else:
            mat_yuv[:, i:i + 2, 2] = mat[:, i, 1:]

    return Image(mat_yuv, 'yuv', img.path)


def histogram_eq(img: Image):
    """
    Perform histogram equalization on the input image.

    See https://en.wikipedia.org/wiki/Histogram_equalization.
    """

    img_eq = np.copy(img.array)

    # img_eq = cv2.cvtColor(img_eq, cv2.COLOR_RGB2HSV)

    img_eq[:, :, 0] = cv2.equalizeHist(img_eq[:, :, 0])
    img_eq[:, :, 1] = cv2.equalizeHist(img_eq[:, :, 1])
    img_eq[:, :, 2] = cv2.equalizeHist(img_eq[:, :, 2])

    # img_eq = cv2.cvtColor(img_eq, cv2.COLOR_HSV2RGB)

    return Image(img_eq, img.format)


def scale_hsv(img: Image, scale):
    """
    Randomly change the brightness of the input image.

    Protected against overflow.
    """
    if img.format is 'bgr':
        hsv = cv2.cvtColor(img.array, cv2.COLOR_RGB2HSV)
    elif img.format is 'yuv':
        rgb = cv2.cvtColor(img.array, cv2.COLOR_YUV2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    else:
        raise ValueError("Unknown Color format")

    img_transf = hsv * scale
    img_transf[img_transf > 255] = 255
    img_transf[img_transf < 0] = 0
    img_transf = img_transf.astype(np.uint8)

    if img.format is 'bgr':
        org = cv2.cvtColor(img_transf, cv2.COLOR_HSV2RGB)
    elif img.format is 'yuv':
        rgb = cv2.cvtColor(img_transf, cv2.COLOR_HSV2RGB)
        org = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    else:
        raise ValueError("Unknown Color format")

    return Image(org, img.format)


def crop(img: Image, min_xy=(0, 0), max_xy=None, label: ImgLabel = None):
    if max_xy is None:
        max_xy = img.shape[1], img.shape[0]

    x_min, x_max = min_xy[0], max_xy[0]
    y_min, y_max = min_xy[1], max_xy[1]

    y_max_cv, y_min_cv = img.shape[0] - min_xy[1], img.shape[0] - max_xy[1]

    img_crop = Image(img.array[int(y_min_cv):int(y_max_cv), int(x_min):int(x_max)].copy(), img.format)
    if label is None:
        return img_crop
    else:
        objs_crop = []
        for obj in label.objects:
            delta_x = int(x_min)
            delta_y = int(y_min)
            points = obj.poly.points.copy()

            points[:, 0] -= delta_x
            points[:, 1] -= delta_y

            # points[:, 0] = np.maximum(0, np.minimum(points[:, 0], img_crop.shape[1]))
            # points[:, 1] = np.maximum(0, np.minimum(points[:, 1], img_crop.shape[0]))
            obj_crop = obj.copy()
            obj_crop.poly.points = points

            # if obj_crop.poly.area >= 20 and (0.33 < obj_crop.poly.width / obj_crop.poly.height < 3):
            objs_crop.append(obj_crop)

        label_crop = label.copy()
        label_crop.objects = objs_crop
        return img_crop, label_crop


def split_video(src_file, out_dir):
    vidcap = cv2.VideoCapture(src_file)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(out_dir + "/frame{0:05d}.jpg".format(count), image)  # save frame as JPEG file
        count += 1


def chromatic_distortion(img: Image, scale: (float, float, float), t_x: (float, float, float),
                         t_y: (float, float, float), label: ImgLabel = None):
    transmat = np.zeros((2, 3, 3))
    for i in range(3):
        transmat[:, :, i] = np.array([[scale[i], 0, t_x[i]],
                                      [0, scale[i], t_y[i]]])

    mat = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        if img.format is not 'bgr':
            raise ValueError("Chromatic:: Wrong Color format!")

        mat[:, :, i] = cv2.warpAffine(src=img.array[:, :, i], M=transmat[:, :, i],
                                      dsize=(mat.shape[1], mat.shape[0]))

    if label:
        return Image(mat, img.format), label
    else:
        return Image(mat, img.format)


def change_exposure(img: Image, delta_exposure, contrast=1.0, label: ImgLabel = None):
    mat = img.array.copy()

    for i in range(3):
        exposure = np.log(255 / mat[:, :, i] - 1) / (-contrast)
        reexposure = exposure + delta_exposure
        reexposed = 255 / (1 + np.exp(-contrast * reexposure))
        mat[:, :, i] = reexposed

    if label:
        return Image(mat, img.format), label
    else:
        return Image(mat, img.format)
