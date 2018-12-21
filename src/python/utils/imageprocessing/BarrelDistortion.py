import cv2
import numpy as np
from cv2.cv2 import remap

from utils.fileaccess.utils import load_file, save_file
from utils.imageprocessing.DistortionModel import DistortionModel
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel
from utils.timing import tic


class BarrelDistortion(DistortionModel):
    @staticmethod
    def from_file(mapping_file: str):
        return load_file(mapping_file)

    def save(self, filename='barrel_dist.pkl'):
        save_file(self, filename, './')

    def __init__(self, img_shape, rad_dist_params, squeeze=1.0, tangential_dist_params=(0, 0),
                 max_iterations=100, distortion_radius=1.0, center=None, conv_thresh=0.01):
        """
        Barrel Distortion model.
        [Vass, G., & Perlaki, T. (n.d.). Applying and removing lens distortion in post production.
        Retrieved from http://www.vassg.hu/pdf/vass_gg_2003_lo.pdf]

        :param img_shape: original image shape
        :param rad_dist_params: parameters for radial distortion
        :param squeeze: parameter for squeeze effect
        :param tangential_dist_params: parameters for non-radial distortion (yshift, xshift)
        :param distortion_radius: radius on which distortion should be applied
        :param center: center around which distortion is applied
        :param conv_thresh: convergence threshold for newton approximation
        :param max_iterations: maximum iterations for newton approximation
        """
        self.img_shape = img_shape
        self.epsilon = conv_thresh
        self.center = center
        self.distortion_radius = distortion_radius
        self.non_rad_dist_params = tangential_dist_params
        self.max_iterations = max_iterations
        self.squeeze = squeeze
        self.rad_dist_params = rad_dist_params
        self.map_u, self.map_d = None, None

    def undistort(self, img: Image, label: ImgLabel = None, scale=1.0):
        """
        Remove distortion from an image
        :param img: image
        :param label: label
        :return: img,label without distortion
        """
        mat = img.array
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])
        if self.map_d is None:
            self.map_d = self._create_dmapping()

        map_scaled = self._scale(self.map_d, center, 1 / scale)

        mat_undistorted = self._apply_mapping(mat, map_scaled)
        if label is not None:
            label_undistorted = self._undistort_label(label, scale)
            return Image(mat_undistorted, img.format), label_undistorted
        else:
            return Image(mat_undistorted, img.format)

    def distort(self, img: Image, label: ImgLabel = None, scale=1.0):
        """
        Apply distortion to an image
        :param img: image
        :param label:  label
        :return: img, label with distortion
        """
        mat = img.array
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])

        if self.map_u is None:
            self.map_u = self._create_umapping()

        map_scaled = self._scale(self.map_u, center, 1 / scale)

        mat_undistorted = self._apply_mapping(mat, map_scaled)
        if label is not None:
            label_distorted = self._distort_label(label, scale)
            return Image(mat_undistorted, img.format), label_distorted
        else:
            return Image(mat_undistorted, img.format)

    def _scale(self, map, center, scale):
        map = map - center
        map *= scale
        map += center
        return map

    def _undistort_label(self, label: ImgLabel, scale=1.0):
        objects_u = []
        for obj in label.objects:
            corners = obj.poly.points
            corners = np.expand_dims(corners, 0)
            corners_norm = self._normalize(corners.astype(np.float))
            corners_norm *= scale
            corners_u = self._model(corners_norm)
            corners_u = self._denormalize(corners_u)[0]
            obj_u = obj.copy()
            obj_u.poly.points = corners_u

            if (len(corners_u[(corners_u[:, 0] < 0) | (corners_u[:, 0] > self.img_shape[1])]) +
                len(corners_u[(corners_u[:, 1] < 0) | (corners_u[:, 1] > self.img_shape[0])])) > 2:
                continue
            else:
                objects_u.append(obj_u)

        return ImgLabel(objects_u)

    def _distort_label(self, label: ImgLabel, scale=1.0):
        """
        Move the coordinates of the label according to distortion.
        :param label: image label
        :param mapping: mapping containing new pixel coordinates given the old coordinates
        :return: distorted label
        """

        objects_distorted = []
        for obj in label.objects:
            corners = obj.poly.points
            corners = np.expand_dims(corners, 0)
            corners_norm = self._normalize(corners.astype(np.float))
            corners_norm *= scale
            corners_d = self._inverse_model_approx(corners_norm, np.zeros_like(corners))
            corners_d = self._denormalize(corners_d)[0]

            obj_d = obj.copy()
            obj_d.points = corners_d

            if (len(corners_d[(corners_d[:, 0] < 0) | (corners_d[:, 0] > self.img_shape[1])]) +
                len(corners_d[(corners_d[:, 1] < 0) | (corners_d[:, 1] > self.img_shape[0])])) > 2:
                continue
            else:
                objects_distorted.append(obj_d)

        return ImgLabel(objects_distorted)

    def _create_umapping(self):
        """
        Creates mapping to remove distortion.
        :return:  mapping to remove distortion
        """
        h, w = self.img_shape
        x = np.tile(np.arange(0, w), (h, 1))
        y = np.tile(np.reshape(np.arange(0, h), (h, 1)), (1, w))
        coords = np.concatenate((np.expand_dims(x, -1), np.expand_dims(y, -1)), -1)
        coords_norm = self._normalize(coords.astype(np.float))
        map_u = self._model(coords_norm)
        map_u = self._denormalize(map_u)
        return map_u

    def _create_dmapping(self):
        """
        Creates mapping to apply distortion.
        :return:  mapping to apply distortion
        """
        h, w = self.img_shape
        x = np.tile(np.arange(0, w), (h, 1))
        y = np.tile(np.reshape(np.arange(0, h), (h, 1)), (1, w))
        coords = np.concatenate((np.expand_dims(x, -1), np.expand_dims(y, -1)), -1)
        coords_norm = self._normalize(coords.astype(np.float))
        map_d = self._inverse_model_approx(coords_norm, np.zeros_like(coords_norm))
        map_d = self._denormalize(map_d)
        return map_d

    def _model(self, coord: np.array):
        """
        Model for distortion.:
        :param coord: distorted pixel coordinates
        :return: undistored pixel coordinates
        """
        k_1, k_2 = self.rad_dist_params
        l_y, l_x = self.non_rad_dist_params
        s = self.squeeze
        x = coord[:, :, 0]
        y = coord[:, :, 1]

        x_u = x * (1 + k_1 * x ** 2 + k_1 * (1 + l_x) * y ** 2 + k_2 * (x ** 2 + y ** 2) ** 2)
        y_u = y * (1 + k_1 / s * x ** 2 + k_1 / s * (1 + l_y) * y ** 2 + k_2 / s * (x ** 2 + y ** 2) ** 2)
        mat_u = np.concatenate((np.expand_dims(x_u, -1), np.expand_dims(y_u, -1)), -1)

        return mat_u

    def _normalize(self, xy: np.array):
        """
        Normalizes the pixel coordinates so that the origin is center and the radius is distortion_radius.
        :param xy: coordinate matrix
        :return: normalize coordinate matrix
        """
        h, w = self.img_shape[:2]

        center = self.center if self.center is not None else np.array([w, h]) / 2

        h /= self.distortion_radius
        w /= self.distortion_radius

        mat_dimension_less = (xy - center) / np.linalg.norm(np.array([w, h]) / 2)
        return mat_dimension_less

    def _denormalize(self, coord: np.array):
        """
        Scales the normalized pixel coordinates back to original size.
        :param coord: normalized pixel coordinates
        :return: denormalized coordinates
        """
        h, w = self.img_shape[:2]
        center = self.center if self.center is not None else np.array([w, h]) / 2

        h /= self.distortion_radius
        w /= self.distortion_radius

        mat_denorm = coord * np.linalg.norm(np.array([w, h]) / 2) + center

        return mat_denorm

    @staticmethod
    def _bilinear_interp(mat: np.array, x, y):
        """
        Approximates f(x,y) using f(0,0),f(0,1),f(1,0),f(1,1) that is
        the new image given the 4 neighbouring pixels of the source image
        :param mat: source image
        :param x: source x
        :param y: source y
        :return: interpolated new pixel
        """
        channels = mat.shape[-1]
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = int(np.ceil(x))
        y1 = int(np.ceil(y))
        f00 = mat[y0, x0]
        f01 = mat[y1, x0]
        f10 = mat[y0, x1]
        f11 = mat[y1, x1]
        f = np.array([[f00, f01],
                      [f10, f11]])

        y_ = y - float(y0)
        x_ = x - float(x0)
        y_vec = np.array([[1 - y_], [y_]])
        x_vec = np.array([[1 - x_, x_]])
        out = np.zeros((channels,), dtype=np.int8)
        for i in range(channels):
            out[i] = (x_vec.dot(f[:, :, i].dot(y_vec))).astype(np.int8)

        return out.T

    def _apply_mapping(self, mat: np.array, mapping: np.array):
        """
        Distort the image with bilinear interpolation.
        :param mat: image
        :param mapping: mapping containing the old indeces given the new coordinates
        :return: distorted image
        """
        mat_remaped = remap(mat, mapping[:, :, 0].astype(np.float32), mapping[:, :, 1].astype(np.float32),
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return mat_remaped

    def _inverse_model_approx(self, coord: np.array, init):
        """
        Approximates the inverted model to apply distortion with newton method
        :param coord: pixel coordinates
        :param init: initial guess for mapping
        :return: approximated pixel coordinates after distortion
        """
        h, w = coord.shape[:2]
        xy = init.copy().astype(np.float)
        for t in range(self.max_iterations):
            tic()
            xy_prev = xy.copy()
            f = self._model(xy_prev)
            df = self._gradient(xy_prev)
            for i in range(h):
                for j in range(w):
                    df0 = df[i, j]
                    np.fill_diagonal(df0, np.diag(df0) + 0.000001)
                    xy_prev0 = np.reshape(xy_prev[i, j], (2, 1))

                    diff = np.reshape((f[i, j] - coord[i, j]), (2, 1))

                    update = xy_prev0 - np.linalg.inv(df0).dot(diff)
                    xy[i, j] = update.flatten()
            delta = np.linalg.norm(xy - xy_prev)
            if delta < self.epsilon:
                break
            # toc('iteration: {} - delta: {} - time: '.format(t, np.round(delta, 2)))
        return xy

    def _gradient(self, coord: np.array):
        """
        Gradient of distortion model
        :param coord: pixel coordinates
        :return: value of gradient
        """
        k_1, k_2 = self.rad_dist_params
        l_y, l_x = self.non_rad_dist_params
        s = self.squeeze

        x = coord[:, :, 0]
        y = coord[:, :, 1]
        h, w = coord.shape[:2]

        dx_x = k_1 * x ** 2 + k_1 * y ** 2 * (l_x + 1) + k_2 * (x ** 2 + y ** 2) ** 2 + x * (
                2 * k_1 * x + 4 * k_2 * x * (x ** 2 + y ** 2)) + 1
        dx_y = x * (2 * k_1 * y * (l_x + 1) + 4 * k_2 * y * (x ** 2 + y ** 2))

        dy_x = y * (2 * k_1 * x / s + 4 * k_2 * x * (x ** 2 + y ** 2) / s)
        dy_y = k_1 * x ** 2 / s + k_1 * y ** 2 * (l_y + 1) / s + k_2 * (x ** 2 + y ** 2) ** 2 / s + y * (
                2 * k_1 * y * (l_y + 1) / s + 4 * k_2 * y * (x ** 2 + y ** 2) / s) + 1

        gradient = np.zeros((h, w, 2, 2))
        gradient[:, :, 0, 0] = dx_x
        gradient[:, :, 0, 1] = dx_y
        gradient[:, :, 1, 0] = dy_x
        gradient[:, :, 1, 1] = dy_y
        return gradient

    def __repr__(self):
        return 'Shape: {}\n' \
               'Eps: {}\n' \
               'Center: {}\n' \
               'DistRadius: {}\n' \
               'NonRad Params: {}\n' \
               'MaxIterations: {}\n' \
               'Squeeze: {}\n' \
               'RadDist Params: {}\n'.format(
            self.img_shape,
            self.epsilon,
            self.center,
            self.distortion_radius,
            self.non_rad_dist_params,
            self.max_iterations,
            self.squeeze,
            self.rad_dist_params)
