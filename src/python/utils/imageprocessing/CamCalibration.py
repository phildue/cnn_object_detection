# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import cv2
import numpy as np

from utils.fileaccess.utils import save_file, load_file


class CamCalibration:
    def __init__(self, img_shape, img_type='chess', grid=(12, 8)):

        self.img_type = img_type
        self.img_shape = img_shape
        self.grid = grid
        self.error, self.camera_mat, self.distortion, self.rotation_vectors, self.translation_vectors = None, None, None, None, None

    def calibrate(self, images):
        obj_points, img_points = self._find_points(images)
        if len(obj_points) > 0 and len(img_points) > 0:
            self.error, self.camera_mat, self.distortion, self.rotation_vectors, self.translation_vectors = self._get_cam_params(
                obj_points, img_points)
            return self.error
        else:
            print('Failure.No points found.')
            return np.inf

    def demo(self, img):
        mat = img.array
        h, w = mat.shape[:2]
        cam_mat_new, roi = cv2.getOptimalNewCameraMatrix(self.camera_mat, self.distortion, (w, h), 1, (w, h))

        dst = cv2.undistort(mat, self.camera_mat, self.distortion, None, cam_mat_new)
        cv2.imshow('calibresult', dst)

        # crop the image
        try:
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            cv2.imshow('roi', dst)
        except cv2.error:
            pass
        cv2.waitKey(0)

    def calc_estimation_error(self, obj_points, img_points):
        mean_error = 0
        for i in range(len(obj_points)):
            imgpoints2, _ = cv2.projectPoints(obj_points[i], self.rotation_vectors[i], self.translation_vectors[i],
                                              self.camera_mat, self.distortion)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        return mean_error

    def save(self, path):
        save_file(self, path)

    @staticmethod
    def load(path):
        return load_file(path)

    def _find_points(self, images):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.grid[0] * self.grid[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.grid[1], 0:self.grid[0]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        for img in images:
            gray = cv2.cvtColor(img.array, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('gray',gray)
            # cv2.waitKey(0)
            if self.img_type is 'chess':
                ret, corners = cv2.findChessboardCorners(gray, self.grid, None)
            else:
                ret, corners = cv2.findCirclesGrid(gray, self.grid, None)

            # If found, add object points, image points (after refining them)
            if ret:
                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img.array, self.grid, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        return obj_points, img_points

    def _get_cam_params(self, obj_points, img_points):
        return cv2.calibrateCamera(obj_points, img_points, self.img_shape[::-1], None, None)
