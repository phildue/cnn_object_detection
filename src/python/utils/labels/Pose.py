import math
import numpy as np
from math import sqrt


class Pose:
    METER_2_SCENE_UNIT = 1.167

    def __init__(self, north=0.0, east=0.0, up=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.pitch = pitch
        self.roll = roll
        self.up = up
        self.east = east
        self.north = north
        self.yaw = yaw

    def __repr__(self):
        return "|North:{0:.3f}|East:{1:.3f}|Up:{2:.3f}|\nRoll:{3:.3f}|Pitch:{4:.3f}|Yaw:{5:.3f}|".format(
            self.north,
            self.east,
            self.up, self.roll,
            self.pitch,
            self.yaw)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.pitch == other.pitch and \
               self.roll == other.roll and \
               self.yaw == other.yaw and \
               self.north == other.north and \
               self.east == other.east and \
               self.up == other.up

    def __add__(self, other):
        return Pose(north=self.north + other.north,
                    east=self.east + other.east,
                    up=self.up + other.up,
                    yaw=self.yaw + other.yaw,
                    roll=self.roll + other.roll,
                    pitch=self.pitch + other.pitch, )

    def __sub__(self, other):
        return Pose(north=self.north - other.north,
                    east=self.east - other.east,
                    up=self.up - other.up,
                    yaw=self.yaw - other.yaw,
                    roll=self.roll - other.roll,
                    pitch=self.pitch - other.pitch, )

    @property
    def rotmat_pitch(self):
        return np.array([[1, 0, 0],
                         [0, np.cos(self.pitch), -np.sin(self.pitch)],
                         [0, np.sin(self.pitch), np.cos(self.pitch)]
                         ])

    @property
    def rotmat_roll(self):
        return np.array([[np.cos(self.roll), -np.sin(self.roll), 0],
                         [np.sin(self.roll), np.cos(self.roll), 0],
                         [0, 0, 1]
                         ])

    @property
    def rotmat_yaw(self):
        return np.array([[np.cos(self.yaw), 0, np.sin(self.yaw)],
                         [0, 1, 0],
                         [-np.sin(self.yaw), 0, np.cos(self.yaw)]
                         ])

    @property
    def transfmat(self):
        transformation_mat = np.eye(4)
        transformation_mat[:3, :3] = self.rotmat
        transformation_mat[:3, 3] = self.transvec
        return transformation_mat

    @property
    def transvec(self):
        return np.array([self.east, self.up, self.north]).T

    @transvec.setter
    def transvec(self, translation):
        self.east = translation[0]
        self.up = translation[1]
        self.north = translation[2]

    @property
    def rotmat(self):
        return self.rotmat_pitch.dot(self.rotmat_yaw).dot(self.rotmat_roll)

    @property
    def to_scene_unit(self):
        return Pose(self.north * self.METER_2_SCENE_UNIT,
                    self.east * self.METER_2_SCENE_UNIT,
                    self.up * self.METER_2_SCENE_UNIT,
                    self.roll,
                    self.pitch,
                    self.yaw)

    @property
    def to_meters(self):
        return Pose(self.north / self.METER_2_SCENE_UNIT,
                    self.east / self.METER_2_SCENE_UNIT,
                    self.up / self.METER_2_SCENE_UNIT,
                    self.roll,
                    self.pitch,
                    self.yaw)

    @property
    def magnitude(self):
        return sqrt(self.north ** 2 + self.east ** 2 + self.up ** 2)

    @staticmethod
    def rotmat2euler(mat, cy_thresh=None):
        """ Discover Euler angle vector from 3x3 matrix

        Uses the conventions above.

        Parameters
        ----------
        mat : array-like, shape (3,3)
        cy_thresh : None or scalar, optional
           threshold below which to give up on straightforward arctan for
           estimating x rotation.  If None (default), estimate from
           precision of input.

        Returns
        -------
        z : scalar
        y : scalar
        x : scalar
           Rotations in radians around z, y, x axes, respectively

        Notes
        -----
        If there was no numerical error, the routine could be derived using
        Sympy expression for z then y then x rotation matrix, which is::

          [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
          [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
          [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

        with the obvious derivations for z, y, and x

           z = atan2(-r12, r11)
           y = asin(r13)
           x = atan2(-r23, r33)

        Problems arise when cos(y) is close to zero, because both of::

           z = atan2(cos(y)*sin(z), cos(y)*cos(z))
           x = atan2(cos(y)*sin(x), cos(x)*cos(y))

        will be close to atan2(0, 0), and highly unstable.

        The ``cy`` fix for numerical instability below is from: *Graphics
        Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
        0123361559.  Specifically it comes from EulerAngles.c by Ken
        Shoemake, and deals with the case where cos(y) is close to zero:

        See: http://www.graphicsgems.org/

        The code appears to be licensed (from the website) as "can be used
        without restrictions".
        """
        mat = np.asarray(mat)
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(mat.dtype).eps * 4
            except ValueError:
                cy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = mat.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
        return z, y, x
