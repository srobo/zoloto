from cached_property import cached_property
from cv2 import aruco, moments
from numpy import linalg

from .calibration import CalibrationParameters
from .coords import Coordinates, Orientation, ThreeDCoordinates


class Marker:
    def __init__(
        self,
        id: int,
        corners,
        size: int,
        calibration_params: CalibrationParameters,
        precalculated_vectors=None,
    ):
        self.__id = id
        self.__pixel_corners = corners
        self.__size = size
        self.__camera_calibration_params = calibration_params
        self.__precalculated_vectors = precalculated_vectors

    @property
    def id(self):
        return self.__id

    @property
    def size(self):
        return self.__size

    def _is_eager(self):
        return bool(self.__precalculated_vectors)

    @cached_property
    def pixel_corners(self):
        return [Coordinates(*coords) for coords in self.__pixel_corners]

    @cached_property
    def pixel_centre(self):
        moment = moments(self.__pixel_corners)
        return Coordinates(
            int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"])
        )

    @cached_property
    def distance(self):
        return int(linalg.norm(self.tvec))

    @cached_property
    def orientation(self):
        return Orientation(*self.rvec)

    @cached_property
    def cartesian(self):
        return ThreeDCoordinates(*self.tvec)

    @cached_property
    def __vectors(self):
        if self.__precalculated_vectors is not None:
            return self.__precalculated_vectors

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            [self.__pixel_corners], self.__size, *self.__camera_calibration_params
        )
        return rvec[0][0], tvec[0][0]

    @cached_property
    def rvec(self):
        rvec, _ = self.__vectors
        return rvec

    @cached_property
    def tvec(self):
        _, tvec = self.__vectors
        return tvec
