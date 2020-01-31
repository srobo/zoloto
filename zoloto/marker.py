from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from cached_property import cached_property
from cv2 import aruco
from numpy import arctan2, linalg, ndarray

from zoloto.utils import cached_method, encode_as_json

from .calibration import CalibrationParameters
from .coords import Coordinates, Orientation, Spherical, ThreeDCoordinates
from .exceptions import MissingCalibrationsError


class BaseMarker(ABC):
    def __init__(self, marker_id: int, corners: List[ndarray], size: int):
        self._id = marker_id
        self._pixel_corners = corners
        self._size = size

    @abstractmethod
    def _get_pose_vectors(self) -> Tuple[ndarray, ndarray]:
        raise NotImplementedError()

    @property  # noqa: A003
    def id(self) -> int:
        return self._id

    @property
    def size(self) -> int:
        return self._size

    @property
    def pixel_corners(self) -> List[Coordinates]:
        return [Coordinates(x=float(x), y=float(y)) for x, y in self._pixel_corners]

    @cached_property
    def pixel_centre(self) -> Coordinates:
        tl, _, br, _ = self.pixel_corners
        return Coordinates(x=tl.x + (self._size / 2) - 1, y=br.y - (self._size / 2),)

    @cached_property
    def distance(self) -> int:
        return int(linalg.norm(self._tvec))

    @cached_property
    def orientation(self) -> Orientation:
        return Orientation(*self._rvec)

    @cached_property
    def spherical(self) -> Spherical:
        x, y, z = self._tvec
        return Spherical(
            rot_x=float(arctan2(y, z)), rot_y=float(arctan2(x, z)), dist=self.distance
        )

    @property
    def cartesian(self) -> ThreeDCoordinates:
        return ThreeDCoordinates(*self._tvec.tolist())

    @property
    def _rvec(self) -> ndarray:
        return self._get_pose_vectors()[0]

    @property
    def _tvec(self) -> ndarray:
        return self._get_pose_vectors()[1]

    def __json__(self) -> str:
        return encode_as_json(self.as_dict())

    def as_dict(self) -> Dict[str, Any]:
        marker_dict = {
            "id": self.id,
            "size": self.size,
            "pixel_corners": [list(corner) for corner in self.pixel_corners],
        }
        try:
            marker_dict.update(
                {"rvec": self._rvec.tolist(), "tvec": self._tvec.tolist()}
            )
        except MissingCalibrationsError:
            pass
        return marker_dict


class EagerMarker(BaseMarker):
    def __init__(
        self,
        marker_id: int,
        corners: List[ndarray],
        size: int,
        precalculated_vectors: Tuple[ndarray, ndarray],
    ):
        super().__init__(marker_id, corners, size)
        self.__precalculated_vectors = precalculated_vectors

    def _get_pose_vectors(self) -> Tuple[ndarray, ndarray]:
        return self.__precalculated_vectors


class Marker(BaseMarker):
    def __init__(
        self,
        marker_id: int,
        corners: List[ndarray],
        size: int,
        calibration_params: Optional[CalibrationParameters],
    ):
        super().__init__(marker_id, corners, size)
        self.__calibration_params = calibration_params

    @cached_method
    def _get_pose_vectors(self) -> Tuple[ndarray, ndarray]:
        if self.__calibration_params is None:
            raise MissingCalibrationsError()

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            [self._pixel_corners],
            self._size,
            self.__calibration_params.camera_matrix,
            self.__calibration_params.distance_coefficients,
        )
        return rvec[0][0], tvec[0][0]
