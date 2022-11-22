from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from cached_property import cached_property
from cv2 import aruco
from numpy.typing import NDArray

from zoloto.utils import cached_method

from .calibration import CalibrationParameters
from .coords import (
    CartesianCoordinates,
    Orientation,
    PixelCoordinates,
    SphericalCoordinates,
)
from .exceptions import MissingCalibrationsError
from .marker_type import MarkerType


class BaseMarker(ABC):
    def __init__(
        self, marker_id: int, corners: list[NDArray], size: int, marker_type: MarkerType
    ):
        self.__id = marker_id
        self._pixel_corners = corners
        self.__size = size
        self.__marker_type = marker_type

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} size={self.size} type={self.marker_type.name}>"

    @abstractmethod
    def _get_pose_vectors(self) -> tuple[NDArray, NDArray]:  # pragma: nocover
        raise NotImplementedError()

    @property  # noqa: A003
    def id(self) -> int:  # noqa: A003
        return self.__id

    @property
    def size(self) -> int:
        return self.__size

    @property
    def marker_type(self) -> MarkerType:
        return self.__marker_type

    @property
    def pixel_corners(self) -> list[PixelCoordinates]:
        return [
            PixelCoordinates(x=float(x), y=float(y)) for x, y in self._pixel_corners
        ]

    @cached_property
    def pixel_centre(self) -> PixelCoordinates:
        centre = np.mean(self._pixel_corners, axis=0)
        return PixelCoordinates(x=centre[0], y=centre[1])

    @cached_property
    def distance(self) -> int:
        return self.spherical.distance

    @cached_property
    def orientation(self) -> Orientation:
        return Orientation(*self._rvec)

    @cached_property
    def spherical(self) -> SphericalCoordinates:
        return SphericalCoordinates.from_tvec(*self._tvec.tolist())

    @property
    def cartesian(self) -> CartesianCoordinates:
        return CartesianCoordinates.from_tvec(*self._tvec.tolist())

    @property
    def _rvec(self) -> NDArray:
        return self._get_pose_vectors()[0]

    @property
    def _tvec(self) -> NDArray:
        return self._get_pose_vectors()[1]

    def as_dict(self) -> dict[str, Any]:
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
        corners: list[NDArray],
        size: int,
        marker_type: MarkerType,
        precalculated_vectors: tuple[NDArray, NDArray],
    ):
        super().__init__(marker_id, corners, size, marker_type)
        self.__precalculated_vectors = precalculated_vectors

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} size={self.size} type={self.marker_type.name} distance={self.distance}>"

    def _get_pose_vectors(self) -> tuple[NDArray, NDArray]:
        return self.__precalculated_vectors


class Marker(BaseMarker):
    def __init__(
        self,
        marker_id: int,
        corners: list[NDArray],
        size: int,
        marker_type: MarkerType,
        calibration_params: CalibrationParameters,
    ):
        super().__init__(marker_id, corners, size, marker_type)
        self.__calibration_params = calibration_params

    @cached_method
    def _get_pose_vectors(self) -> tuple[NDArray, NDArray]:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            [self._pixel_corners],
            self.size,
            self.__calibration_params.camera_matrix,
            self.__calibration_params.distance_coefficients,
        )
        return rvec[0][0], tvec[0][0]


class UncalibratedMarker(BaseMarker):
    def _get_pose_vectors(self) -> tuple[NDArray, NDArray]:
        raise MissingCalibrationsError()
