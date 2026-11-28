"""The FK8 liftgate hem in the rear-up projection, in metres (glTF axes).

The upper corners follow the P24 plan, the lower corners follow the body
line in Honda frame 28. The independent straight-on open-hatch photograph
locates the joint through each lamp and the width of the painted jamb.
"""
from functools import lru_cache

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates

AXIS = np.array([-1., .85, 0.]) / np.hypot(1., .85)
ACROSS = np.array([0., 0., 1.])
ALONG = np.cross(AXIS, ACROSS)
PIVOT = 'Pivot_Hatch_Assembly'
HALF_GAP = .0018

# Start on the centre of the lower hem and follow the right side up to the roof.
HALF_OUTLINE = [
    (0, -.914), (.215, -.913), (.350, -.908), (.402, -.903),
    (.426, -.894), (.455, -.872), (.493, -.842), (.528, -.812),
    (.548, -.785), (.564, -.744), (.585, -.679), (.610, -.601),
    (.634, -.535), (.655, -.482), (.666, -.442), (.668, -.413),
    (.663, -.379), (.644, -.305), (.622, -.166), (.598, -.03),
    (.588, .072), (.582, .145), (.574, .191), (.565, .218),
    (.554, .238), (.503, .247), (.417, .249), (.23, .247), (0, .246),
]


@lru_cache(maxsize=1)
def outline():
    half = np.array(HALF_OUTLINE)
    loop = np.vstack([half, half[-2::-1] * [-1, 1]])
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
    spline = CubicSpline(length, loop, bc_type='periodic')
    return spline(np.arange(0, length[-1], .002))


@lru_cache(maxsize=1)
def distance_map():
    step = .0005
    edge = outline()
    origin = edge.min(axis=0) - .02
    size = np.ceil((edge.max(axis=0) + .02 - origin) / step).astype(int) + 1
    mask = np.zeros(tuple(size[::-1]), np.uint8)
    cv2.fillPoly(mask, [np.round((edge-origin)/step*16).astype(np.int32)], 1, shift=4)
    def distance(side):
        return cv2.distanceTransform(side, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (distance(1-mask) - distance(mask)) * step, origin, step


def signed_distance(points):
    """Negative inside the hatch. Exclude the underside along the same rays."""
    uv = np.column_stack([points @ ACROSS, points @ ALONG])
    field, origin, step = distance_map()
    pixels = ((uv - origin) / step).T[::-1]
    d = map_coordinates(field, pixels, order=1, mode='constant', cval=.10)
    return np.maximum(d, 1.73 - .20*uv[:, 1] - points @ AXIS)
