import numpy as np
import pytest

from tos2ca.utils.plot import get_anno_coords


def _make_grid(size=10):
    lat = np.linspace(-5, 5, size)
    lon = np.linspace(-5, 5, size)
    return lat, lon


class TestGetAnnoCoords:
    def test_returns_two_dicts(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        mask[2:5, 2:5] = 1
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        assert isinstance(line_coords, dict)
        assert isinstance(bbox, dict)

    def test_keys_match_series_ids(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        mask[1:3, 1:3] = 1
        mask[6:8, 6:8] = 2
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        assert set(line_coords.keys()) == {1, 2}
        assert set(bbox.keys()) == {1, 2}

    def test_no_events_returns_empty_dicts(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        assert line_coords == {}
        assert bbox == {}

    def test_line_coords_structure(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        mask[3:6, 3:6] = 1
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        # Each value should be a tuple of two lists: ([x_c, x_b], [y_c, y_b])
        for key, val in line_coords.items():
            assert len(val) == 2
            x_vals, y_vals = val
            assert len(x_vals) == 2
            assert len(y_vals) == 2

    def test_bbox_structure(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        mask[2:5, 2:5] = 3
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        for key, val in bbox.items():
            assert len(val) == 2  # (x, y) tuple

    def test_centroid_within_grid_bounds(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        mask[3:7, 3:7] = 1
        line_coords, _ = get_anno_coords(lat, lon, mask)
        x_c, y_c = line_coords[1][0][0], line_coords[1][1][0]
        assert lon.min() <= x_c <= lon.max()
        assert lat.min() <= y_c <= lat.max()

    def test_single_pixel_event(self):
        lat, lon = _make_grid()
        mask = np.zeros((10, 10))
        mask[5, 5] = 1
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        assert 1 in line_coords
        assert 1 in bbox

    def test_multiple_non_contiguous_events(self):
        lat, lon = _make_grid(20)
        mask = np.zeros((20, 20))
        mask[0:3, 0:3] = 1
        mask[10:13, 10:13] = 2
        mask[16:19, 16:19] = 3
        line_coords, bbox = get_anno_coords(lat, lon, mask)
        assert set(line_coords.keys()) == {1, 2, 3}
