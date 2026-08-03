import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from shapely.geometry import MultiPoint

from tos2ca.utils.helpers import (
    getOperatorClass,
    padTimestamps,
    gridPolygons,
    pushBox,
    timerange,
    convertLons,
)
from tos2ca.operators.inequalities import (
    lessThan,
    lessThanOrEqualTo,
    greaterThan,
    greaterThanOrEqualTo,
    equalTo,
    anomalyEvent,
    lessThanSparse,
    lessThanOrEqualToSparse,
    greaterThanSparse,
    greaterThanOrEqualToSparse,
    equalToSparse,
    anomalyEventSparse,
)


# ---------------------------------------------------------------------------
# getOperatorClass
# ---------------------------------------------------------------------------

class TestGetOperatorClass:
    @pytest.mark.parametrize("name,expected", [
        ("lessThan", lessThan),
        ("lessThanOrEqualTo", lessThanOrEqualTo),
        ("greaterThan", greaterThan),
        ("greaterThanOrEqualTo", greaterThanOrEqualTo),
        ("equalTo", equalTo),
        ("anomalyEvent", anomalyEvent),
        ("lessThanSparse", lessThanSparse),
        ("lessThanOrEqualToSparse", lessThanOrEqualToSparse),
        ("greaterThanSparse", greaterThanSparse),
        ("greaterThanOrEqualToSparse", greaterThanOrEqualToSparse),
        ("equalToSparse", equalToSparse),
        ("anomalyEventSparse", anomalyEventSparse),
    ])
    def test_returns_correct_class(self, name, expected):
        assert getOperatorClass(name) is expected

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            getOperatorClass("nonExistent")


# ---------------------------------------------------------------------------
# padTimestamps
# ---------------------------------------------------------------------------

class TestPadTimestamps:
    def _base_timestamps(self):
        return {
            "202001010000": ["mask_indices"],
            "202001020000": ["mask_indices"],
        }

    def test_prepend_only(self):
        ts = self._base_timestamps()
        result = padTimestamps(ts, {"units": "days", "quantity": 1}, first=True)
        keys = list(result.keys())
        assert keys[0] == "201912310000"
        assert keys[1:] == list(ts.keys())

    def test_append_only(self):
        ts = self._base_timestamps()
        result = padTimestamps(ts, {"units": "days", "quantity": 1}, last=True)
        keys = list(result.keys())
        assert keys[-1] == "202001030000"
        assert keys[:-1] == list(ts.keys())

    def test_prepend_and_append(self):
        ts = self._base_timestamps()
        result = padTimestamps(ts, {"units": "hours", "quantity": 6}, first=True, last=True)
        keys = list(result.keys())
        assert keys[0] == "201912311800"
        assert keys[-1] == "202001020600"
        assert keys[1:-1] == list(ts.keys())

    def test_minutes_interval(self):
        ts = {"202001010030": ["mask_indices"]}
        result = padTimestamps(ts, {"units": "minutes", "quantity": 30}, first=True, last=True)
        keys = list(result.keys())
        assert keys[0] == "202001010000"
        assert keys[-1] == "202001010100"

    def test_months_interval(self):
        ts = {"202003010000": ["mask_indices"]}
        result = padTimestamps(ts, {"units": "months", "quantity": 1}, first=True, last=True)
        keys = list(result.keys())
        assert keys[0] == "202002010000"
        assert keys[-1] == "202004010000"

    def test_neither_first_nor_last_returns_unchanged(self):
        ts = self._base_timestamps()
        result = padTimestamps(ts, {"units": "days", "quantity": 1})
        assert list(result.keys()) == list(ts.keys())

    def test_values_preserved(self):
        ts = {"202001010000": ["mask_indices"]}
        result = padTimestamps(ts, {"units": "hours", "quantity": 1}, first=True, last=True)
        for v in result.values():
            assert v == ["mask_indices"]


# ---------------------------------------------------------------------------
# gridPolygons
# ---------------------------------------------------------------------------

class TestGridPolygons:
    def test_returns_polygon(self):
        from shapely.geometry import Polygon
        poly = gridPolygons(0.0, 0.0, 0.5, 0.5)
        assert isinstance(poly, Polygon)

    def test_polygon_contains_centroid(self):
        from shapely.geometry import Point
        poly = gridPolygons(10.0, 20.0, 0.5, 0.5)
        # The centroid of the grid cell should be inside the polygon
        assert poly.contains(Point(20.0, 10.0))

    def test_polygon_has_five_coordinates(self):
        poly = gridPolygons(0.0, 0.0, 1.0, 1.0)
        # WKT POLYGON closes the ring so first == last; exterior coords = 5
        assert len(poly.exterior.coords) == 5


# ---------------------------------------------------------------------------
# pushBox
# ---------------------------------------------------------------------------

class TestPushBox:
    def _make_mp(self, points):
        return MultiPoint(points)

    def test_expands_bounds(self):
        mp = self._make_mp([(0.0, 0.0), (10.0, 10.0)])
        min_lon, min_lat, max_lon, max_lat = pushBox(2.0, mp)
        assert min_lon == pytest.approx(-2.0)
        assert min_lat == pytest.approx(-2.0)
        assert max_lon == pytest.approx(12.0)
        assert max_lat == pytest.approx(12.0)

    def test_zero_push_returns_original_bounds(self):
        mp = self._make_mp([(5.0, 3.0), (15.0, 8.0)])
        min_lon, min_lat, max_lon, max_lat = pushBox(0.0, mp)
        assert min_lon == pytest.approx(5.0)
        assert min_lat == pytest.approx(3.0)
        assert max_lon == pytest.approx(15.0)
        assert max_lat == pytest.approx(8.0)

    def test_single_point(self):
        mp = self._make_mp([(5.0, 5.0)])
        min_lon, min_lat, max_lon, max_lat = pushBox(1.0, mp)
        assert min_lon == pytest.approx(4.0)
        assert max_lon == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# timerange
# ---------------------------------------------------------------------------

class TestTimerange:
    def test_hourly_range(self):
        start = datetime(2020, 1, 1, 0, 0)
        end = datetime(2020, 1, 1, 3, 0)
        result = timerange(start, end, "h")
        assert len(result) == 4
        assert result[0] == datetime(2020, 1, 1, 0, 0)
        assert result[-1] == datetime(2020, 1, 1, 3, 0)

    def test_daily_range(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 5)
        result = timerange(start, end, "D")
        assert len(result) == 5

    def test_single_point_range(self):
        start = datetime(2020, 6, 15)
        end = datetime(2020, 6, 15)
        result = timerange(start, end, "D")
        assert result == [datetime(2020, 6, 15)]

    def test_returns_naive_datetimes(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        result = timerange(start, end, "h")
        for dt in result:
            assert dt.tzinfo is None

    def test_returns_list(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 3)
        result = timerange(start, end, "D")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# convertLons
# ---------------------------------------------------------------------------

class TestConvertLons:
    def test_negative_range(self):
        min_lon, max_lon = convertLons(-180, -90)
        assert min_lon >= 0
        assert max_lon <= 360

    def test_positive_range_unchanged(self):
        min_lon, max_lon = convertLons(0, 90)
        assert min_lon == pytest.approx(0.0)
        assert max_lon == pytest.approx(90.0)

    def test_180_maps_to_360(self):
        # When 180 is in the range it should convert to 360
        min_lon, max_lon = convertLons(170, 180)
        assert max_lon == pytest.approx(360.0)

    def test_cross_antimeridian(self):
        # -180 to 180 should span the full 0-360 range
        min_lon, max_lon = convertLons(-180, 180)
        assert min_lon == pytest.approx(0.0)
        assert max_lon == pytest.approx(360.0)

    def test_returns_tuple_of_two(self):
        result = convertLons(-10, 10)
        assert len(result) == 2
