import numpy as np
import pytest
from unittest.mock import patch, MagicMock

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
# Helpers
# ---------------------------------------------------------------------------

TIMESTAMPS = ["202001010000", "202001010030"]
GRID = MagicMock()


def _make_images():
    return [
        np.array([[1.0, 5.0], [10.0, 3.0]]),
        np.array([[7.0, 2.0], [4.0, 8.0]]),
    ]


def _dense(cls, images, threshold):
    """Instantiate a dense operator, bypassing TimeOrderedSequence.__init__."""
    with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__", return_value=None):
        obj = cls(images, TIMESTAMPS, GRID, threshold)
    return obj


# ---------------------------------------------------------------------------
# Dense operators — verify masks produced before super().__init__ is called
# ---------------------------------------------------------------------------

class TestLessThan:
    def test_event_name(self):
        obj = _dense(lessThan, _make_images(), 5.0)
        assert obj.event_name == "less_than_threshold"

    def test_masks_correct(self):
        images = _make_images()
        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            lessThan(images, TIMESTAMPS, GRID, 5.0)
            masks_arg = mock_init.call_args[0][0]
        expected = [image < 5.0 for image in images]
        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)


class TestLessThanOrEqualTo:
    def test_event_name(self):
        obj = _dense(lessThanOrEqualTo, _make_images(), 5.0)
        assert obj.event_name == "less_than_or_equal_to_threshold"

    def test_masks_correct(self):
        images = _make_images()
        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            lessThanOrEqualTo(images, TIMESTAMPS, GRID, 5.0)
            masks_arg = mock_init.call_args[0][0]
        expected = [image <= 5.0 for image in images]
        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)


class TestGreaterThan:
    def test_event_name(self):
        obj = _dense(greaterThan, _make_images(), 5.0)
        assert obj.event_name == "greater_than_threshold"

    def test_masks_correct(self):
        images = _make_images()
        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            greaterThan(images, TIMESTAMPS, GRID, 5.0)
            masks_arg = mock_init.call_args[0][0]
        expected = [image > 5.0 for image in images]
        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)


class TestGreaterThanOrEqualTo:
    def test_event_name(self):
        obj = _dense(greaterThanOrEqualTo, _make_images(), 5.0)
        assert obj.event_name == "greater_than_or_equal_to_threshold"

    def test_masks_correct(self):
        images = _make_images()
        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            greaterThanOrEqualTo(images, TIMESTAMPS, GRID, 5.0)
            masks_arg = mock_init.call_args[0][0]
        expected = [image >= 5.0 for image in images]
        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)


class TestEqualTo:
    def test_event_name(self):
        obj = _dense(equalTo, _make_images(), 5.0)
        assert obj.event_name == "equal_to_threshold"

    def test_masks_correct(self):
        images = [np.array([[5.0, 3.0]]), np.array([[1.0, 5.0]])]
        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            equalTo(images, TIMESTAMPS[:1] + TIMESTAMPS[:1], GRID, 5.0)
            masks_arg = mock_init.call_args[0][0]
        expected = [image == 5.0 for image in images]
        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)


class TestAnomalyEvent:
    def test_event_name(self):
        images = _make_images()
        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__", return_value=None):
            obj = anomalyEvent(images, TIMESTAMPS, GRID)
        assert obj.event_name == "anomaly_event"

    def test_masks_use_per_pixel_stats(self):
        images = _make_images()
        img_tensor = np.array(images)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)
        frac_std = 2.0
        expected = [(img - mu) < (frac_std * std) for img in images]

        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            anomalyEvent(images, TIMESTAMPS, GRID, frac_std=frac_std)
            masks_arg = mock_init.call_args[0][0]

        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)

    def test_custom_frac_std(self):
        images = _make_images()
        img_tensor = np.array(images)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)
        frac_std = 1.0
        expected = [(img - mu) < (frac_std * std) for img in images]

        with patch("tos2ca.operators.inequalities.TimeOrderedSequence.__init__") as mock_init:
            mock_init.return_value = None
            anomalyEvent(images, TIMESTAMPS, GRID, frac_std=frac_std)
            masks_arg = mock_init.call_args[0][0]

        for got, exp in zip(masks_arg, expected):
            np.testing.assert_array_equal(got, exp)


# ---------------------------------------------------------------------------
# Sparse operators — create_masks is pure and requires no patching
# ---------------------------------------------------------------------------

SPARSE_IMAGES = [
    np.array([[1.0, 5.0], [10.0, 3.0]]),
    np.array([[7.0, 2.0], [4.0, 8.0]]),
]


class TestLessThanSparse:
    def test_name(self):
        assert lessThanSparse.name == "less_than_threshold"

    def test_create_masks(self):
        detector = lessThanSparse(threshold=5.0)
        masks = detector.create_masks(SPARSE_IMAGES)
        for got, img in zip(masks, SPARSE_IMAGES):
            np.testing.assert_array_equal(got, img < 5.0)

    def test_all_false_when_threshold_below_min(self):
        detector = lessThanSparse(threshold=0.0)
        masks = detector.create_masks(SPARSE_IMAGES)
        assert not any(m.any() for m in masks)


class TestLessThanOrEqualToSparse:
    def test_name(self):
        assert lessThanOrEqualToSparse.name == "less_than_or_equal_to_threshold"

    def test_create_masks(self):
        detector = lessThanOrEqualToSparse(threshold=5.0)
        masks = detector.create_masks(SPARSE_IMAGES)
        for got, img in zip(masks, SPARSE_IMAGES):
            np.testing.assert_array_equal(got, img <= 5.0)

    def test_boundary_value_included(self):
        images = [np.array([[5.0]])]
        detector = lessThanOrEqualToSparse(threshold=5.0)
        assert detector.create_masks(images)[0].all()


class TestGreaterThanSparse:
    def test_name(self):
        assert greaterThanSparse.name == "greater_than_threshold"

    def test_create_masks(self):
        detector = greaterThanSparse(threshold=5.0)
        masks = detector.create_masks(SPARSE_IMAGES)
        for got, img in zip(masks, SPARSE_IMAGES):
            np.testing.assert_array_equal(got, img > 5.0)

    def test_all_false_when_threshold_above_max(self):
        detector = greaterThanSparse(threshold=100.0)
        masks = detector.create_masks(SPARSE_IMAGES)
        assert not any(m.any() for m in masks)


class TestGreaterThanOrEqualToSparse:
    def test_name(self):
        assert greaterThanOrEqualToSparse.name == "greater_than_or_equal_to_threshold"

    def test_create_masks(self):
        detector = greaterThanOrEqualToSparse(threshold=5.0)
        masks = detector.create_masks(SPARSE_IMAGES)
        for got, img in zip(masks, SPARSE_IMAGES):
            np.testing.assert_array_equal(got, img >= 5.0)

    def test_boundary_value_included(self):
        images = [np.array([[5.0]])]
        detector = greaterThanOrEqualToSparse(threshold=5.0)
        assert detector.create_masks(images)[0].all()


class TestEqualToSparse:
    def test_name(self):
        assert equalToSparse.name == "equal_to_threshold"

    def test_create_masks(self):
        images = [np.array([[5.0, 3.0]]), np.array([[1.0, 5.0]])]
        detector = equalToSparse(threshold=5.0)
        masks = detector.create_masks(images)
        for got, img in zip(masks, images):
            np.testing.assert_array_equal(got, img == 5.0)

    def test_no_match(self):
        images = [np.array([[1.0, 2.0]])]
        detector = equalToSparse(threshold=99.0)
        assert not detector.create_masks(images)[0].any()


class TestAnomalyEventSparse:
    def test_name(self):
        assert anomalyEventSparse.name == "anomaly_event"

    def test_create_masks_default_frac_std(self):
        detector = anomalyEventSparse()
        masks = detector.create_masks(SPARSE_IMAGES)
        img_tensor = np.array(SPARSE_IMAGES)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)
        expected = [(img - mu) < (2.0 * std) for img in SPARSE_IMAGES]
        for got, exp in zip(masks, expected):
            np.testing.assert_array_equal(got, exp)

    def test_create_masks_custom_frac_std(self):
        frac_std = 0.5
        detector = anomalyEventSparse(frac_std=frac_std)
        masks = detector.create_masks(SPARSE_IMAGES)
        img_tensor = np.array(SPARSE_IMAGES)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)
        expected = [(img - mu) < (frac_std * std) for img in SPARSE_IMAGES]
        for got, exp in zip(masks, expected):
            np.testing.assert_array_equal(got, exp)

    def test_returns_one_mask_per_image(self):
        detector = anomalyEventSparse()
        masks = detector.create_masks(SPARSE_IMAGES)
        assert len(masks) == len(SPARSE_IMAGES)
