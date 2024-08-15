import numpy as np
from fortracc_module.flow import TimeOrderedSequence
from fortracc_module.detectors import Detector
from typing import List, Optional
from fortracc_module.objects import GeoGrid


class lessThan(TimeOrderedSequence):
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        Class to see if image values are less than a specified threshold value.
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        self.event_name = 'less_than_threshold'
        masks = [image < threshold for image in images]
        super().__init__(masks, timestamps, grid)


class lessThanOrEqualTo(TimeOrderedSequence):
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        Class to see if image values are less than or equal to  a specified threshold value.
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        self.event_name = 'less_than_or_equal_to_threshold'
        masks = [image <= threshold for image in images]
        super().__init__(masks, timestamps, grid)


class greaterThan(TimeOrderedSequence):
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        Class to see if image values are greater than a specified threshold value.
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        self.event_name = 'greater_than_threshold'
        masks = [image > threshold for image in images]
        super().__init__(masks, timestamps, grid)


class greaterThanOrEqualTo(TimeOrderedSequence):
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        Class to see if image values are greater than a specified threshold value.
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        self.event_name = 'greater_than_or_equal_to_threshold'
        masks = [image >= threshold for image in images]
        super().__init__(masks, timestamps, grid)


class equalTo(TimeOrderedSequence):
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            threshold: float
    ):
        """
        Class to see if image values are equal to a specified threshold value.
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float threshold: The threshold below which a phenomenon is defined.  The threshold should have the
                                same units as the values in `images`.
        """
        self.event_name = 'equal_to_threshold'
        masks = [image == threshold for image in images]
        super().__init__(masks, timestamps, grid)


class anomalyEvent(TimeOrderedSequence):
    """
    Specific TimeOrderedSequence object that defines phenomenon based on a threshold that is calculated per pixel
    as the standard deviation.  The threshold is compared to the images minus the pixel wise mean.  Any pixel that is
    less than the threshold is treated as a part of the phenomenon.
    """
    def __init__(
            self,
            images: List[np.array],
            timestamps: List[str],
            grid: GeoGrid,
            frac_std: Optional[float] = 2.0
    ):
        """
        :param List[np.array] images: A list of images given as np.arrays.  Each pixel should have the same units
                                      as the provided threshold.
        :param List[str] timestamps: A list of timestamps with the same size as `masks` where each item is a
                                     string formatted as YYYYMMDDhhmm(e.g. 201501011430) giving the datetime of the
                                     corresponding mask.
        :param GeoGrid grid: A GeoGrid object that defines the geographical grid on which the masks live.
        :param float frac_std: The standard deviation you wish to investigate.  Set to 2.0 by default.

        """
        img_tensor = np.array(images)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)

        self.event_name = 'anomaly_event'
        masks = [(image - mu) < (frac_std * std) for image in images]
        super().__init__(masks, timestamps, grid)


class lessThanSparse(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    less than the provided threshold is treated as a part of the phenomenon.
    """
    name = "less_than_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold below which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image < self.threshold for image in images]

class lessThanOrEqualToSparse(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    less than or equal to the provided threshold is treated as a part of the phenomenon.
    """
    name = "less_than_or_equal_to_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold below which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image <= self.threshold for image in images]

class greaterThanSparse(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    greater than the provided threshold is treated as a part of the phenomenon.
    """
    name = "greater_than_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold above which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image > self.threshold for image in images]

class greaterThanOrEqualToSparse(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    greater than or equal to the provided threshold is treated as a part of the phenomenon.
    """
    name = "greater_than_or_equal_to_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold above which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image >= self.threshold for image in images]

class equalToSparse(Detector):
    """
    `Detector` which defines phenomenon based on a threshold.  Any pixel that is
    equal to the provided threshold is treated as a part of the phenomenon.
    """
    name = "equal_to_threshold"

    def __init__(
        self,
        threshold: float,
    ):
        """
        Parameters
        ----------
        threshold: float
            The threshold above which a phenomenon is defined.  The threshold should have the
            same units as the values in `images` passed to `create_masks`.
        """
        self.threshold = threshold
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        return [image == self.threshold for image in images]


class anomalyEventSparse(Detector):
    """
    `Detector` which defines phenomenon based on a threshold that is calculated per pixel
    as the standard deviation.  The threshold is compared to the images minus the pixel wise mean.  
    Any pixel that is less than the threshold is treated as a part of the phenomenon.
    """
    name = "anomaly_event"

    def __init__(
        self,
        frac_std: Optional[float] = 2.0,
    ):
        """
        Parameters
        ----------
        frac_std: Optional[float]
            Used as a multiplier to the pixel-wise standard deviation which defines the threshold
            below which a phenomenon is defined.
        """
        self.frac_std = frac_std
    
    def create_masks(
        self,
        images: List[np.array]
    ) -> List:
        img_tensor = np.array(images)
        mu = img_tensor.mean(0)
        std = img_tensor.std(0)

        return [(image - mu) < (self.frac_std * std) for image in images]
