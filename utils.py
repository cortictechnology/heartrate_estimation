import sys
import os
import numpy as np

COLOR = [(0, 153, 0), (234, 187, 105), (0, 0, 255), (80, 190, 168)]


class circularlist(object):
    def __init__(self, size, data=[]):
        """Initialization"""
        self.index = 0
        self.size = size
        self._data = list(data)[-size:]

    def append(self, value):
        """Append an element"""
        if len(self._data) == self.size:
            self._data[self.index] = value
        else:
            self._data.append(value)
        self.index = (self.index + 1) % self.size

    def __getitem__(self, key):
        """Get element by index, relative to the current index"""
        if len(self._data) == self.size:
            return self._data[(key + self.index) % self.size]
        else:
            return self._data[key]

    def __repr__(self):
        """Return string representation"""
        return self._data.__repr__() + " (" + str(len(self._data)) + " items)"

    def calc_average(self):
        num_data = len(self._data)
        if num_data == 0:
            return 0
        sum = 0
        for val in self._data:
            sum = sum + val
        return float(sum) / num_data


def frame_norm(frame, bbox):
    return (
        np.clip(np.array(bbox), 0, 1)
        * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]
    ).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
