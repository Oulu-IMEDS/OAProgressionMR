import random
import numpy as np
import math


class RandomCrop(object):
    def __init__(self, output_size, ndim):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size,) * ndim
        else:
            self.output_size = output_size
        self.ndim = ndim
        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Args:
            img: (ch, d0, ...) ndarray
            mask: (ch, d0, ...) ndarray

        Raises:
            ValueError: if input shape is less than the output size
        """
        ds_in = img.shape[1:]
        ds_out = self.output_size

        for d_in, d_out in zip(ds_in, ds_out):
            if d_in < d_out:
                msg = f"Invalid crop size {repr(ds_out)} for input {repr(ds_in)}"
                raise ValueError(msg)

        ds_start = [math.floor(self.state[f"ratio_d{idx}"] * (ds_in[idx] - ds_out[idx]))
                    for idx in range(self.ndim)]
        ds_end = [ds_start[idx] + ds_out[idx] for idx in range(self.ndim)]

        sel = [slice(ds_start[idx], ds_end[idx]) for idx in range(self.ndim)]
        sel = tuple([slice(None), ] + sel)
        img = np.ascontiguousarray(img[sel])
        if mask is not None:
            mask = np.ascontiguousarray(mask[sel])
            return img, mask
        else:
            return img

    def randomize(self):
        for idx in range(self.ndim):
            self.state[f"ratio_d{idx}"] = random.random()  # [0., 1.)


class CenterCrop(object):
    def __init__(self, output_size, ndim):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size,) * ndim
        else:
            self.output_size = output_size
        self.ndim = ndim

    def __call__(self, img, mask=None):
        """

        Args:
            img: (ch, d0, ...) ndarray
            mask: (ch, d0, ...) ndarray
        """
        ds_in = img.shape[1:]
        ds_out = self.output_size

        for d_in, d_out in zip(ds_in, ds_out):
            if d_in < d_out:
                msg = f"Invalid crop size {repr(ds_out)} for input {repr(ds_in)}"
                raise ValueError(msg)

        ds_offset = [(ds_in[idx] - ds_out[idx]) // 2
                     for idx in range(self.ndim)]

        sel = [slice(ds_offset[idx], ds_offset[idx] + ds_out[idx])
               for idx in range(self.ndim)]
        sel = tuple([slice(None), ] + sel)
        img = np.ascontiguousarray(img[sel])
        if mask is not None:
            mask = np.ascontiguousarray(mask[sel])
            return img, mask
        else:
            return img
