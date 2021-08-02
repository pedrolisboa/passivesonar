from _typeshed import NoneType
from typing import Union
import xarray as xr
import numpy as np

from dataclasses import dataclass

class AudioArray(xr.DataArray):
    def __init__(self, signal:np.ndarray, 
                 sample_rate: int,
                 attrs=None, channel_axis:int = -1):
        if signal.ndim != 2:
            raise ValueError("Wrong dim")
        if channel_axis not in [0, 1, -1]:
            raise ValueError("Invalid value for channel_axis")

        signal = np.swapaxes(signal, channel_axis, -1)

        if attrs is None:
            attrs = dict()
        attrs["sr"] = sample_rate

        super().__init__(signal, dims=["frame", "channel"], attrs=attrs)

    @property
    def sr(self):
        return self.attrs["sr"]



class TimeFreqArray(xr.DataArray):
    def __init__(self, sxx, freq, time,  
                 nfft, noverlap, sr, attrs=None):
        if sxx.ndim == 3:
            dims = ["freq", "time", "channel"]
        elif sxx.ndim == 2:
            dims = ["freq", "time"]
        else:
            raise ValueError("Wrong dim")

        super(TimeFreqArray, self).__init__(data=sxx, coords=[freq, time], dims=dims, attrs=attrs)

        if attrs is None:
            attrs = dict()
        attrs.update({
            "nfft": nfft,
            "noverlap": noverlap,
            "sr": sr
        })

    @property
    def nfft(self):
        return self.attrs["nfft"]

    @property
    def noverlap(self):
        return self.attrs["noverlap"]

    @property
    def sr(self):
        return self.attrs["sr"]
