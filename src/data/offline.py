import os
import sys
import numpy as np
import soundfile as sf

from typing import Any, Callable

import numpy as np

from dataclasses import dataclass

from src.features.signal import lofar

from frozenlist import FrozenList
from functools import partial

def load_raw_data(input_db_path, run_pattern=None, verbose=0):
    """
        Loads sonar audio datafiles on memory. 

        This function returns a nested hashmap associating each run audio data with its
        class and filename. The audio information is an instance of MetaArray class, an extension
        of a numpy.ndarray storing metada information. 
        In this case the metadata is composed by the frames stored in a 
        numpy array, the file informed sample rate and the respective class and run name (filename).
        
        E.g. for database '4classes' the returned dictionary will be set like:
        
        ClassA:
            navio10.wav: 
                signal: np.array
                sample_rate: np.float64
            navio11.wav: 
                signal: np.array
                sample_rate: np.float64
        ClassB:
            navio20.wav: 
                ...
            navio21.wav:
                ...
            ...
        ...
            
        params:
            input_data_path (string): 
                path to database folder
        return (SonarTree): 
                nested dicionary in which the basic unit contains
                a record of the audio (signal key) in np.array format
                and the sample_rate (fs key) stored in floating point. 
                The returned object also contains a method for applying
                functions over the runs (see SonarTree.apply).
                the map is made associating each tuple to the corresponding
                name of the run (e.g. )
    """

    if verbose:
        print('Reading Raw data in path %s' % input_db_path)

    class_folders = [folder for folder in os.listdir(input_db_path)
                        if not folder.startswith('.')]
    
    raw_data = list()
    raw_metadata = list()
    for cls_folder in class_folders:
        runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
        if not runfiles:  # No files found inside the class folder
            if verbose:
                print('Empty directory %s' % cls_folder)
            continue
        if verbose:
            print('Reading %s' % cls_folder)

        runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
        runfiles = sorted(runfiles)

        runpaths = [os.path.join(input_db_path, cls_folder, runfile)
                    for runfile in runfiles]

        runfiles = [runfile.replace('.wav', '') for runfile in runfiles]

        audio_data = [read_audio_file(runpath) for runpath in runpaths]
        
        if run_pattern is not None:
            import re
            for i, runfile in enumerate(runfiles):
                m = re.search(run_pattern, runfile)
                if not m:
                    raise ValueError(f"Run Pattern was passed but no match found for {i}th run {runfile}")
                runfiles[i] = m.group(0)


        for (audio, sr), runfile in zip(audio_data, runfiles):
            run = AudioRun(audio, sr, cls_folder, runfile)
            raw_data.append(run)

        # raw_data[cls_folder] = {runfile: signal
        #                         for runfile, (signal, _)  in zip(runfiles, audio_data) }
        # raw_metadata[cls_folder] = {runfile: sr
        #                             for runfile, (_, sr)  in zip(runfiles, audio_data) }
        
    return FrozenVector(raw_data, AudioRun)#, raw_metadata


def read_audio_file(filepath):
    signal, fs = sf.read(filepath, always_2d=True)
  
    return signal, fs


@dataclass(frozen=True)
class Run:
    data: np.ndarray
    sr: int
    ship_class: int
    id: int


@dataclass(frozen=True)
class AudioRun(Run):
    def lofar(self, final_sr, *args, **kwargs):
        sxx, freq, time = lofar(self.data, self.sr, final_sr=final_sr, channel_axis=1, *args, **kwargs)
        return SpecRun(data=sxx, sr=final_sr, ship_class=self.ship_class, 
                       id=self.id, time=time, freq=freq)


@dataclass(frozen=True)
class SpecRun(Run):
    time: np.ndarray
    freq: np.ndarray

    def labels(self, attr):
        size = len(self.time)
        return np.repeat(getattr(self, attr), size)

    def cepstrum(self, nfft):
        sxx = self.data
        
        h, w, c = sxx.shape
        cep = np.zeros((h, w//2 + 1, c))
        for i in range(sxx.shape[-1]):
            cep[:,:, i] = np.fft.rfft(sxx[:,:, i], axis=1)/nfft
            
        df = self.freq[1] - self.freq[0]
        quefrency_vector = np.fft.rfftfreq(sxx.shape[1], df)
            
        cep = np.absolute(cep)
        return SpecRun(data=cep, sr=self.sr, ship_class=self.ship_class, 
                       id=self.id, time=self.time, freq=quefrency_vector)
    
    def liftering(self, nfft, cut, high=True):
        sxx = self.data
        
        h, w, c = sxx.shape
        cep = np.zeros((h, w//2 + 1, c))
        for i in range(sxx.shape[-1]):
            cep[:,:, i] = np.fft.rfft(sxx[:,:, i], axis=1)/nfft
            
        if high:
            cep[:, :cut, :] = 0
        else:
            cep[:, cut:, :] = 0
            
        for i in range(sxx.shape[-1]):
            sxx[:,:,i] = np.fft.irfft(cep[:,:,i]*nfft, n=sxx.shape[1], axis=1)
            
        return SpecRun(data=sxx, sr=self.sr, ship_class=self.ship_class, 
                       id=self.id, time=self.time, freq=self.freq)


class FrozenVector(FrozenList):
    def __init__(self, iterable, base) -> None:
        super().__init__(iterable)
        self.__base = base

        for elem in self:
            if not isinstance(elem, self.__base):
                raise TypeError("Wrong type")
        self.freeze()

    def __repr__(self) -> str:
        s = super().__repr__()
        return s.replace("FrozenList", f"FrozenVector[{self.__base}]")

    def call(self, method, *args, **kwargs):
        newvalues = [getattr(elem, method)(*args, **kwargs) for elem in self]
        newbase = type(newvalues[0])
        return FrozenVector(newvalues, newbase)


    def get(self, attr):
        newvalues = [getattr(elem, attr) for elem in self]
        newbase = type(newvalues[0])
        return FrozenVector(newvalues, newbase)


    # def __getattr__(self, attr, *args, **kwargs):
    #     if hasattr(self, attr):
    #         return self.__dict__[attr]

    #     if not hasattr(self[0], attr):
    #         raise AttributeError()

    #     _attr = getattr(self[0], attr)
    #     if callable(_attr):
    #         def attr_vectorized(*args, **kwargs):
    #             return self.call(attr, *args, **kwargs)
    #         return attr_vectorized
    #     return self.get(attr)


    def map(self, fn, *args, **kwargs):
        pfn = partial(fn, *args, **kwargs)
        newvalues = list(map(pfn, self))
        newbase = newvalues[0]
        return FrozenVector(newvalues, newbase)
