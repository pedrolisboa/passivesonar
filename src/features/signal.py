from __future__ import division
from numpy import convolve
from scipy.signal import decimate, hanning, spectrogram, lfilter, cheby2, butter, cheb2ord, hilbert
from librosa import stft, fft_frequencies, frames_to_time
import numpy as np
import math

from fractions import Fraction

import xarray as xr

import scipy
from xarray.core.dataarray import DataArray


def tpsw(signal, npts=None, n=None, p=None, a=None):
    x = np.copy(signal)
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=int(round(npts*.04/2.0+1))
    if p is None:
        p =int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
    else:
        h = np.ones((1, 2*n+1))
        p = 1
    h /= np.linalg.norm(h, 1)

    def apply_on_spectre(xs):
        c= convolve(h, xs, mode='full')
        return c
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    ix = int(np.floor((h.shape[0] + 1)/2.0)) # Defasagem do filtro
    mx = mx[ix-1:npts+ix-1] # Corrige da defasagem
    # Corrige os pontos extremos do espectro
    ixp = ix - p
    mult=2*ixp/np.concatenate([np.ones(p-1)*ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis] # Correcao dos pontos extremos
    mx[:ix,:] = mx[:ix,:]*(np.matmul(mult, np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*np.matmul(np.flipud(mult),np.ones((1, x.shape[1]))) # Pontos finais
    #return mx
    # Elimina picos para a segunda etapa da filtragem
    #indl= np.where((x-a*mx) > 0) # Pontos maiores que a*mx
    indl = (x-a*mx) > 0
    #x[indl] = mx[indl]
    x = np.where(indl, mx, x)
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    mx=mx[ix-1:npts+ix-1,:]
    # Corrige pontos extremos do espectro
    mx[:ix,:]=mx[:ix,:]*(np.matmul(mult,np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*(np.matmul(np.flipud(mult),np.ones((1,x.shape[1])))) # Pontos finais

    # if signal.ndim == 1:
    #     mx = mx[:, 0]
    return mx




def lofar(data, sr, 
          final_sr, 
          nfft=1024, 
          noverlap=0, 
          channel_axis=1,
          max_freq=None, 
          tonal_threshold=-4, # dB
          use_tpsw=True,
          return_onesided=True,
          **tpsw_args):

    if not isinstance(data, (np.ndarray, xr.DataArray)):
        raise ValueError(f"Data must be of type numpy.ndarray. {type(data)} was received")
    
    if data.ndim > 2:
        raise ValueError(f"Data must be a two dimensonal numpy array (frame and channel dimension). {data.ndim} dimensions found.")

    signal = data.copy()

    time_axis = 0 if channel_axis == 1 else 1
    signal = resample(signal, sr, final_sr, axis=0)
    sr = final_sr
    
    # def partial_rolling_window(data, window=nfft, overlap=noverlap):
    #     return rolling_window(data,  window=nfft, overlap=noverlap)
    # chunk_stack = np.apply_along_axis(partial_rolling_window, arr=data, axis=0)
    
    freq, time, sxx = spectrogram(signal,
                                  window=('hann'),
                                  nperseg=nfft,
                                  noverlap=noverlap,
                                  nfft=nfft,
                                  fs=sr,
                                  detrend='constant',
                                  return_onesided=return_onesided,
                                  axis=time_axis,
                                  scaling='spectrum',
                                  mode='magnitude')
    
    # win = np.hanning(nfft)[None, :, None]
    # sxx = np.fft.rfft(chunk_stack*win, axis=1)/nfft

    # sxx = np.absolute(sxx)
    # sxx = sxx / tpsw(sxx, **tpsw_args)
    # sxx = 20*np.log10(sxx)

    #freq = np.fft.fftfreq(nfft, d=1/sr)
    
    sxx = np.swapaxes(sxx, channel_axis, -1)
                                
    sxx = np.absolute(sxx)

    if use_tpsw:
        bg_sxx = np.empty_like(sxx)
        for channel in range(sxx.shape[-1]):
            bg_sxx[:, :, channel] = tpsw(sxx[:, :, channel], **tpsw_args)

        sxx = sxx / bg_sxx
    sxx = 20*np.log10(sxx)

    if tonal_threshold is not None:
        sxx[sxx < tonal_threshold] = 0

    sxx = np.swapaxes(sxx, 0, 1) # swap time and freq axis
    
    if not return_onesided:
        tmp = freq[:nfft//2].copy()
        freq[:nfft//2] = freq[nfft//2:]
        freq[nfft//2:] = tmp[::].copy()

        tmp = sxx[:, :nfft//2, :].copy()
        sxx[:, :nfft//2, :] = sxx[:, nfft//2:, :]
        sxx[:, nfft//2:, :] = tmp.copy()

    return sxx, freq, time

#
# 
# def _lofar_helper(data, sr, final_sr, nfft, noverlap, channel_axis, max_freq, tonal_threshold, **tpsw_args):
    


def resample(signal, fs, final_fs, window=('kaiser', 5.0), axis=0):
    resample_ratio = Fraction(final_fs, fs)

    upsampling_factor = resample_ratio.numerator
    downsampling_factor = resample_ratio.denominator

    resampled_signal = scipy.signal.resample_poly(
        signal, 
        upsampling_factor, 
        downsampling_factor,
        axis=axis, 
        window=window
    )

    return resampled_signal


def rolling_window(x: np.ndarray, window: int, overlap:int, padded: bool = True):
    hop = window - overlap

    idx = np.arange(0, len(x) - window + 1, hop).reshape((-1, 1)) + np.arange(window).reshape((1, -1))
    if padded:
        last_i = idx[-1,-1]
        pad_arr = np.repeat(0.0, window - (len(x) - last_i))
        
        last_window = np.concatenate((x[last_i:], pad_arr), axis=0)
        return np.concatenate((x[idx], last_window.reshape((1, -1))), axis = 0)
    return x[idx]

    
def demon(data, fs, n_fft=1024, max_freq=35, apply_bandpass=True, bandpass_specs=None, method='abs'):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be of type numpy.ndarray. %s was passed" % type(data))
    x = data.copy()

    first_pass_sr = 1250 # 31250/25

    q1 = round(fs/first_pass_sr) # 25 for 31250 sample rate ; decimatio ratio for 1st pass
    q2 = round((fs/q1)/(2*max_freq)) # decimatio ratio for 2nd pass

    fft_over = math.floor(n_fft-2*max_freq*0.5)

    if apply_bandpass:
        nyq = fs/2
        if bandpass_specs is None:
            wp = [1000/nyq, 2000/nyq]
            ws = [700/nyq, 2300/nyq]
            rp = 0.5
            As = 50
        elif isinstance(bandpass_specs, dict):
            try:
                fp = bandpass_specs["fp"]
                fs = bandpass_specs["fs"]

                wp = np.array(fp)/nyq
                ws = np.array(fs)/nyq
                
                rp = bandpass_specs["rs"]
                As = bandpass_specs["as"]
            except KeyError as e:
                raise KeyError("Missing %s specification for bandpass filter" % e)
        else:
            raise ValueError("bandpass_specs must be of type dict. %s was passed" % type(bandpass_specs))
        
        N, wc = cheb2ord(wp, ws, rp, As) 
        b, a = cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
        x = lfilter(b, a, x, axis=0)

    if method=='hilbert':
        x = hilbert(x)
    elif method=='abs':
        x = np.abs(x) # demodulation
    else:
        raise ValueError("Method not found")

    x = decimate(x, q1, ftype='fir', zero_phase=False)
    x = decimate(x, q2, ftype='fir', zero_phase=False)

    final_fs = (fs//q1)//q2

    x /= x.max()
    x -= np.mean(x)
    sxx = stft(x,
               window=('hann'),
               win_length=n_fft,
               hop_length=(n_fft - fft_over),
               n_fft=n_fft)
    freq = fft_frequencies(sr=final_fs, n_fft=n_fft)
    time = frames_to_time(np.arange(0, sxx.shape[1]), 
                   sr=final_fs, hop_length=(n_fft - fft_over))

    sxx = np.absolute(sxx)
    
    sxx = sxx / tpsw(sxx)

    sxx, freq = sxx[8:, :], freq[8:] # ??

    return np.transpose(sxx), freq, time
