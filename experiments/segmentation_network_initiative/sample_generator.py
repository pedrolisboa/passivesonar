import numpy as np
import sys
from modified_lofar import lofar


def gen_sim_lofargram(seconds, sample_rate, return_axis=False):
    n_freqs = np.random.randint(1,11)

    bands = frequency_gen(n_freqs)
    x = np.zeros(seconds*sample_rate)
    for f, p in bands:
        x += stochastic_narrowband_tone(seconds, sample_rate, f, p)

    noise = background_noise(seconds, sample_rate)

    y = x  + noise
    y = y[:, np.newaxis] 

    sxx, freq, time = lofar(y, sr=sample_rate, final_sr=7350, use_tpsw=True, tonal_threshold=-4)
    
    sxx = sxx[:, (512-21):, :] # padding for unet
    freq = freq[(512-21):]
    
    mask = sxx.copy().squeeze()
    mask[:, :] = 0
    for f, p in bands:
        mask[:, (freq >= (f - p)) & (freq <= (f + p))] = 1
    
    full_mask = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=bool)
    full_mask[:, :, 0] =  mask.astype(bool)
    full_mask[:, :, 1] = ~mask.astype(bool)
    
    if return_axis:
        return sxx, full_mask, freq, time
    return sxx, full_mask

def frequency_band_gen(freq_array=None):
    #base_width = np.random.uniform(0.005, 0.010)
    base_width = np.random.uniform(7, 15)
    base_width = np.random.normal(10, 5)
    if freq_array is None:
        center_freq = np.random.uniform(20, 6600//2)
        
        return center_freq, base_width
    else:
        idx = np.random.randint(0, len(freq_array)-1)
        center_freq = freq_array[idx]
        interval_mask = (freq_array > center_freq - base_width) & (freq_array < center_freq +base_width)
        
        freq_interval = np.argwhere(freq_array[interval_mask])
        
        freq_array = np.delete(freq_array, [freq_interval])
        
        return center_freq, base_width, freq_array

def frequency_gen(n=1):
    freq_array = np.arange(100, 6600//2 - 30, 1)
    bands = list()
    for i_freq in range(n):
        center, base, freq_array = frequency_band_gen(freq_array)
        bands.append((center, base))
    return bands

def narrowband_tone(seconds, sample_rate, f, width, signal_amplitude):
    t = np.linspace(0, seconds, seconds*sample_rate)
    
    x = band_limited_noise(f-  width , f + width, samples=seconds*sample_rate, samplerate=sample_rate)
    x = scl(x)
    
    ampl = max(1500/f, f/1500)**0.5
    x *= signal_amplitude*ampl
    return x

def stochastic_narrowband_tone(seconds, sample_rate, f, width):
    signal_amplitude = np.random.lognormal(mean=0, sigma=1.2, size=seconds*sample_rate)
    width_array = width*(1+np.random.normal(0, 1.8, size=seconds*sample_rate))
    x = narrowband_tone(seconds, sample_rate, f, width_array, signal_amplitude)
    return x

def background_noise(seconds, sample_rate):
    noise_base_value = np.random.uniform(5, 20)
    noise_fluct_limit = 20 - noise_base_value
    
    noise_fluct_scale = np.random.uniform(1, noise_fluct_limit)
    noise_amplitude = noise_base_value + np.random.exponential(scale=noise_fluct_scale, size=seconds*sample_rate)
    
    noise_att = np.random.uniform(1.5, 2.5)
    
    noise = spectrum_noise(lambda x:pink_spectrum(x, att=np.log10(noise_att)*10), seconds*sample_rate, sample_rate)
    noise = scl(noise)
    noise *= noise_amplitude
    
    return noise
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def scl(x):
    return (x - x.mean())/x.std()

def spectrum_noise(spectrum_func, samples=1024, rate=44100):
    """ 
    make noise with a certain spectral density
    """
    freqs = np.fft.rfftfreq(samples, 1.0/rate)            # real-fft frequencies (not the negative ones)
    spectrum = np.zeros_like(freqs, dtype='complex')      # make complex numbers for spectrum
    spectrum[1:] = spectrum_func(freqs[1:])               # get spectrum amplitude for all frequencies except f=0
    phases = np.random.uniform(0, 2*np.pi, len(freqs)-1)  # random phases for all frequencies except f=0
    spectrum[1:] *= np.exp(1j*phases)                     # apply random phases
    noise = np.fft.irfft(spectrum)                        # return the reverse fourier transform
    noise = np.pad(noise, (0, samples - len(noise)), 'constant') # add zero for odd number of input samples
 
    return noise
 
def pink_spectrum(f, f_min=0, f_max=np.inf, att=np.log10(2.0)*10):
    """
    Define a pink (1/f) spectrum
        f     = array of frequencies
        f_min = minimum frequency for band pass
        f_max = maximum frequency for band pass
        att   = attenuation per factor two in frequency in decibel.
                Default is such that a factor two in frequency increase gives a factor two in power attenuation.
    """
    # numbers in the equation below explained:
    #  0.5: take the square root of the power spectrum so that we get an amplitude (field) spectrum 
    # 10.0: convert attenuation from decibel to bel
    #  2.0: frequency factor for which the attenuation is given (octave)
    s = f**-( 0.5 * (att/10.0) / np.log10(2.0) )  # apply attenuation
    s[np.logical_or(f < f_min, f > f_max)] = 0    # apply band pass
    return s