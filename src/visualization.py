import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
import seaborn as sns
import numpy as np

def waterfall_spectrogram(ax, freq, time, sxx, title='', cmap='jet', cbar_unit='dB', cbar_label='Magnitude', show_rpm=False):
    if show_rpm:
        freq *= 60
        xlabel = "Rotation (rpm)"
        # tick_multiplier = 60
    else:
        xlabel = "Frequency (Hz)"
        # tick_multiplier = 1

    
    #changing for imshow to handle negative frequencies, as pcolormesh does not handle well non monotonic coords
    #x0, y0 = freq[0], time[0]
    #x1, y1 = freq[-1], time[-1]
    #num_x, num_y = len(freq), len(time)
    #dx = 1. * (x1 - x0) / (num_x-1)
    #dy = 1. * (y1 - y0) / (num_y-1)

    #cmap = ax.imshow(sxx, extent=[x0 - dx/2, x1 + dx/2, y0 - dy/2, y1 + dy/2], origin='upper', aspect='auto', interpolation='nearest')
    cmap = ax.pcolormesh(freq, time, sxx, cmap=cmap, vmin=sxx.min(), shading='auto',vmax=sxx.max());

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")

    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.xaxis.set_minor_locator(MaxNLocator(5))
    
    ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_minor_locator(MaxNLocator(2))

    if cmap is not None:
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.ax.set_ylabel(f"{cbar_label} {cbar_unit}", rotation=270)
    
        return ax, cbar
    return ax