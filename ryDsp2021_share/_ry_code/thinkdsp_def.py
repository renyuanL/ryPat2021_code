
import copy
import math

import numpy as np
import random
import scipy
import scipy.stats
import scipy.fftpack
import subprocess
import warnings

from wave import open as open_wave
from scipy.io import wavfile

import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz

try:
    from IPython.display import Audio
except:
    warnings.warn(
        "Can't import Audio from IPython.display; " "Wave.make_audio() will not work."
    )


PI2= math.pi * 2



def random_seed(x):
    """Initialize the random and np.random generators.

    x: int seed
    """
    random.seed(x)
    np.random.seed(x)




def play_wave(filename="sound.wav", player="aplay"):

    """Plays a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    cmd = "%s %s" % (player, filename)
    popen = subprocess.Popen(cmd, shell=True)
    popen.communicate()


def find_index(x, xs):
    """Find the index corresponding to a given value in an array."""
    n = len(xs)
    start = xs[0]
    end = xs[-1]
    i = round((n - 1) * (x - start) / (end - start))
    return int(i)


def unbias(ys):
    """Shifts a wave array so it has mean 0.

    ys: wave array

    returns: wave array
    """
    return ys - ys.mean()


def normalize(ys, amp=1.0):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.

    ys: wave array
    amp: max amplitude (pos or neg) in result

    returns: wave array
    """
    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)


def shift_right(ys, shift=1):
    """Shifts a wave array to the right and zero pads.

    ys: wave array
    shift: integer shift

    returns: wave array
    """
    res = np.zeros(len(ys) + shift)
    res[shift:] = ys
    return res


def shift_left(ys, shift=1):
    """Shifts a wave array to the left.

    ys: wave array
    shift: integer shift

    returns: wave array
    """
    return ys[shift:]


def truncate(ys, n):
    """Trims a wave array to the given length.

    ys: wave array
    n: integer length

    returns: wave array
    """
    return ys[:n]


def quantize(ys, bound, dtype):
    """Maps the waveform to quanta.

    ys: wave array
    bound: maximum amplitude
    dtype: numpy data type of the result

    returns: quantized signal
    """
    if max(ys) > 1 or min(ys) < -1:
        warnings.warn("Warning: normalizing before quantizing.")
        ys = normalize(ys)

    zs = (ys * bound).astype(dtype)
    return zs


def apodize(ys, framerate, denom=20, duration=0.1):
    """Tapers the amplitude at the beginning and end of the signal.

    Tapers either the given duration of time or the given
    fraction of the total duration, whichever is less.

    ys: wave array
    framerate: int frames per second
    denom: float fraction of the segment to taper
    duration: float duration of the taper in seconds

    returns: wave array
    """
    # a fixed fraction of the segment
    n = len(ys)
    k1 = n // denom

    # a fixed duration of time
    k2 = int(duration * framerate)

    k = min(k1, k2)

    w1 = np.linspace(0, 1, k)
    w2 = np.ones(n - 2 * k)
    w3 = np.linspace(1, 0, k)

    window = np.concatenate((w1, w2, w3))
    return ys * window


def infer_framerate(ts):
    """Given ts, find the framerate.

    Assumes that the ts are equally spaced.

    ts: sequence of times in seconds

    returns: frames per second
    """
    # TODO: confirm that this is never used and remove it
    dt = ts[1] - ts[0]
    framerate = 1.0 / dt
    return framerate



def mag(a):
    """Computes the magnitude of a numpy array.

    a: numpy array

    returns: float
    """
    return np.sqrt(np.dot(a, a))


def zero_pad(array, n):
    """Extends an array with zeros.

    array: numpy array
    n: length of result

    returns: new NumPy array
    """
    res = np.zeros(n)
    res[: len(array)] = array
    return res


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like

    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties

    https://matplotlib.org/api/axes_api.html

    In addition, you can use `legend=False` to suppress the legend.

    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.

    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.plt.legend.html

    """
    underride(options, loc="best", frameon=False)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


def remove_from_legend(bad_labels):
    """Removes some labels from the legend.

    bad_labels: sequence of strings
    """
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in bad_labels:
            handle_list.append(handle)
            label_list.append(label)
    ax.legend(handle_list, label_list)


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


