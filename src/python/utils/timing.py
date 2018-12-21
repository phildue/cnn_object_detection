import timeit

import numpy as np


def tic():
    Ticker.start_times.append(timeit.default_timer())


def toc(message: str = "Time passed: ", verbose=True):
    stop = None
    if not Ticker.start_times:
        print("Toc::No timers set")
    else:
        start = Ticker.start_times.pop()
        stop = timeit.default_timer() - start
        if verbose:
            print(message + str(int(np.round(stop, 2))) + " seconds")
    return stop


def tuc(message: str = "Time passed: ", verbose=True):
    stop = None
    if not Ticker.start_times:
        print("Toc::No timers set")
    else:
        start = Ticker.start_times[-1]
        stop = timeit.default_timer() - start
        if verbose:
            print(message + str(int(np.round(stop, 2))) + " seconds")
    return stop


class Ticker:
    start_times = []
