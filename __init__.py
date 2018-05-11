#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
fibflatten

This module contains utilities to flatten FIB-SEM data.

frequently we have a priori knowledge that an edge should be flat
but do to artifacts with the SIFT alignment algorithm or even physical
effects during freeze substitution or resin embedding the bottom surface
becomes warped

Copyright (c) 2018, David Hoffman
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.external.tifffile as tif
import scipy.signal
import scipy.ndimage as ndi
from .rolling_ball import rolling_ball_filter


def find_edge_line(d, max_percentage=0.25, win=11, order=3, poly=3, invert=False, diagnostics=False):
    """Find the "edge" along a given line"""
    if max_percentage < 0.0:
        return np.nan
    deriv = scipy.signal.savgol_filter(d, win, poly, 1)
    deriv[abs(deriv) < np.nanmax(abs(deriv)) * max_percentage] = 0
    if invert:
        func = scipy.signal.argrelmin
    else:
        func = scipy.signal.argrelmax
    extrema = func(deriv, order=order)[0]
    # make some plots to evaluate how well it works
    if diagnostics:
        fig, (ax0, ax1) = plt.subplots(2)
        ax0.plot(d)
        ax1.plot(deriv)
        for m in extrema:
            for ax in (ax0, ax1):
                ax.axvline(m, c="r")
    if len(extrema):
        # return the last extrema, this will need to be made more flexible
        return extrema[-1]
    else:
        # if no extrema are found try again with a lower threshold
        return find_edge_line(d, max_percentage=max_percentage - 0.05, win=win, order=order,
                              poly=poly, invert=invert, diagnostics=diagnostics)


def find_edge_plane(d, **kwargs):
    diagnostics = kwargs.pop("diagnostics", False)
    edge = np.array([find_edge_line(dd, **kwargs) for dd in d.T])
    if diagnostics:
        plt.matshow(d, cmap="Greys")
        plt.plot(edge, "r")
    return edge
