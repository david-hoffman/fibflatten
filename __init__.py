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

import dask
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
    nans = np.isfinite(d)
    try:
        deriv_temp = scipy.signal.savgol_filter(d[nans], win, poly, 1)
    except ValueError:
        return np.nan
    deriv = np.zeros(len(d))
    deriv[nans] = deriv_temp
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


def sift_baseline(edges, median_kernel=15, ballsize=15, diagnostics=False, **kwargs):
    """Estimate the SIFT baseline error from the raw edges"""
    # remove noise with broad median filter
    med_edges = ndi.median_filter(edges, median_kernel)
    # the sift baseline should only be in one direction
    baseline = np.nanmean(med_edges, 1)
    # clean up some more with rolling ball filter
    baseline_smth = rolling_ball_filter(baseline, ballsize, **kwargs)[1]
    if diagnostics:
        vdict = vminmax(edges)
        plt.matshow(edges, **vdict)
        plt.matshow(med_edges, **vdict)
        plt.figure()
        plt.plot(baseline)
        plt.plot(baseline_smth)
        plt.gca().set_ylim([vdict["vmin"], vdict["vmax"]])
        
    return baseline_smth - baseline_smth.mean()


def baseline_angle(baseline, start=None, stop=None, diagnostics=False):
    """Find the mean angle of the baseline"""
    # calculate x
    x = np.arange(len(baseline))
    # slice if requested (remove edge effects)
    s = slice(start, stop)
    # fit a line to the data
    m, b = np.polyfit(x[s], baseline[s], 1)
    # extract theta
    theta = np.arctan(m)
    # generate baseline without rotation (actually it's just subtraction, which is why later in
    # generate_new_coords we subtract it after removing the rotation)
    new_baseline = baseline - m*x + b
    new_baseline -= new_baseline.mean()
    if diagnostics:
        fig, (ax0, ax1) = plt.subplots(2)
        ax0.plot(x, baseline)
        ax0.plot(x, m*x + b)
        ax0.axvline(x[s][0], color="r")
        ax0.axvline(x[s][-1], color="r")
        ax1.plot(x, new_baseline)
        
    return new_baseline, theta


def generate_new_coords(data, baseline, theta):
    """Generates new coordinates from a given baseline to be used in warping later"""
    # we assume that all SIFT distortion is along the z axis so we need to warp the ZY coordinates
    coords = np.indices(data.shape[:2])
    # find the center (rotations need to be performed about the center)
    center = np.array(data.shape[:2])[:, None, None] // 2
    # shift, rotate, shift back
    rot_coords = apply(rotmatrix(-theta), coords - center) + center
    # remove baseline
    # the order may look weird here, shouldn't we remove the base line first? But remember rot_coords[1]
    # are still all Y coordinates (their positions in the matrix have been rotated)
    rot_coords_nb = np.array((rot_coords[0], rot_coords[1] + baseline[:, None]))
    
    return rot_coords, rot_coords_nb


def remove_sift_baseline(data, new_coords):
    # in this case we pre-rotate everything before applying it on a plane by plane basis
    # and we preallocate the result volume which, for large data, results in much faster
    # processing.
    data_new = np.empty_like(data)
    data_new_rolled = np.rollaxis(data_new, -1) 
    data_rolled = np.rollaxis(data, -1)
    # apply transformation and put in pre-allocated volume
    dask.delayed([dask.delayed(ndi.map_coordinates)(d, new_coords, dn) for d, dn in zip(data_rolled, data_new_rolled)]).compute()
    
    return data_new


def rotmatrix(angle):
    """Generate a rotation matrix"""
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def apply(transform, coords):
    """Apply a matrix transformation to the coordinates"""
    return np.tensordot(transform, coords, axes=1)


def vminmax(data, p=1):
    return dict(vmin=np.nanpercentile(data, p), vmax=np.nanpercentile(data, 100 - p))


def plot_coords(img, coords):
    """show the coordinates and the warped image"""
    
    limit_kwds = dict(vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    
    fig, ((g00, g01), (g10, g11)) = fig, grid = plt.subplots(2, 2, figsize=(20,4))
    
    # show coordinates
    g00.matshow(coords[0].T)
    g00.contour(coords[0].T, colors="k")
    g00.set_title("Y coordinate")
    g01.matshow(coords[1].T)
    g01.contour(coords[1].T, colors="k")
    g01.set_title("X coordinate")
    
    # make new image
    img2 = ndi.map_coordinates(img, coords)
    # show images
    g10.matshow(img.T, **limit_kwds)
    g10.set_title("Original Image")
    g11.matshow(img2.T, **limit_kwds)
    g11.set_title("Warped Image")
    
    for g in grid.ravel():
        g.axis("off")