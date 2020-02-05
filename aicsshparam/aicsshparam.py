import vtk
import pyshtools
import numpy as np
import pandas as pd
from skimage import transform as sktrans
from scipy import interpolate as spinterp

from . import aicsshtools

"""
    Computes the spherical harmonic parametrization of a starlike object
    in the 3D input image "seg".

    :return:
        * dictionary with the coefficients plus the chi square error.
        * (
            img - Input image after pre-processing
            mesh - Polydata representation of the input image
            cm - Centroid of input object
            grid_down - Parametric grid representing input object
            grid_rec - Parametric grid representing sh parametrization
            coeffs - sh coefficies in matrix form
            chi2 - goodness of fit
        )
"""

def get_shcoeffs(seg, params):

    # Parameters

    min_size = 128 # pixels (fixed)

    align = 'align2d' if "align" not in params else params["align"]

    sigma = None if "sigma" not in params else params["sigma"]

    lmax = 8 if "lmax" not in params else params["lmax"]

    # Check input size

    features = {}

    if len(seg.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(seg.shape))

    # Features names

    l_labels = np.repeat(np.arange(lmax+1).reshape(-1,1),lmax+1,axis=1)
    m_labels = np.repeat(np.arange(lmax+1).reshape(-1,1),lmax+1,axis=1).T
    l_labels = l_labels.reshape(-1)
    m_labels = m_labels.reshape(-1) 

    lm_labels  = ['shcoeffs_L{:d}M{:d}C'.format(l,m) for (l,m) in zip(l_labels,m_labels)]
    lm_labels += ['shcoeffs_L{:d}M{:d}S'.format(l,m) for (l,m) in zip(l_labels,m_labels)]

    lm_labels += ['shcoeffs_energy{:d}'.format(l) for l in range(1,lmax+1)]

    ft_labels = lm_labels + ['shcoeffs_chi2']

    # Size threshold

    seg[seg>0] = 1  # Make input is binary

    if seg.sum() < min_size:
        for f in ft_labels:
            features[f] = np.nan
        return features, None

    mesh, (img,cm) = aicsshtools.get_polydata_from_numpy(volume=seg, lcc=False, sigma=sigma, center=True, size_threshold=min_size)

    if mesh is None:
        return features, None

    x, y, z = [], [], []
    for i in range(mesh.GetNumberOfPoints()):
        r = mesh.GetPoints().GetPoint(i)
        x.append(r[0])
        y.append(r[1])
        z.append(r[2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if align == 'align2d':
        x, y, z = aicsshtools.align_points(x,y,z)
    elif align == 'align3d':
        x, y, z = aicsshtools.align_points(x, y, z, from_axis=0, to_axis=0, align3d=True)
        x, y, z = aicsshtools.align_points(x, y, z, from_axis=1, to_axis=1, align3d=True)

    # Translate and align mesh points

    mesh = aicsshtools.update_mesh_points(mesh,x,y,z)

    # SH calculation

    r = np.sqrt(x**2+y**2+z**2)
    lat = np.arccos(np.divide(z, r, out=np.zeros_like(r), where=r!=0))
    lon = np.pi + np.arctan2(y,x)

    imethod = 'nearest'
    points = np.concatenate([np.array(lon).reshape(-1,1),np.array(lat).reshape(-1,1)], axis=1)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2*np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=  np.pi, num=128, endpoint=True)
    )
    grid = spinterp.griddata(points, r, (grid_lon, grid_lat), method=imethod)

    # Fit grid data with SH

    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    if coeffs[0,0,0] < 1e-5: # if first coeff too small return nan
        for f in ft_labels:
            features[f] = np.nan
        return features, None

    # Reconstruct grid

    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Compute rec error

    grid_down = sktrans.resize(grid,output_shape=grid_rec.shape, preserve_range=True)

    chi2 = np.power(grid_down-grid_rec, 2).mean()

    energy = np.square(np.abs(coeffs)/coeffs[0,0,0])
    energy = energy.sum(axis=0).sum(axis=1)

    features = np.concatenate((np.concatenate((coeffs[0].ravel(),coeffs[1].ravel())),energy[1:],[chi2]))

    # return results

    features = pd.DataFrame(features.reshape(1,-1))

    features.columns = ft_labels
    
    return features.to_dict("records")[0], (img, mesh, cm, grid_down, grid_rec, coeffs, chi2)
