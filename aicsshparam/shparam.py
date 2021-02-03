import pyshtools
import numpy as np
from vtk.util import numpy_support
from skimage import transform as sktrans
from scipy import interpolate as spinterp

from . import shtools


def get_shcoeffs(
    image, lmax, sigma=0, compute_lcc=True, alignment_2d=True, make_unique=False
):

    """Compute spherical harmonics coefficients that describe an object stored as
    an image.

        Calculates the spherical harmonics coefficients that parametrize the shape
        formed by the foreground set of voxels in the input image. The input image
        does not need to be binary and all foreground voxels (background=0) are used
        in the computation. Foreground voxels must form a single connected component.
        If you are sure that this is the case for the input image, you can set
        compute_lcc to False to speed up the calculation. In addition, the shape is
        expected to be centered in the input image.

        Parameters
        ----------
        image : ndarray
            Input image. Expected to have shape ZYX.
        lmax : int
            Order of the spherical harmonics parametrization. The higher the order
            the more shape details are represented.
        Returns
        -------
        coeffs_dict : dict
            Dictionary with the spherical harmonics coefficients and the mean square
            error between input and its parametrization
        grid_rec : ndarray
            Parametric grid representing sh parametrization
        image_ : ndarray
            Input image after pre-processing (lcc calculation, smooth and binarization).
        mesh : vtkPolyData
            Polydata representation of image_.
        grid_down : ndarray
            Parametric grid representing input object.
        transform : tuple of floats
            (xc, yc, zc, angle) if alignment_2d is True or
            (xc, yc, zc) if alignment_2d is False. (xc, yc, zc) are the coordinates
            of the shape centroid after alignment; angle is the angle used to align
            the image

        Other parameters
        ----------------
        sigma : float, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        compute_lcc : bool, optional
            Whether to compute the largest connected component before appliying the
            spherical harmonic parametrization, default is True. Set compute_lcc to
            False in case you are sure the input image contains a single connected
            component. It is crucial that parametrization is calculated on a single
            connected component object.
        alignment_2d : bool
            Wheather the image should be aligned in 2d. Default is True.
        make_unique : bool
            Set true to make sure the alignment rotation is unique.
            
        Notes
        -----
        Alignment mode '2d' allows for keeping the z axis unchanged which might be
        important for some applications.

        Examples
        --------
        >>> import numpy as np
        >>> from aicsshparam import shparam, shtools
        >>>
        >>> img = np.ones((32,32,32), dtype=np.uint8)
        >>>
        >>> (coeffs, grid_rec), (image_, mesh, grid, transform) = shparam.get_shcoeffs(image=img, lmax=2)
        >>> mse = shtools.get_reconstruction_error(grid,grid_rec)
        >>>
        >>> print('Coefficients:', coeffs)
        >>> print('Error:', mse)
        Coefficients: {'shcoeffs_L0M0C': 18.31594310878251, 'shcoeffs_L0M1C': 0.0, 'shcoeffs_L0M2C':
        0.0, 'shcoeffs_L1M0C': 0.020438775421611564, 'shcoeffs_L1M1C': -0.0030960466571801513,
        'shcoeffs_L1M2C': 0.0, 'shcoeffs_L2M0C': -0.0185688727281408, 'shcoeffs_L2M1C':
        -2.9925077712704384e-05, 'shcoeffs_L2M2C': -0.009087503958673892, 'shcoeffs_L0M0S': 0.0,
        'shcoeffs_L0M1S': 0.0, 'shcoeffs_L0M2S': 0.0, 'shcoeffs_L1M0S': 0.0, 'shcoeffs_L1M1S':
        3.799611612562637e-05, 'shcoeffs_L1M2S': 0.0, 'shcoeffs_L2M0S': 0.0, 'shcoeffs_L2M1S':
        3.672543904347801e-07, 'shcoeffs_L2M2S': 0.0002230857005948496}
        Error: 2.3738182456948795
    """

    if len(image.shape) != 3:
        raise ValueError("Incorrect dimensions: {}".format(image.shape))

    if image.sum() == 0:
        raise ValueError("No foreground voxels found. Is the input image empty?")

    # Binarize the input. We assume that everything that is not background will
    # be use for parametrization
    image_ = image.copy()
    image_[image_ > 0] = 1

    # Alignment
    if alignment_2d:
        # Align the points such that the longest axis of the 2d
        # xy max projected shape will be horizontal (along x)
        image_, angle = shtools.align_image_2d(
            image=image_, make_unique=make_unique
        )
        image_ = image_.squeeze()

    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, centroid = shtools.get_mesh_from_image(image=image_, sigma=sigma)

    if not image_[tuple([int(u) for u in centroid[::-1]])]:
        print('Warning: Centroid seems to fall off the object...')
        
    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    transform = centroid + ((angle,) if alignment_2d else ())

    # Translate and update mesh normals
    mesh = shtools.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate(
        [np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1
    )

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # Reconstruct grid. Look at pyshtools for detail.
    grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Resize the input grid to match the size of the reconstruction
    grid_down = sktrans.resize(grid, output_shape=grid_rec.shape, preserve_range=True)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)

    keys = []
    for suffix in ["C", "S"]:
        for (l, m) in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{l}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return (coeffs_dict, grid_rec), (image_, mesh, grid_down, transform)
