import vtk
import warnings
import pyshtools
import numpy as np
from vtk.util import numpy_support
from skimage import transform as sktrans
from scipy import interpolate as spinterp

from . import shtools


def get_shcoeffs(
    image: np.array,
    lmax: int,
    sigma: float = 0,
    compute_lcc: bool = True,
    alignment_2d: bool = True,
    make_unique: bool = False,
):
    """
    Compute spherical harmonics coefficients that describe an object stored as
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
    image\\_ : ndarray
        Input image after pre-processing (lcc calculation, smooth and binarization).
    mesh : vtkPolyData
        Polydata representation of image\\_.
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
        Whether to compute the largest connected component before applying the
        spherical harmonic parametrization, default is True. Set compute_lcc to
        False in case you are sure the input image contains a single connected
        component. It is crucial that parametrization is calculated on a single
        connected component object.
    alignment_2d : bool
        Whether the image should be aligned in 2d. Default is True.
    make_unique : bool
        Set true to make sure the alignment rotation is unique.

    Notes
    -----
    Alignment mode '2d' allows for keeping the z axis unchanged which might be
    important for some applications.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from aicsshparam import shparam, shtools

        img = np.ones((32,32,32), dtype=np.uint8)

        (coeffs, grid_rec), (image_, mesh, grid, transform) =
            shparam.get_shcoeffs(image=img, lmax=2)
        mse = shtools.get_reconstruction_error(grid, grid_rec)

        print('Coefficients:', coeffs)

        >>> Coefficients: {'shcoeffs_L0M0C': 18.31594310878251, 'shcoeffs_L0M1C': 0.0,
        'shcoeffs_L0M2C': 0.0, 'shcoeffs_L1M0C': 0.020438775421611564, 'shcoeffs_L1M1C':
        -0.0030960466571801513, 'shcoeffs_L1M2C': 0.0, 'shcoeffs_L2M0C':
        -0.0185688727281408, 'shcoeffs_L2M1C': -2.9925077712704384e-05,
        'shcoeffs_L2M2C': -0.009087503958673892, 'shcoeffs_L0M0S': 0.0,
        'shcoeffs_L0M1S': 0.0, 'shcoeffs_L0M2S': 0.0, 'shcoeffs_L1M0S': 0.0,
        'shcoeffs_L1M1S': 3.799611612562637e-05, 'shcoeffs_L1M2S': 0.0,
        'shcoeffs_L2M0S': 0.0, 'shcoeffs_L2M1S': 3.672543904347801e-07,
        'shcoeffs_L2M2S': 0.0002230857005948496}

        print('Error:', mse)

        >>> Error: 2.3738182456948795
    """

    if len(image.shape) != 3:
        raise ValueError(
            "Incorrect dimensions: {}. Expected 3 dimensions.".format(image.shape)
        )

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
        image_, angle = shtools.align_image_2d(image=image_, make_unique=make_unique)
        image_ = image_.squeeze()

    # Converting the input image into a mesh using regular marching cubes
    mesh, image_, centroid = shtools.get_mesh_from_image(image=image_, sigma=sigma)

    if not image_[tuple([int(u) for u in centroid[::-1]])]:
        warnings.warn("Mesh centroid seems to fall outside the object. This indicates\
        the mesh may not be a manifold suitable for spherical harmonics\
        parameterization.")

    transform = centroid + ((angle,) if alignment_2d else ())

    # Fit the SH expansion on the mesh point coordinates. The mesh is
    # assumed to be centered at the origin at this point.
    coeffs_dict, grid_rec, grid_down, mesh = _get_shcoeffs_from_mesh_coords(mesh, lmax)

    return (coeffs_dict, grid_rec), (image_, mesh, grid_down, transform)


def _get_shcoeffs_from_mesh_coords(mesh, lmax: int):
    """Fit the spherical harmonics expansion on the point coordinates of a
    mesh that is already centered at the origin.

    This is the shared fitting core used by both the image-based
    (``get_shcoeffs``) and mesh-based (``get_shcoeffs_from_mesh``) entry
    points. It extracts the mesh point coordinates, converts them to
    spherical coordinates, interpolates the radius onto a regular
    (lon, lat) grid and expands it with pyshtools.

    Parameters
    ----------
    mesh : vtkPolyData
        Mesh centered at the origin.
    lmax : int
        Order of the spherical harmonics parametrization.

    Returns
    -------
    coeffs_dict : dict
        Dictionary with the spherical harmonics coefficients.
    grid_rec : ndarray
        Parametric grid representing the SH parametrization.
    grid_down : ndarray
        Parametric grid representing the input object.
    mesh : vtkPolyData
        Mesh with updated point normals.
    """

    # Get coordinates of mesh points
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Update mesh normals
    mesh = shtools.update_mesh_points(mesh, x, y, z)

    # Cartesian to spherical coordinates convertion
    rad = np.sqrt(x**2 + y**2 + z**2)
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
        for L, m in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{L}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))

    return coeffs_dict, grid_rec, grid_down, mesh


def get_shcoeffs_from_mesh(
    mesh: vtk.vtkPolyData,
    lmax: int,
    alignment_2d: bool = True,
    make_unique: bool = False,
):
    """Compute spherical harmonics coefficients that describe an object
    stored as a surface mesh.

    This is the high-level entry point for users who already have a mesh
    (e.g. from a meshing pipeline, CAD or simulation) and want to avoid
    voxelizing it back into an image. It reuses the same fitting core as
    the image-based ``get_shcoeffs``.

    The surface is represented as a single radius per direction
    r(lon, lat), so it is expected to be closed and star-shaped about its
    centroid. ``check_mesh_for_parametrization`` is called to warn (not
    raise) when these assumptions appear to be violated.

    Notes
    -----
    The ``alignment_2d`` and ``make_unique`` options behave the same way as
    in ``get_shcoeffs``, but the alignment is computed differently and the
    two paths are not guaranteed to produce bit-identical orientations for
    the same object:

    - The image path runs PCA on all foreground *voxel* (x, y) coordinates
      (a volume point set), rotates the *image* (with interpolation), then
      re-meshes.
    - This mesh path runs PCA on the *surface vertex* (x, y) coordinates
      and rotates the vertices directly (no interpolation).

    The principal-axis angle formula is shared, but because the two point
    sets differ the resulting angle can differ slightly. For a clean,
    well-aligned object the two paths agree closely. Reproducing the image
    path exactly would require voxelizing and aligning as an image, which
    defeats the purpose of this entry point.

    Parameters
    ----------
    mesh : vtkPolyData
        Input surface mesh.
    lmax : int
        Order of the spherical harmonics parametrization. The higher the
        order the more shape details are represented.
    alignment_2d : bool
        Whether the mesh should be aligned in 2d before parametrization.
        The alignment is computed on the mesh vertex point cloud (PCA of
        the xy coordinates) and applied as a rotation in the xy plane,
        leaving z unchanged. Default is True.
    make_unique : bool
        Set true to make sure the alignment rotation is unique.

    Returns
    -------
    coeffs_dict : dict
        Dictionary with the spherical harmonics coefficients.
    grid_rec : ndarray
        Parametric grid representing the SH parametrization.
    image\\_ : None
        Always None for mesh input; the slot is kept so the return shape
        matches ``get_shcoeffs`` and callers can unpack both identically.
    mesh : vtkPolyData
        The centered (and optionally aligned) mesh with updated normals.
    grid_down : ndarray
        Parametric grid representing the input object.
    transform : tuple of floats
        (xc, yc, zc, angle) if alignment_2d is True or (xc, yc, zc)
        otherwise. (xc, yc, zc) is the centroid of the input mesh; angle
        is the rotation used to align it.

    Examples
    --------

    .. code-block:: python

        from aicsshparam import shparam, shtools

        mesh = shtools.get_mesh_from_vertices_faces(vertices, faces)

        (coeffs, grid_rec), (_, mesh, grid, transform) =
            shparam.get_shcoeffs_from_mesh(mesh=mesh, lmax=2)
        mse = shtools.get_reconstruction_error(grid, grid_rec)
    """

    if mesh.GetPoints() is None or mesh.GetNumberOfPoints() == 0:
        raise ValueError("No mesh points found. Is the input mesh empty?")

    # Warn (without raising) on violations of the geometric assumptions.
    shtools.check_mesh_for_parametrization(mesh)

    # Work on a copy so the caller's mesh is not mutated.
    mesh_aligned = vtk.vtkPolyData()
    mesh_aligned.DeepCopy(mesh)
    mesh = mesh_aligned

    # Center the mesh on its centroid (origin), then align on the point
    # cloud so the parametrization is computed in a canonical frame.
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    centroid = tuple(coords.mean(axis=0))
    x = coords[:, 0] - centroid[0]
    y = coords[:, 1] - centroid[1]
    z = coords[:, 2] - centroid[2]

    if alignment_2d:
        x, y, angle = shtools.align_points_2d(x=x, y=y, make_unique=make_unique)

    mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(np.c_[x, y, z], deep=True))
    mesh.Modified()

    transform = centroid + ((angle,) if alignment_2d else ())

    coeffs_dict, grid_rec, grid_down, mesh = _get_shcoeffs_from_mesh_coords(mesh, lmax)

    return (coeffs_dict, grid_rec), (None, mesh, grid_down, transform)


def get_shcoeffs_from_vertices_faces(
    vertices: np.array,
    faces: np.array,
    lmax: int,
    alignment_2d: bool = True,
    make_unique: bool = False,
):
    """Compute spherical harmonics coefficients from raw mesh arrays.

    Convenience wrapper around ``get_shcoeffs_from_mesh`` for users who
    have a mesh as numpy arrays of vertices and faces rather than as a
    vtkPolyData object.

    Parameters
    ----------
    vertices : np.array
        Array of vertex coordinates with shape (N, 3).
    faces : np.array
        Array of triangular faces with shape (M, 3) of vertex indices.
    lmax : int
        Order of the spherical harmonics parametrization.
    alignment_2d : bool
        Whether the mesh should be aligned in 2d. Default is True.
    make_unique : bool
        Set true to make sure the alignment rotation is unique.

    Returns
    -------
    Same as ``get_shcoeffs_from_mesh``.
    """

    mesh = shtools.get_mesh_from_vertices_faces(
        vertices=vertices, faces=faces, translate_to_origin=False
    )

    return get_shcoeffs_from_mesh(
        mesh=mesh,
        lmax=lmax,
        alignment_2d=alignment_2d,
        make_unique=make_unique,
    )
