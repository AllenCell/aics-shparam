import vtk
import pyshtools
import numpy as np
from vtk.util import numpy_support
from skimage import filters as skfilters
from skimage import morphology as skmorpho
from sklearn import decomposition as skdecomp

def rotation_matrix(axis, theta):

    """
        Creates a rotation matrix in 3D. This will rotate an objact about the axis
        `axis` by an angle `angle`.

        Parameters
        ----------
        axis : tuple of floats
            x y and z coordinates of the versor that defines the axis of rations.
        angle : float
            Angle of rotation in radians.
        Returns
        -------
        mx_rot : ndarray
            Matrix of rotation

        Other parameters
        ----------------
        sigma : int, optional
            The degree of smooth to be applied to the input image, default is 0 (no
            smooth)
        Notes
        -----
        This is just an implementation of Rodrigues' rotation formula:
            http://mathworld.wolfram.com/RodriguesRotationFormula.html

    """

    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    mx_rot = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return mx_rot

def get_polydata_from_numpy(array, sigma=0, lcc=True, translate_to_origin=True):

    """
        Converts a numpy array into a vtkImageData and then into a 3d mesh
        using vtkContourFilter. The input is assumed to be binary and the
        isosurface value is set to 0.5.

        Optionally the input can be pre-processed by i) extracting the largest
        connected component and ii) applying a gaussian smooth to it. In case
        smooth is used, the image is binarize using thershold 1/e.

        A size threshold is applying to garantee that enough points will be
        used to compute the SH parametrization.

        Also, points as the edge of the image are set to zero (background)
        to make sure the mesh forms a manifold.

        Parameters
        ----------
        array : ndarray
            Input array where the polydata will be computed on
        Returns
        -------
        polydata : vtkPolyData
            3d mesh in VTK format
        img_output : ndarray
            Input image after pre-processing
        centroid : tuple of floats
            x, y, z coordinates of the mesh centroid

        Other parameters
        ----------------
        lcc : bool, optional
            Wheather or not to comput the polydata only on the largest
            connected component found in the input connected component,
            default is True.
        sigma : float, optional
            The degree of smooth to be applied to the input image, default
            is 0 (no smooth).
        translate_to_origin : bool, optional
            Wheather or not translate the mesh to the origin (0,0,0),
            default is True.

    """

    img = array.copy()

    # VTK requires YXZ
    img = np.swapaxes(img, 0, 2)

    # Extracting the largest connected component
    if lcc:

        img = skmorpho.label(
            img.astype(np.uint8)
        )

        counts = np.bincount(img.flatten())

        lcc = 1 + np.argmax(counts[1:])

        img[img != lcc] = 0
        img[img == lcc] = 1

    # Smooth binarize the input image and binarize
    if sigma:

        img = skfilters.gaussian(
            img.astype(np.float32),
            sigma = (sigma, sigma, sigma)
        )

        img[img < 1.0/np.exp(1.0)] = 0
        img[img > 0] = 1

        if img.sum() == 0:
            raise ValueError("No foreground voxels found after pre-processing. Try using sigma=0.")

    # Set image border to 0 so that the mesh forms a manifold
    img[[0,-1], :, :] = 0
    img[:, [0,-1], :] = 0
    img[:, :, [0,-1]] = 0
    img = img.astype(np.float32)

    if img.sum() == 0:
        raise ValueError("No foreground voxels found after pre-processing. Is the object of interest centered?")

    # Create vtkImageData
    imgdata = vtk.vtkImageData()
    imgdata.SetDimensions(img.shape)

    img = img.transpose(2, 1, 0)
    img_output = img.copy()
    img = img.flatten()
    arr = numpy_support.numpy_to_vtk(img, array_type=vtk.VTK_FLOAT)
    arr.SetName('Scalar')
    imgdata.GetPointData().SetScalars(arr)

    # Create 3d mesh
    cf = vtk.vtkContourFilter()
    cf.SetInputData(imgdata)
    cf.SetValue(0, 0.5)
    cf.Update()

    polydata = cf.GetOutput()

    # Calculate the mesh centroid
    xo, yo, zo = 0, 0, 0
    for i in range(polydata.GetNumberOfPoints()):
        x, y, z = polydata.GetPoints().GetPoint(i)
        xo += x
        yo += y
        zo += z
    xo /= polydata.GetNumberOfPoints()
    yo /= polydata.GetNumberOfPoints()
    zo /= polydata.GetNumberOfPoints()
    centroid = (xo, yo, zo)
    
    # Translate to origin
    if translate_to_origin:
        for i in range(polydata.GetNumberOfPoints()):
            x, y, z = polydata.GetPoints().GetPoint(i)
            polydata.GetPoints().SetPoint(i, x-xo, y-yo, z-zo)

    return polydata, img_output, centroid

def align_points(x, y, z, from_pc=0, to_axis=0):

    """

        Aligns a set of 3d points based on their principal components.
        The alignment is made by rotating the points in such a way that the
        pricipal component `from_pc` ends up aligned with the cartesian axis
        `to_axis`.

        We also flip the points coordinates in each axes if necessary to
        maintain positive the correlation coefficient on each pairwise
        projection.

        Parameters
        ----------
        x, y and z : ndarray
            Input array with x, y and z coordinates of the points

        Returns
        -------
        x_rot, y_rot and z_rot : ndarray
            Aligned x, y and z coordinates of the points

        Other parameters
        ----------------
        from_pc : {0,1,2}
            Principal component which will be aligned with the cartesian
            axis specified by `to_axis`. 0 - pc with largest variance, 2 - pc
            with smallest variance, default is 0.
        to_axis : {0,1,2}
            Cartesian axis of alignment. 0 = x, 1 = y and 2 = z. Default is 0.

    """

    xyz = np.hstack([
        x.reshape(-1, 1),
        y.reshape(-1, 1),
        z.reshape(-1, 1)])

    cartesian_axes = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

    eigenvecs = skdecomp.PCA(n_components=3).fit(xyz).components_

    # Make sure aigenvectors are unitary
    theta = np.arccos(np.clip(np.dot(eigenvecs[from_pc], cartesian_axes[to_axis]), -1.0, 1.0))

    # Vectorial product to get the pivot vector
    pivot = np.cross(eigenvecs[from_pc], cartesian_axes[to_axis])
    
    # Check if points are already aligned
    if np.square(pivot).sum() < 1e-8:
        return x, y, z

    rot_mx = rotation_matrix(pivot, theta)
                
    xyz_rot = np.dot(rot_mx, xyz.T).T

    # Correlation coeff xz, yz:

    for ax in [0,1]:
        if np.corrcoef(xyz_rot[:, ax], xyz_rot[:, 2])[0, 1] < 0.0:
            xyz_rot[:, ax] *= -1

    return xyz_rot[:, 0], xyz_rot[:, 1], xyz_rot[:, 2]


def align_points_2d(x, y, z):

    """

        Aligns a set of 3d points based on their two principal components
        calculated on the xy plane.
        The alignment is made by rotating the points in such a way that the
        1st pricipal component ends up aligned with the x-axis and the 2nd
        principal component is aligned with y-axis.

        We also flip the points coordinates in each axes if necessary to
        maintain positive the correlation coefficient on each pairwise
        projection.

        Parameters
        ----------
        x, y and z : ndarray
            Input array with x, y and z coordinates of the points
        Returns
        -------
        x_rot and y_rot : ndarray
            Aligned x and y coordinates of the points.

    """

    xy = np.hstack([
        x.reshape(-1, 1),
        y.reshape(-1, 1)])

    eigenvecs = skdecomp.PCA(n_components=2).fit(xy).components_

    theta_proj = -np.arctan2(eigenvecs[0][1],eigenvecs[0][0])

    rot_mx = [
        [np.cos(theta_proj), np.sin(-theta_proj)],
        [np.sin(theta_proj), np.cos(theta_proj)]]
        
    xy_rot = np.dot(rot_mx, xy.T).T

    # Correlation coeff xz, yz:

    for ax in [0, 1]:
        if np.corrcoef(xy_rot[:,ax], z)[0, 1] < 0.0:
            xy_rot[:, ax] *= -1

    return xy_rot[:, 0], xy_rot[:, 1]

'''
    
'''

def update_mesh_points(mesh, x_new, y_new, z_new):

    """

        Updates the xyz coordinates of points in the input mesh
        with new coordinates provided.

        Parameters
        ----------
        mesh : vtkPolyData
            Mesh in VTK format to be updated.
        x_new, y_new and z_new : ndarray
            Array containing the new coordinates.
        Returns
        -------
        mesh_updated : vtkPolyData
            Mesh with updated coordinates.
        
        Notes
        -----
        This function also re-calculate the new normal vectors
        for the updated mesh.
    """

    for i in range(mesh.GetNumberOfPoints()):
        mesh.GetPoints().SetPoint(i, x_new[i], y_new[i], z_new[i])
    mesh.Modified()

    # Fix normal vectors orientation

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.Update()

    mesh_updated = normals.GetOutput()

    return mesh_updated

'''
    
'''

def get_reconstruction_from_grid(grid, centroid=(0, 0, 0)):

    """

        Converts a parametric 2D grid of type (lon,lat,rad) into
        a 3d mesh.

        Parameters
        ----------
        grid : ndarray
            Input grid where the element grid[i,j] represents the
            radial coordinate at longitude i*2pi/grid.shape[0] and
            latitude j*pi/grid.shape[1].

        Returns
        -------
        mesh : vtkPolyData
            Mesh that represents the input parametric grid.
        
        Other parameters
        ----------------
        centroid : tuple of floats, optional
            x, y and z coordinates of the centroid where the mesh
            will be translated to, default is (0,0,0).

    """

    res_lat = grid.shape[0]
    res_lon = grid.shape[1]

    # Creates an initial spherical mesh with right dimensions.
    rec = vtk.vtkSphereSource()
    rec.SetPhiResolution(res_lat+2)
    rec.SetThetaResolution(res_lon)
    rec.Update()
    rec = rec.GetOutput()

    n = rec.GetNumberOfPoints()

    grid_ = grid.T.flatten()

    # Update the points coordinates of the spherical mesh according to the inout grid
    for j, lon in enumerate(np.linspace(0, 2*np.pi, num=res_lon, endpoint=False)):
        for i, lat in enumerate(np.linspace(np.pi/(res_lat+1), np.pi, num=res_lat, endpoint=False)):
            theta = lat
            phi = lon - np.pi
            k = j * res_lat + i
            x = centroid[0] + grid_[k] * np.sin(theta)*np.cos(phi)
            y = centroid[1] + grid_[k] * np.sin(theta)*np.sin(phi)
            z = centroid[2] + grid_[k] * np.cos(theta)
            rec.GetPoints().SetPoint(k+2, x, y, z)
    # Update coordinates of north and south pole points
    north = grid_[::res_lat].mean()
    south = grid_[res_lat-1::res_lat].mean()
    rec.GetPoints().SetPoint(0, centroid[0]+0, centroid[1]+0, centroid[2]+north)
    rec.GetPoints().SetPoint(1, centroid[0]+0, centroid[1]+0, centroid[2]-south)

    # Compute normal vectors
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    normals.Update()

    mesh = normals.GetOutput()

    return mesh

def get_reconstruction_from_coeffs(coeffs, lrec=0):

    """

        Converts a set of spherical harmonic coefficients into
        a 3d mesh.

        Parameters
        ----------
        coeffs : ndarray
            Input array of spherical harmonic coefficients. These
            array has dimensions 2xLxM, where the first dimension
            is 0 for cosine-associated coefficients and 1 for 
            sine-associated coefficients. Second and thrid dimensions
            represent the expansion parameters (l,m).

        Returns
        -------
        mesh : vtkPolyData
            Mesh that represents the input parametric grid.

        Other parameters
        ----------------
        lrec : int, optional
            Only coefficients l<lrec will be used for creating the
            mesh, default is 0 meaning all coefficients available
            in the matrix coefficients will be used.
        
        Notes
        -----
            The mesh resolution is set by the size of the coefficients
            matrix and therefore not affected by lrec.
            
    """

    coeffs_ = coeffs.copy()

    if (lrec > 0) and (lrec < coeffs_.shape[1]):
        coeffs_[:, lrec:, :] = 0

    grid = pyshtools.expand.MakeGridDH(coeffs_, sampling=2)
    mesh = get_reconstruction_from_grid(grid)
    
    return mesh, grid

def get_reconstruction_error(grid_input, grid_rec):

    """

        Compute mean square error between two parametric grids.

        Parameters
        ----------
        grid_input : ndarray
            Parametric grid
        grid_rec : ndarray
            Reconstructed parametric grid

        Returns
        -------
        mse : float
            Mean square error

    """

    mse = np.power(grid_input - grid_rec, 2).mean()

    return mse

def save_polydata(mesh, filename):

    """

        Saves a mesh as a vtkPolyData file.

        Parameters
        ----------
        mesh : vtkPolyData
            Input mesh
        filename : str
            File path where the mesh will be saved

    """

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(filename)
    writer.Write()

