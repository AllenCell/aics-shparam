import vtk
import pyshtools
import numpy as np
import pandas as pd
from vtk.util import numpy_support
from scipy import stats as spstats
from skimage import filters as skfilters
from skimage import morphology as skmorpho
from sklearn import decomposition as skdecomp

'''
    Creates a rotation matrix in 3D. This will rotate an objact about the
axis "axis" by an angle "angle" (rad).
'''

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

'''
    Converts a numpy array into a vtkimagedata and extract an isosurface.
    
    The input is assumed to be binary and the isosurface is set to 0.5.
    
    Optionally the input can be pre-processed by extracting the largest
    connected component and applying a gaussian smooth to it.
    
    A size threshold is applying to garantee that enough points will be
    used to compute the SH parametrization.

    Also, points as the edge of the image are set to zero (background)
'''

def get_polydata_from_numpy(volume, lcc=False, sigma=0, center=False, size_threshold=0):

    volume = np.swapaxes(volume,0,2)

    if lcc:

        volume = skmorpho.label(volume)

        counts = np.bincount(volume.reshape(-1))
        lcc = 1 + np.argmax(counts[1:])

        volume[volume!=lcc] = 0
        volume[volume==lcc] = 1

    if sigma:

        volume = volume.astype(np.float32)
        volume = skfilters.gaussian(volume,sigma=(sigma,sigma,sigma))
        volume[volume<1.0/np.exp(1.0)] = 0
        volume[volume>0] = 1
        volume = volume.astype(np.uint8)

    volume[[0,-1],:,:] = 0
    volume[:,[0,-1],:] = 0
    volume[:,:,[0,-1]] = 0
    volume = volume.astype(np.float32)

    if volume.sum() < size_threshold:
        return None, (None,None)

    img = vtk.vtkImageData()
    img.SetDimensions(volume.shape)

    volume = volume.transpose(2,1,0)
    volume_output = volume.copy()
    volume = volume.flatten()
    arr = numpy_support.numpy_to_vtk(volume, array_type=vtk.VTK_FLOAT)
    arr.SetName('Scalar')
    img.GetPointData().SetScalars(arr)

    cf = vtk.vtkContourFilter()
    cf.SetInputData(img)
    cf.SetValue(0, 0.5)
    cf.Update()

    polydata = cf.GetOutput()

    xo, yo, zo = 0, 0, 0
    if center: # Translate the surface to the provided center point.
        for i in range(polydata.GetNumberOfPoints()):
            x, y, z = polydata.GetPoints().GetPoint(i)
            xo += x
            yo += y
            zo += z
        xo /= polydata.GetNumberOfPoints()
        yo /= polydata.GetNumberOfPoints()
        zo /= polydata.GetNumberOfPoints()
        for i in range(polydata.GetNumberOfPoints()):
            x, y, z = polydata.GetPoints().GetPoint(i)
            polydata.GetPoints().SetPoint(i,x-xo,y-yo,z-zo)

    return polydata, (volume_output,(xo,yo,zo))

'''
    Points alignment is critical for invariant SH parametrization. Given the
    biological relevance of the z axis, we always preserve this orientarion
    and therefore only 2d alignment is performed.

    The alignment is based on PCA and then we flip the shape if necessary
    to maintain the correlation coefficient of axis xz and yz always positive.
'''

def align_points(x, y, z, from_axis=0, to_axis=0, align3d=False):
   
    df = pd.DataFrame({'x':x,'y':y,'z':z})

    if align3d:

        cartesian_axes = np.array([[1,0,0],[0,1,0],[0,0,1]])

        eigenvecs = skdecomp.PCA(n_components=3).fit(df.values).components_

        theta = np.arccos(np.clip(np.dot(eigenvecs[from_axis], cartesian_axes[to_axis]), -1.0, 1.0))

        pivot = np.cross(eigenvecs[from_axis], cartesian_axes[to_axis])
        
        rot_mx = rotation_matrix(pivot, theta)
        
    else:

        eigenvecs = skdecomp.PCA(n_components=2).fit(df[['x','y']].values).components_

        theta_proj = -np.arctan2(eigenvecs[0][1],eigenvecs[0][0])

        rot_mx = [[np.cos(theta_proj),np.sin(-theta_proj),0],[np.sin(theta_proj),np.cos(theta_proj),0],[0,0,1]]
        
    xyz_rot = np.dot(rot_mx, df.values.T).T

    # Post alignment processing:

    # Correlation coeff:

    for ax in [0,1]:
        if np.corrcoef(xyz_rot[:,ax],xyz_rot[:,2])[0,1] < 0.0:
            xyz_rot[:,ax] *= -1

    if align3d:
        if np.corrcoef(xyz_rot[:,0],xyz_rot[:,1])[0,1] < 0.0:
            xyz_rot[:,2] *= -1

    # Skewness (not as good as shown in assay-dev-tests/shcoeffs/skewness-vs-corrcoef.ipynb)

    fx, fy = 1, 1

    # for fx,fy in [[1,1],[-1,1],[1,-1],[-1,-1]]:
    #     skx = spstats.skew(fx*xyz_rot[:,0])
    #     sky = spstats.skew(fy*xyz_rot[:,1])
    #     if (skx>0) & (sky>0):
    #         break

    return fx*xyz_rot[:,0], fy*xyz_rot[:,1], xyz_rot[:,2]

'''
    Update the points of a mesh and calculate the new normal vectors.
'''

def update_mesh_points(mesh, x_new, y_new, z_new):

    for i in range(mesh.GetNumberOfPoints()):
        mesh.GetPoints().SetPoint(i,x_new[i],y_new[i],z_new[i])
    mesh.Modified()

    # Fix normal vectors orientation

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.Update()

    return normals.GetOutput()

'''
    This function returns a mesh represented by a parametric 2D grid.
'''

def get_reconstruction_from_grid(grid, cm=(0,0,0)):

    rec = vtk.vtkSphereSource()
    rec.SetPhiResolution(grid.shape[0]+2)
    rec.SetThetaResolution(grid.shape[1])
    rec.Update()
    rec = rec.GetOutput()

    n = rec.GetNumberOfPoints()
    res_lat = grid.shape[0]
    res_lon = grid.shape[1]

    grid = grid.T.flatten()

    # Coordinates

    for j, lon in enumerate(np.linspace(0, 2*np.pi, num=res_lon, endpoint=False)):
        for i, lat in enumerate(np.linspace(np.pi/(res_lat+1), np.pi, num=res_lat, endpoint=False)):
            theta = lat
            phi = lon - np.pi
            k = j * res_lat + i
            x = cm[0] + grid[k] * np.sin(theta)*np.cos(phi)
            y = cm[1] + grid[k] * np.sin(theta)*np.sin(phi)
            z = cm[2] + grid[k] * np.cos(theta)
            rec.GetPoints().SetPoint(k+2,x,y,z)
    north = grid[::res_lat].mean()
    south = grid[res_lat-1::res_lat].mean()
    rec.GetPoints().SetPoint(0,cm[0]+0,cm[1]+0,cm[2]+north)
    rec.GetPoints().SetPoint(1,cm[0]+0,cm[1]+0,cm[2]-south)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    normals.Update()

    return normals.GetOutput()

'''
    This function returns a mesh described by sh coefficients stored
    in a pandas dataframe.
'''

def get_reconstruction_from_dataframe(df, cm=(0,0,0)):

    l_values = [fn.replace('shcoeffs_L','').split('M')[0] for fn in df.columns if 'shcoeffs_L' in fn]

    l_values = [int(l) for l in l_values]

    lmax = 1 + np.max(l_values)

    coeffs = np.zeros((2,lmax,lmax), dtype=np.float32)

    for id_type, coeff_type in enumerate(['C','S']): # cosine and sine

        for l in range(lmax):

            for m in range(lmax):

                var = [fn for fn in df.columns if f'L{l}M{m}{coeff_type}' in fn]

                coeffs[id_type,l,m] = df[var].values[0]

    grid = pyshtools.expand.MakeGridDH(coeffs, sampling=2)
    
    mesh = get_reconstruction_from_grid(grid)

    return mesh

'''
    Save meshes as vtk files.
'''

def save_polydata(mesh, filename):

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(filename)
    writer.Write()