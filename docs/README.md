# AICS Spherical Hamonics Parametrization

Spherical harmonics parametrization for starlike shapes. Segmented images are converted into triangular meshes using standard marching cubes algorithm from VTK (vtk.org) and centered at the origin. Coordinates of mesh points are converted from cartesian to geographic system (latitude, longitude and altitude). Altitude coordinates are then interpolated over a (lat,lon) grid where each cell has a resolution of π/128.

We used the Python package pyshtools (github.com/SHTOOLS) to expand the equally spaced grid into spherical harmonics using Driscoll and Healy’s sampling theorem [1].

[1] Driscoll, J.R. and D.M. Healy, Computing Fourier transforms and convolutions on the 2-sphere, Adv. Appl. Math., 15, 202-250, 1994.

```python
	import numpy as np
	from aicsshparam import shparam, shtools

	# Create a binary test image of a cube with size 32x32x32
	img = np.ones((32,32,32), dtype=np.uint8)

	# Calculate the spherical harmonics expansion up to order lmax = 2
	(coeffs, grid_rec), (image_, mesh, grid, transform) = shparam.get_shcoeffs(image=img, lmax=2)

	# Calculate the corresponding reconstruction error
	mse = shtools.get_reconstruction_error(grid,grid_rec)

	# Print results
	print('Coefficients:', coeffs)
	print('Error:', mse)
```

The output `coeffs` is a dict that constains the sh coefficients values. `grid_rec` (ndarray) is the reconstructed equally spaced grid. Additional outputs are `image_` (ndarray) that is the input image after pre-processing, `mesh` (ndarray) is a triangular repsentation of `image_` centered at origin, `grid` (ndarray) is the equally spaced grid corresponding to `image_`, and `transform` is a tuple of floats that contains the centroid of the shape and the angle applied for alignment. The similar `grid_rec` and `grid` are, the better the parametrization is.

To reconstruct the parametrized mesh from `grid_rec` one could do

```python
	# Reconstruct mesh from grid
	mesh_rec = aicsshparam.shtools.get_reconstruction_from_grid(grid_rec)

	# Save mesh
	aicsshparam.shtools.save_polydata(mesh_rec, 'mesh_rec.vtk')
```

VTK mesh files can be open using Paraview (paraview.org) for further inspection.
