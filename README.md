# AICS Spherical Hamonics Parametrization

Calculates the coefficients of the spherical harmonic expansion that describes a 3d object stored in a binary image. The method works best for startlike objects.

### Example

import numpy as np

import aicsshparam

img = np.ones((32,32,32), dtype=np.uint8)

(coeffs, grid_rec), (image_, mesh, centroid, grid) = aicsshparam.get_shcoeffs(image=img, lmax=2)

mse = aicsshparam.aicsshtools.get_reconstruction_error(grid,grid_rec)

print('SH coefficients:', coeffs)

print('Mean square error:', mse)
```