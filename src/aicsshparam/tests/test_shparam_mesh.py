"""Tests for the mesh-based SPHARM entry points in aicsshparam.shparam.

Covers:
- get_shcoeffs_from_mesh / get_shcoeffs_from_vertices_faces happy path
- Return-value shape matches the image path (image_ slot is None)
- Consistency with the image path on a shared mesh
- Hard-error validation for malformed vertices/faces
- Warnings for non-watertight meshes and off-center centroids
"""

import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from aicsshparam import shparam, shtools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_vertices_faces(radius=10.0, theta_res=24, phi_res=24):
    """Build a triangulated sphere as (vertices, faces) numpy arrays."""
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetThetaResolution(theta_res)
    source.SetPhiResolution(phi_res)
    source.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(source.GetOutput())
    tri.Update()
    poly = tri.GetOutput()

    vertices = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())

    faces = numpy_support.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3), "Expected triangular faces"
    faces = faces[:, 1:]

    return vertices.astype(float), faces.astype(np.int64)


def _sphere_image(radius=10, shape=32):
    """Build a binary ZYX image containing a centered sphere."""
    zz, yy, xx = np.indices((shape, shape, shape))
    c = shape // 2
    img = ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) <= radius**2
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_vertices_faces_happy_path():
    vertices, faces = _sphere_vertices_faces()

    (coeffs, grid_rec), (image_, mesh, grid_down, transform) = (
        shparam.get_shcoeffs_from_vertices_faces(vertices, faces, lmax=4)
    )

    # image_ slot must be None for mesh input (shape parity with get_shcoeffs).
    assert image_ is None
    assert isinstance(coeffs, dict)
    assert "shcoeffs_L0M0C" in coeffs
    # transform carries centroid + angle when aligned.
    assert len(transform) == 4

    # A sphere should reconstruct well: L0M0C dominates, error is small.
    mse = shtools.get_reconstruction_error(grid_down, grid_rec)
    assert mse < 1.0
    l0 = abs(coeffs["shcoeffs_L0M0C"])
    assert l0 > 0
    # Higher-order energy is tiny relative to the mean radius term.
    high = sum(abs(v) for k, v in coeffs.items() if not k.startswith("shcoeffs_L0"))
    assert high < 0.1 * l0


def test_alignment_2d_false_drops_angle():
    vertices, faces = _sphere_vertices_faces()

    (_, _), (_, _, _, transform) = shparam.get_shcoeffs_from_vertices_faces(
        vertices, faces, lmax=2, alignment_2d=False
    )

    assert len(transform) == 3  # only the centroid, no angle


def test_consistency_with_image_path():
    """Feeding the image path's mesh to the mesh path should reproduce
    essentially the same coefficients."""
    img = _sphere_image()

    (coeffs_img, _), (_, mesh, _, _) = shparam.get_shcoeffs(image=img, lmax=4)

    # mesh from get_shcoeffs is already centered + aligned.
    (coeffs_mesh, _), (image_, _, _, _) = shparam.get_shcoeffs_from_mesh(
        mesh=mesh, lmax=4
    )

    assert image_ is None
    for key in coeffs_img:
        assert coeffs_mesh[key] == pytest.approx(coeffs_img[key], abs=1e-6)


def test_does_not_mutate_input_mesh():
    vertices, faces = _sphere_vertices_faces()
    mesh = shtools.get_mesh_from_vertices_faces(
        vertices, faces, translate_to_origin=False
    )
    before = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData()).copy()

    shparam.get_shcoeffs_from_mesh(mesh=mesh, lmax=2)

    after = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------------
# Validation: hard errors
# ---------------------------------------------------------------------------


def test_bad_vertices_shape_raises():
    with pytest.raises(ValueError):
        shtools.get_mesh_from_vertices_faces(np.zeros((5, 2)), np.zeros((1, 3)))


def test_bad_faces_shape_raises():
    vertices, _ = _sphere_vertices_faces()
    with pytest.raises(ValueError):
        shtools.get_mesh_from_vertices_faces(vertices, np.zeros((4, 4), dtype=int))


def test_out_of_range_face_index_raises():
    vertices, faces = _sphere_vertices_faces()
    faces = faces.copy()
    faces[0, 0] = len(vertices) + 10
    with pytest.raises(ValueError):
        shtools.get_mesh_from_vertices_faces(vertices, faces)


def test_non_finite_vertices_raises():
    vertices, faces = _sphere_vertices_faces()
    vertices = vertices.copy()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError):
        shtools.get_mesh_from_vertices_faces(vertices, faces)


def test_empty_mesh_raises():
    empty = vtk.vtkPolyData()
    empty.SetPoints(vtk.vtkPoints())
    with pytest.raises(ValueError):
        shparam.get_shcoeffs_from_mesh(mesh=empty, lmax=2)


# ---------------------------------------------------------------------------
# Validation: warnings (compute still proceeds)
# ---------------------------------------------------------------------------


def test_open_mesh_warns():
    """A single triangle is an open surface and should warn, not raise."""
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = shtools.get_mesh_from_vertices_faces(vertices, faces)

    with pytest.warns(UserWarning, match="closed manifold"):
        shtools.check_mesh_for_parametrization(mesh)


def test_closed_centered_mesh_no_warning():
    vertices, faces = _sphere_vertices_faces()
    mesh = shtools.get_mesh_from_vertices_faces(vertices, faces)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        shtools.check_mesh_for_parametrization(mesh)
