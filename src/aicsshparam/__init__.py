# -*- coding: utf-8 -*-

"""Top-level package for aics-shparam."""

__author__ = "Matheus Viana"
__email__ = "matheus.viana@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.11"

from .shparam import get_shcoeffs_from_mesh, get_shcoeffs_from_vertices_faces  # noqa: E402,F401


def get_module_version():
    return __version__
