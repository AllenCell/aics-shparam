from pathlib import Path
import numpy as np
import pandas
from aicsshparam.shapespace import controller, shapespace
from aicsshparam.shapespace.shapemode_tools import ShapeModeCalculator

config = {
    "aggregation": {"type": ["avg"]},
    "appName": "cvapipe_analysis",
    "data": {
        "nucleus": {"alias": "NUC", "channel": "dna_segmentation", "color": "#3AADA7"}
    },
    "features": {
        "SHE": {
            "aliases": ["NUC"],
            "alignment": {"align": True, "reference": "nucleus", "unique": False},
            "lmax": 16,
            "sigma": 2,
        },
        "aliases": ["NUC"],
    },
    "parameterization": {
        "inner": "NUC",
        "number_of_interpolating_points": 32,
        "outer": "MEM",
        "parameterize": ["RAWSTR", "STR"],
    },
    "preprocessing": {
        "filtering": {"csv": "", "filter": False, "specs": {}},
        "remove_mitotics": True,
        "remove_outliers": True,
    },
    "project": {
        "local_staging": Path(__file__).parent / "all/shape_analysis/shape_space",
        "overwrite": True,
    },
    "shapespace": {
        "aliases": ["NUC"],
        "map_points": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "number_of_shape_modes": 8,
        "plot": {
            "frame": False,
            "limits": [-150, 150, -80, 80],
            "swapxy_on_zproj": False,
        },
        "removal_pct": 0.25,
        "sorter": "NUC",
    },
    "structures": {
        "lamin": [
            "nuclear envelope",
            "#084AE7",
            "{'raw': (475,1700), 'seg': (0,30), 'avgseg': (0,60)}",
        ]
    },
}

def random_shcoeffs_dataframe(nrows=100):
    df = pandas.DataFrame({
        "NUC_shape_volume": np.random.normal(size=nrows),
        "NUC_position_depth": np.random.normal(size=nrows),
    })
    for L in range(0, config["features"]["SHE"]["lmax"]+1):
        for m in range(0, config["features"]["SHE"]["lmax"]+1):
            for suffix in ["C", "S"]:
                df[f"NUC_shcoeffs_L{L}M{m}{suffix}"] = np.random.normal(size=nrows)
    return df


def test_shapespace():
    # ARRANGE
    df = random_shcoeffs_dataframe()

    # ACT
    control = controller.Controller(config)
    space = shapespace.ShapeSpace(control)
    space.execute(df)

    # ASSERT
    assert space.shape_modes is not None


def test_shapespace_transform():
    # ARRANGE
    df1 = random_shcoeffs_dataframe()
    df2 = random_shcoeffs_dataframe()

    # ACT
    control = controller.Controller(config)
    space = shapespace.ShapeSpace(control)
    space.execute(df1)
    result = space.pca.transform(df2[[c for c in df2.columns if "shcoeffs" in c]])

    # ASSERT
    assert result is not None


def test_shape_mode_viz():
    # ARRANGE
    df = random_shcoeffs_dataframe()

    # ACT
    control = controller.Controller(config)
    calculator = ShapeModeCalculator(control)
    calculator.set_data(df)
    calculator.execute()

    # ASSERT
    # look for output files
