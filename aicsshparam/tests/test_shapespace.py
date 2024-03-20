import filecmp
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from aicsshparam.shapespace import controller, shapespace
from aicsshparam.shapespace.shapemode_tools import ShapeModeCalculator


config = controller.Config(
    project=controller.Project(
        local_staging=Path(__file__).parent / "all/shape_analysis/shape_space",
        overwrite=True,
    ),
    data={
        "nucleus": controller.DataDefinition(
            alias="NUC",
            channel="dna_segmentation",
            color="#3AADA7"
        )
    },
    features=controller.Features(
        SHE=controller.SphericalHarmonicDefinition(
            aliases=["NUC"],
            lmax=16,
            sigma=2,
            alignment=controller.SphericalHarmonicAlignment(
                align=True,
                reference="nucleus",
                unique=False
            ),
        ),
        aliases=["NUC"],
    ),
    shapespace=controller.ShapeSpace(
        aliases=["NUC"],
        map_points=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        number_of_shape_modes=8,
        plot=controller.PlotDefinition(
            frame=False,  # TODO what is frame for?
            limits=[-150, 150, -80, 80],
            swapxy_on_zproj=False,
        ),
        removal_pct=0.25,
        sorter="NUC",
    ),
    parameterization=["RAWSTR", "STR"],
    structures={
        "lamin": [
            "nuclear envelope",
            "#084AE7",
            "{'raw': (475,1700), 'seg': (0,30), 'avgseg': (0,60)}",
        ]
    },
    aggregation="avg",
)

def random_shcoeffs_dataframe(nrows=100):
    lmax = config.features.SHE.lmax
    shcoeffs_cols = [
        f"NUC_shcoeffs_L{L}M{m}{suffix}"
        for L in range(1, lmax+1)
        for m in range(0, L+1)
        for suffix in ["C", "S"]
        if suffix == "C" or m >= 1  # sine components are zero for m=0
    ]
    df = pd.DataFrame(
        data=np.random.normal(size=(nrows, len(shcoeffs_cols))),
        columns=shcoeffs_cols
    )
    df["NUC_shcoeffs_L0M0C"] = 5.0  # Each row is mostly spherical
    df["NUC_shape_volume"] = np.random.normal(size=nrows)
    df["NUC_position_depth"] = np.random.normal(size=nrows)
    df["structure_name"] = "lamin"
    return df


def assert_NUC_PC_columns_nonzero(df):
   for column in [
        "NUC_PC1",
        "NUC_PC2",
        "NUC_PC3",
        "NUC_PC4",
        "NUC_PC5",
        "NUC_PC6",
        "NUC_PC7",
        "NUC_PC8",
    ]:
        assert df[column].dtype == np.float64
        numzero = sum(df[column] == 0.0)
        # Since we used gaussian data the result should almost always have no zeros
        assert numzero < 3

DATA_PATH = Path(__file__).parent / "data"

def test_shapespace():
    # ARRANGE
    np.random.seed(101)
    df = random_shcoeffs_dataframe(nrows=100)

    # ACT
    control = controller.Controller(config)
    space = shapespace.ShapeSpace(control)
    space.execute(df)

    # ASSERT
    assert len(space.shape_modes) >= 75  # Some outliers filtered out
    assert_NUC_PC_columns_nonzero(space.shape_modes)
    # Check that output is unchanged
    expected = pd.read_csv(DATA_PATH / "shape_modes.csv").set_index("Unnamed: 0")
    expected.index.names = [None]
    pd.testing.assert_frame_equal(space.shape_modes, expected)


def test_shapespace_transform():
    # ARRANGE
    np.random.seed(101)
    df1 = random_shcoeffs_dataframe()
    df2 = random_shcoeffs_dataframe(nrows=100)

    # ACT
    control = controller.Controller(config)
    space = shapespace.ShapeSpace(control)
    space.execute(df1)
    result = space.pca.transform(df2[[c for c in df2.columns if "shcoeffs" in c]].values)

    # ASSERT
    assert len(result) == 100  # No outliers filtered out
    result_as_df = pd.DataFrame(result, columns=[f"NUC_PC{i}" for i in range(1, 9)])
    assert_NUC_PC_columns_nonzero(result_as_df)
    # Check that output is unchanged
    expected = pd.read_csv(DATA_PATH / "transformed.csv").drop(columns=["Unnamed: 0"])
    pd.testing.assert_frame_equal(result_as_df, expected)


def test_shape_mode_viz():
    # ARRANGE
    np.random.seed(101)
    df = random_shcoeffs_dataframe(nrows=50)

    # ACT
    control = controller.Controller(config)
    calculator = ShapeModeCalculator(control)
    calculator.set_data(df)
    calculator.execute(use_vtk_for_intersection=False)

    # ASSERT
    output_directory = config.project.local_staging
    files = [
        output_directory / "shapemode/pca/explained_variance.png",
        output_directory / "shapemode/avgshape/combined.tif",
        output_directory / "shapemode/avgshape/NUC_PC8_z.gif",
    ]
    for file in files:
        assert file.exists()
        assert filecmp.cmp(file, DATA_PATH / file.name, shallow=False)

    # CLEAN UP
    shutil.rmtree(Path(__file__).parent / "all")
