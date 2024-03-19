import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import animation, colormaps
from scipy import stats as spstats
from aicsimageio import AICSImage, writers
from . import io

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"

class PlotMaker(io.DataProducer):
    """
    Support class. Should not be instantiated directly.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    dpi = 72

    def __init__(self, control):
        super().__init__(control)
        self.df = None
        self.figs = []
        self.dataframes = []

    def workflow(self):
        pass

    def execute(self, display=True, **kwargs):
        if "dpi" in kwargs:
            self.dpi = kwargs["dpi"]
        prefix = None if "prefix" not in kwargs else kwargs["prefix"]
        self.workflow()
        self.save(display, prefix)
        self.figs = []

    def save(self, display=True, prefix=None, full_path_provided=False):
        for (fig, signature) in self.figs:
            if display:
                fig.show()
            else:
                fname = signature
                if hasattr(self, "full_path_provided") and self.full_path_provided:
                    save_dir = self.output_folder
                else:
                    save_dir = self.control.get_staging()/self.subfolder
                save_dir.mkdir(parents=True, exist_ok=True)
                if prefix is not None:
                    fname = f"{prefix}_{signature}"
                fig.savefig(save_dir/f"{fname}.png", facecolor="white")
                fig.savefig(save_dir/f"{fname}.pdf", facecolor="white")
                plt.close(fig)
        for (df, signature) in self.dataframes:
            fname = signature
            if hasattr(self, "full_path_provided") and self.full_path_provided:
                save_dir = self.output_folder
            else:
                save_dir = self.control.get_staging()/self.subfolder
            save_dir.mkdir(parents=True, exist_ok=True)
            if prefix is not None:
                fname = f"{prefix}_{signature}"
            df.to_csv(save_dir/f"{fname}.csv")


class ShapeSpacePlotMaker(PlotMaker):
    """
    Class for creating plots for shape space.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, control, subfolder: Optional[str] = None):
        super().__init__(control)
        if subfolder is None:
            self.subfolder = "shapemode/pca"
        else:
            self.subfolder = subfolder

    def workflow(self):
        return

    def plot_explained_variance(self, space):
        npcs = self.control.get_number_of_shape_modes()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=self.dpi)
        ax.plot(100 * space.pca.explained_variance_ratio_[:npcs], "-o")
        title = "Cum. variance: (1+2) = {0}%, Total = {1}%".format(
            int(100 * space.pca.explained_variance_ratio_[:2].sum()),
            int(100 * space.pca.explained_variance_ratio_[:].sum()),
        )
        ax.set_xlabel("Component", fontsize=18)
        ax.set_ylabel("Explained variance (%)", fontsize=18)
        ax.set_xticks(np.arange(npcs))
        ax.set_xticklabels(np.arange(1, 1 + npcs))
        ax.set_title(title, fontsize=18)
        plt.tight_layout()
        self.figs.append((fig, "explained_variance"))
        return

    def save_feature_importance(self, space):
        path = f"{self.subfolder}/feature_importance.txt"
        abs_path_txt_file = self.control.get_staging() / path
        os.makedirs(Path(abs_path_txt_file.parent), exist_ok=True)
        with open(abs_path_txt_file, "w") as flog:
            for col, sm in enumerate(self.control.iter_shape_modes()):
                exp_var = 100 * space.pca.explained_variance_ratio_[col]
                print(f"\nExplained variance {sm}={exp_var:.1f}%", file=flog)
                '''_PC: raw loading, _aPC: absolute loading and
                _cPC: normalized cummulative loading'''
                pc_name = space.axes.columns[col]
                df_sorted = space.df_feats.sort_values(
                    by=[pc_name.replace("_PC", "_aPC")], ascending=False
                )
                pca_cum_contrib = np.cumsum(
                    df_sorted[pc_name.replace("_PC", "_aPC")].values /
                    df_sorted[pc_name.replace("_PC", "_aPC")].sum()
                )
                pca_cum_thresh = np.abs(pca_cum_contrib - 0.80).argmin()
                df_sorted = df_sorted.head(n=pca_cum_thresh + 1)
                print(df_sorted[[
                    pc_name,
                    pc_name.replace("_PC", "_aPC"),
                    pc_name.replace("_PC", "_cPC"), ]].head(), file=flog
                )
        return

    def plot_pairwise_correlations(self, space, off=0):
        df = space.shape_modes
        nf = len(df.columns)
        if nf < 2:
            return
        npts = df.shape[0]
        cmap = colormaps["tab10"]
        prange = []
        for f in df.columns:
            prange.append(np.percentile(df[f].values, [off, 100 - off]))
        # Create a grid of nfxnf
        fig, axs = plt.subplots(nf, nf, figsize=(2 * nf, 2 * nf), sharex="col",
                                gridspec_kw={"hspace": 0.1, "wspace": 0.1},
                                )
        for f1id, f1 in enumerate(df.columns):
            yrange = []
            for f2id, f2 in enumerate(df.columns):
                ax = axs[f1id, f2id]
                y = df[f1].values
                x = df[f2].values
                valids = np.where((
                    (y > prange[f1id][0]) &
                    (y < prange[f1id][1]) &
                    (x > prange[f2id][0]) &
                    (x < prange[f2id][1])))
                if f2id < f1id:
                    xmin = x[valids].min()
                    xmax = x[valids].max()
                    ymin = y[valids].min()
                    ymax = y[valids].max()
                    yrange.append([ymin, ymax])
                    ax.plot(x[valids], y[valids], ".",
                            markersize=2, color="black", alpha=0.8)
                    ax.plot([xmin, xmax], [xmin, xmax], "--")
                    if f2id:
                        plt.setp(ax.get_yticklabels(), visible=False)
                        ax.tick_params(axis="y", which="both", length=0.0)
                    if f1id < nf - 1:
                        ax.tick_params(axis="x", which="both", length=0.0)
                # Add annotations on upper triangle
                elif f2id > f1id:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis="x", which="both", length=0.0)
                    ax.tick_params(axis="y", which="both", length=0.0)
                    pearson, p_pvalue = spstats.pearsonr(x, y)
                    spearman, s_pvalue = spstats.spearmanr(x, y)
                    ax.text(0.05, 0.8, f"Pearson: {pearson:.2f}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                    ax.text(0.05, 0.6, f"P-value: {p_pvalue:.1E}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                    ax.text(0.05, 0.4, f"Spearman: {spearman:.2f}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                    ax.text(0.05, 0.2, f"P-value: {s_pvalue:.1E}", size=10, ha="left",
                            transform=ax.transAxes,
                            )
                # Single variable distribution at diagonal
                else:
                    ax.set_frame_on(False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.tick_params(axis="y", which="both", length=0.0)
                    ax.hist(x[valids], bins=16, density=True, histtype="stepfilled",
                            color="white", edgecolor="black", label="Complete",
                            )
                    ax.hist(x[valids], bins=16, density=True, histtype="stepfilled",
                            color=cmap(0), alpha=0.2, label="Incomplete",
                            )
                if f1id == nf - 1:
                    ax.set_xlabel(f2, fontsize=7)
                if not f2id and f1id:
                    ax.set_ylabel(f1, fontsize=7)
            if yrange:
                ymin = np.min([ymin for (ymin, ymax) in yrange])
                ymax = np.max([ymax for (ymin, ymax) in yrange])
                for f2id, f2 in enumerate(df.columns):
                    ax = axs[f1id, f2id]
                    if f2id < f1id:
                        ax.set_ylim(ymin, ymax)

        # Global annotation
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", top=False,
                        bottom=False, left=False, right=False)
        plt.title(f"Total number of points: {npts}", fontsize=24)

        self.figs.append((fig, "pairwise_correlations"))
        return


class ShapeModePlotMaker(PlotMaker):
    """
    Class for creating plots for shape mode step.

    WARNING: This class should not depend on where
    the local_staging folder is.
    """

    def __init__(self, control, subfolder: Optional[str] = None):
        super().__init__(control)
        if subfolder is None:
            self.subfolder = "shapemode/avgshape"
        else:
            self.subfolder = subfolder

    def workflow(self):
        return

    def animate_contours(self, contours, prefix):
        hmin, hmax, vmin, vmax = self.control.get_plot_limits()
        offset = 0.05 * (hmax - hmin)

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.tight_layout()
        plt.close()
        ax.set_xlim(hmin - offset, hmax + offset)
        ax.set_ylim(vmin - offset, vmax + offset)
        ax.set_aspect("equal")
        if not self.control.get_plot_frame():
            ax.axis("off")

        lines = []
        for alias, _ in contours.items():
            color = self.control.get_color_from_alias(alias)
            (line,) = ax.plot([], [], lw=2, color=color)
            lines.append(line)

        def animate(i):
            for alias, line in zip(contours.keys(), lines):
                ct = contours[alias][i]
                mx = ct[:, 0]
                my = ct[:, 1]
                line.set_data(mx, my)
            return lines

        n = self.control.get_number_of_map_points()
        anim = animation.FuncAnimation(
            fig, animate, frames=n, interval=100, blit=True
        )
        fname = self.control.get_staging() / f"{self.subfolder}/{prefix}.gif"
        anim.save(fname, fps=n)
        plt.close("all")
        return

    def load_animated_gif(self, shape_mode, proj):
        fname = self.control.get_staging() / f"{self.subfolder}/{shape_mode}_{proj}.gif"
        image = AICSImage(fname).data.squeeze()
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return image

    def combine_and_save_animated_gifs(self):
        stack = []
        for sm in tqdm(self.control.get_shape_modes()):
            imx = self.load_animated_gif(sm, "x")
            imy = self.load_animated_gif(sm, "y")
            imz = self.load_animated_gif(sm, "z")
            img = np.concatenate([imz, imy, imx], axis=-2)
            stack.append(img)
        stack = np.array(stack)
        stack = np.concatenate(stack[:], axis=-3)
        stack = np.rollaxis(stack, -1, 1)
        fname = self.control.get_staging() / f"{self.subfolder}/combined.tif"
        os.makedirs(fname.parent, exist_ok=True)
        writers.ome_tiff_writer.OmeTiffWriter.save(stack, fname, overwrite_file=True, dim_order="CZYX")
        return
