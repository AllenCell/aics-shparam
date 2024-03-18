import numpy as np
import multiprocessing
from pathlib import Path


class Controller:
    """
    Functionalities for communicating with the config
    file.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, config):
        self.config = config
        self.config['log'] = {}
        self.set_abs_path_to_local_staging_folder(config['project']['local_staging'])
        self.data_section = self.config['data']
        self.features_section = self.config['features']
        self.space_section = self.config['shapespace']
        self.param_section = self.config['parameterization']

    def set_abs_path_to_local_staging_folder(self, path):
        self.abs_path_local_staging = Path(path)

    def get_abs_path_to_local_staging_folder(self):
        return self.abs_path_local_staging

    def get_staging(self):  # shortcut
        return self.get_abs_path_to_local_staging_folder()

    def overwrite(self):
        return self.config['project']['overwrite']

    def get_data_name_alias_dict(self):
        return self.data_section

    def get_data_alias_name_dict(self):
        named_aliases = self.get_data_name_alias_dict()
        return dict([(v['alias'], k) for k, v in named_aliases.items()])

    def get_name_from_alias(self, alias):
        return self.get_data_alias_name_dict()[alias]

    def get_color_from_alias(self, alias):
        return self.data_section[self.get_name_from_alias(alias)]['color']

    def get_alignment_reference_name(self):
        return self.features_section['SHE']['alignment']['reference']

    def get_alignment_reference_alias(self):
        name = self.get_alignment_reference_name()
        if name:
            return self.get_data_name_alias_dict()[name]['alias']
        return None

    def get_alignment_moving_aliases(self):
        ref = self.get_alignment_reference_alias()
        aliases = self.get_aliases_with_shcoeffs_available()
        return [a for a in aliases if a != ref]

    def get_aliases_with_shcoeffs_available(self):
        return self.features_section['SHE']['aliases']

    def get_lmax(self):
        return self.features_section['SHE']['lmax']

    def get_aliases_for_pca(self):
        return self.space_section['aliases']

    def get_features_for_pca(self, df):
        prefixes = [f"{alias}_shcoeffs_L" for alias in self.get_aliases_for_pca()]
        return [f for f in df.columns if any(w in f for w in prefixes)]

    def get_shape_modes_prefix(self):
        return "_".join(self.get_aliases_for_pca())

    def get_alias_for_sorting_pca_axes(self):
        return self.space_section['sorter']

    def get_removal_pct(self):
        return self.space_section['removal_pct']

    def get_number_of_shape_modes(self):
        return self.space_section['number_of_shape_modes']

    def get_shape_modes(self):
        p = self.get_shape_modes_prefix()
        return [f"{p}_PC{s}" for s in range(1, 1 + self.get_number_of_shape_modes())]

    def get_map_points(self):
        return self.space_section['map_points']

    def get_map_point_indexes(self):
        return np.arange(1, 1 + self.get_number_of_map_points())

    def get_number_of_map_points(self):
        return len(self.get_map_points())

    def get_center_map_point_index(self):
        return int(0.5 * (self.get_number_of_map_points() + 1))

    def get_plot_limits(self):
        return self.space_section['plot']['limits']

    def get_plot_frame(self):
        return self.space_section['plot']['frame']

    def swapxy_on_zproj(self):
        return self.space_section['plot']['swapxy_on_zproj']

    def iter_shape_modes(self):
        for s in self.get_shape_modes():
            yield s

    def get_variables_values_for_aggregation(self, include_genes=True):
        variables = {}
        variables['shape_mode'] = self.get_shape_modes()
        variables['mpId'] = self.get_map_point_indexes()
        variables['aggtype'] = self.config['aggregation']['type']
        variables['alias'] = self.param_section['parameterize']
        if include_genes:
            structs = self.config['structures']
            variables['structure'] = [k for k in structs.keys()]
        return variables

    def get_gene_names(self):
        return [k for k in self.config['structures'].keys()]

    @staticmethod
    def get_ncores():
        return multiprocessing.cpu_count()
