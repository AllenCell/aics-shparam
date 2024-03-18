class DataProducer():
    """
    Functionalities for steps that perform calculations
    in a per row fashion.

    WARNING: All classes are assumed to know the whole
    structure of directories inside the local_staging
    folder and this is hard coded. Therefore, classes
    may break if you move saved files from the places
    their are saved.
    """

    def __init__(self, control):
        self.row = None
        self.control = control
        self.subfolder = None

    def workflow(self):
        pass

    def get_output_file_name(self):
        pass

    def save(self):
        return None

    def set_row(self, row):
        self.row = row
        if "CellIds" in row:
            self.CellIds = self.row.CellIds
            if isinstance(self.CellIds, str):
                self.CellIds = eval(self.CellIds)

    def execute(self, row):
        computed = False
        self.set_row(row)
        path_to_output_file = self.check_output_exist()
        if (path_to_output_file is None) or self.control.overwrite():
            try:
                self.workflow()
                computed = True
                path_to_output_file = self.save()
            except Exception as ex:
                print(f"\n>>>{ex}\n")
                path_to_output_file = None
        self.status(row.name, path_to_output_file, computed)
        return path_to_output_file

    def get_output_file_path(self):
        path = f"{self.subfolder}/{self.get_output_file_name()}"
        return self.control.get_staging() / path

    def check_output_exist(self):
        path_to_output_file = self.get_output_file_path()
        if path_to_output_file.is_file():
            return path_to_output_file
        return None

    @staticmethod
    def status(idx, output, computed):
        msg = "FAIL"
        if output is not None:
            msg = "COMPLETE" if computed else "SKIP"
        print(f"Index {idx} {msg}. Output: {output}")
