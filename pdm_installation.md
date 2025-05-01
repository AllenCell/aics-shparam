# Installation and Usage with PDM

1. [Install PDM](https://pdm-project.org/en/latest/#recommended-installation-method).  
   (Make sure that `pdm-backend` is installed prior to synchronizing dependencies)

2. Install project dependencies by running:  
   `pdm sync --no-isolation`

3. Prefix every `python` command with `pdm run`. For example:

   ```
   pdm run python src/aicsshparam/tests/dummy_test.py
   ```

Note: The current lock file includes all dependencies. If you only need the runtime dependencies, delete the existing lock file and generate a new one running `pdm lock`.
