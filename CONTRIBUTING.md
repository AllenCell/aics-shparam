# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!

Ready to contribute? Here's how to set up `aicsshparam` for local development.

1. Fork the `aicsshparam` repo on GitHub.

2. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/aics-shparam.git
    ```

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd src/aicsshparam/
    pip install -e .[dev]
    ```

4. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests, including testing other Python versions with make:

    ```bash
    make build
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Resolves gh-###. Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

7. Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.

1. **Ensure all changes are committed.**  
   Refer to [bump-my-version on PyPI](https://pypi.org/project/bump-my-version/0.9.1/) for more details on versioning.

2. **Bump the version number.**  
   Run one of the following commands depending on the type of version update:
   ```bash
   bump-my-version major # for major releases with breaking changes
   bump-my-version minor # for minor releases with new features
   bump-my-version patch # for patch releases with bug fixes
   ```

3. **Push the changes and tags to the repository.**
   ```bash
   git push
   git push --tags
   ```

---

This will release a new package version on Git and GitHub. When your commit message includes "Bump version," it triggers a GitHub Action that automatically publishes the package to PyPI!