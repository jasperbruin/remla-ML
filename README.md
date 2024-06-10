remla-group3
==============================

This is for the Release Engineering Course

## Test Results

![Workflow Update](https://github.com/remla24-team3/model-training/actions/workflows/pytest.yml/badge.svg)
![Test Results](https://remla24-team3.github.io/model-training/badges/badge.svg)

## Project Organization
------------

    ├── LICENSE
    ├── Makefile               <- Makefile with commands like `make data` or `make train`
    ├── README.md              <- The top-level README for developers using this project.
    ├── artifacts              <- Outputs from the machine learning models and data processing.
    │   ├── predicted          <- Contains prediction outputs such as metrics.json.
    │   ├── tokenized          <- Stores tokenized datasets for training, validating, and testing.
    │   └── trained            <- Contains the trained model files like model.keras.
    ├── data
    │   ├── external           <- Data from third party sources.
    │   ├── interim            <- Intermediate data that has been transformed.
    │   ├── processed          <- The final, canonical data sets for modeling.
    │   └── raw                <- The original, immutable data dump.
    ├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
    │   ├── Makefile
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   └── make.bat
    ├── dvc.lock
    ├── dvc.yaml
    ├── models                 <- (Empty directory placeholder, expected to hold model scripts)
    ├── notebooks
    │   └── phishing-detection-cnn.ipynb <- Jupyter notebook for demonstrating model application.
    ├── params.yaml            <- Configuration parameters for the project.
    ├── poetry.lock
    ├── pyproject.toml
    ├── references             <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports
    │   └── figures            <- Generated graphics and figures to be used in reporting.
    ├── setup.py               <- makes project pip installable (pip install -e .) so src can be imported.
    ├── src                    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes src a Python module.
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── download_dataset.py
    │   │   ├── make_dataset.py
    │   │   └── tokenize_dataset.py
    │   ├── features
    │   │   ├── __init__.py
    │   │   └── build_features.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   └── visualization
    │       ├── __init__.py
    │       └── visualize.py
    ├── test_environment.py
    └── tox.ini                <- tox file with settings for running tox; see tox.readthedocs.io



--------

## Using Poetry for Dependency Management

This project uses [Poetry](https://python-poetry.org/) to manage its Python dependencies.

### Installation

1. **Install Poetry**: First, make sure you have Poetry installed on your system. You can install it via pip:

    ```bash
    pip install poetry
    ```

    For alternative installation methods, refer to the [official Poetry documentation](https://python-poetry.org/docs/#installation).

2. **Navigate to the Project Directory**: Change your current directory to the root directory of the cloned repository:

    ```bash
    cd your-path/remla-ML-group3
    ```

3. **Install Dependencies**: Use Poetry to install the project dependencies:

    ```bash
    poetry install
    ```

    This command will create a virtual environment and install all the required dependencies listed in the `pyproject.toml` file.

### Using the Poetry Shell

Poetry provides a virtual environment shell where you can execute commands within the context of your project's dependencies. To activate the Poetry shell, run:

```bash
poetry shell
```

## Essential DVC Commands

Here are the key DVC commands you will use frequently while working with our project:

- **Pull Data/Models from Remote Storage**:
  ```bash
  dvc pull
  ```
  This command downloads the latest versions of data, models, and dependencies from the remote storage to your local environment.

- **Reproduce Experiments**:
  ```bash
  dvc repro
  ```
  Use this command to rerun pipeline stages defined in `dvc.yaml` if their dependencies have changed. This ensures that your project components are up-to-date and that the results are consistent.

- **Push Data/Models to Remote Storage**:
  ```bash
  dvc push
  ```
  After you make changes or updates to data or model files locally, use this command to upload them to the remote storage, keeping it synchronized with your latest developments.

- **Show metrics**:
  ```bash
  dvc metrics show
  ```
  This command will show the results of dvc repro that are collected in `artifacts/predicted/metrics.json`.

# Testing
To ensure that the project is working as expected, you can run the following command:

```bash
 poetry run pytest tests/
```

### Keeping Everything Synchronized

To ensure consistency across all team members' environments:

- Run `dvc pull` to update your local files with the latest versions from the remote storage when you start working or after cloning the repository.
- After significant changes, always use `dvc push` to sync your updates back to the remote storage.

Following these practices will help maintain a smooth and efficient workflow for everyone involved in the project.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Decision documentation
The decision process on how the project is designed is documented here.

### Cookiecutter
For the project templating, [Cookiecutter](https://github.com/cookiecutter/cookiecutter) is used. It standardizes the project setup process with predefined directory structures and configuration files. 

### Poetry
[Poetry](https://python-poetry.org/) is used as a dependency management tool instead of requirements.txt. This simplifies the depenency management by making sure collaborators use the same versions of dependencies across different environments. Additionally, it automates virtual environments.

### Pylint and Flake8
To analyze code for errors, bugs and style inconsistencies, the linters [Pylint](https://pypi.org/project/pylint/) and [Flake8](https://flake8.pycqa.org/) are used. Pylint ensures overall code correctness and maintainability, while Flake8 has strengths in style consistency and code readability.
