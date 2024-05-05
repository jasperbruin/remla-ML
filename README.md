remla-group3
==============================

This is for the Release Engineering Course

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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
