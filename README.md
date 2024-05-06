remla-group3
==============================

This is for the Release Engineering Course

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


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


## Decision documentation
The decision process on how the project is designed is documented here.

### Cookiecutter
For the project templating, [Cookiecutter](https://github.com/cookiecutter/cookiecutter) is used. It standardizes the project setup process with predefined directory structures and configuration files. 

### Poetry
[Poetry](https://python-poetry.org/) is used as a dependency management tool instead of requirements.txt. This simplifies the depenency management by making sure collaborators use the same versions of dependencies across different environments. Additionally, it automates virtual environments.

### Pylint and Flake8
To analyze code for errors, bugs and style inconsistencies, the linters [Pylint](https://pypi.org/project/pylint/) and [Flake8](https://flake8.pycqa.org/) are used. Pylint ensures overall code correctness and maintainability, while Flake8 has strengths in style consistency and code readability.
