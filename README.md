remla-group3
==============================

This is for the Release Engineering Course

## Test Results

![Workflow Update](https://github.com/remla24-team3/model-training/actions/workflows/pytest.yml/badge.svg)

### Accessing ML Tests Artifacts
1. Go to [GitHub Actions](https://github.com/remla24-team3/model-training/actions) page of our repository.
2. Click on the last ML Model Tests workflow run.
3. Scroll down to the **Artifacts** section at the bottom of the workflow run page to find and download the following:
   - **coverage-report**: Testing adequacy - HTML coverage report.
   - **ml-metrics**: ML metrics and other relevant outputs from the tests are reported in this log.
   - **test-badge**: Test result badge for the latest run. (also displayed in the README)

##  Repository Structure

```sh
└── model-training/
    ├── .github
    │   └── workflows
    ├── LICENSE
    ├── README.md
    ├── dvc.lock
    ├── dvc.yaml
    ├── notebook
    │   └── phishing-detection-cnn.ipynb
    ├── params.yaml
    ├── poetry.lock
    ├── pyproject.toml
    ├── pytest.ini
    ├── report_tests.py
    ├── src
    │   ├── data
    │   ├── models
    │   └── visualization
    ├── tests
    │   ├── test_features_data.py
    │   ├── test_model_development.py
    │   ├── test_model_infrastructure.py
    │   └── test_monitoring.py
    └── tox.ini
```
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
To run the tests you can run the following command:

```bash
 poetry run pytest tests/
```

This will run **features_data**, **model_development**, **model_infrastructure**, **monitoring** tests, including various tests on data slices, mutamorphic testing and robustness tests.

### Keeping Everything Synchronized

To ensure consistency across all team members' environments:

- Run `dvc pull` to update your local files with the latest versions from the remote storage when you start working or after cloning the repository.
- After significant changes, always use `dvc push` to sync your updates back to the remote storage.

Following these practices will help maintain a smooth and efficient workflow for everyone involved in the project.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Tools Used 

### Cookiecutter
For the project templating, [Cookiecutter](https://github.com/cookiecutter/cookiecutter) is used. It standardizes the project setup process with predefined directory structures and configuration files. 

### Poetry
[Poetry](https://python-poetry.org/) is used as a dependency management tool instead of requirements.txt. This simplifies the depenency management by making sure collaborators use the same versions of dependencies across different environments. Additionally, it automates virtual environments.

### Pylint and Flake8
To analyze code for errors, bugs and style inconsistencies, the linters [Pylint](https://pypi.org/project/pylint/) and [Flake8](https://flake8.pycqa.org/) are used. Pylint ensures overall code correctness and maintainability, while Flake8 has strengths in style consistency and code readability.
