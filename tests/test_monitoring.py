# pylint: disable=all
# flake8: noqa

import os
from datetime import datetime, timezone

MAX_MODEL_AGE = 50  # Considered stale after 30 days


def get_file_age(file_path):
    if os.path.exists(file_path):
        modification_time = os.path.getmtime(file_path)
        modification_time_readable = datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')

        # Get the current time
        current_time = datetime.now(timezone.utc).timestamp()

        # Calculate the age of the file
        age_seconds = current_time - modification_time
        age_days = age_seconds / (60 * 60 * 24)

        print(f"File Last Modified: {modification_time_readable}")
        print(f"File Age: {age_days:.2f} days")
        return age_days
    else:
        print("File not found.")
        return None


def test_check_for_staleness():
    """
    Test model staleness.
    """
    model_file_path = 'data/external/model.keras'

    model_age = get_file_age(model_file_path)
    assert model_age <= MAX_MODEL_AGE
