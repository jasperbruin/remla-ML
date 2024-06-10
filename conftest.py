import json
import io
import sys

import pytest


class CaptureStdout:
    def __init__(self):
        self._stdout = sys.stdout
        self.buffer = io.StringIO()

    def __enter__(self):
        sys.stdout = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    with CaptureStdout() as capture:
        outcome = yield
    rep = outcome.get_result()
    if rep.when == 'call':
        captured_output = capture.buffer.getvalue()
        item._report_sections.append(
            ('Captured stdout', 'call', captured_output)
            )
        # Store captured output in a custom attribute on the item object
        setattr(item, '_captured_output', captured_output)
    # Ensure custom _json_report attribute is a dictionary
    if not hasattr(item.config, '_custom_json_report'):
        item.config._custom_json_report = {}
    item.config._custom_json_report[item.nodeid] = rep


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    JSON_REPORT_FILE = session.config.option.json_report_file
    if not JSON_REPORT_FILE:
        return

    with open(JSON_REPORT_FILE, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    for entry in report_data.get('tests', []):
        nodeid = entry['nodeid']
        report = session.config._custom_json_report.get(nodeid)
        if report:
            # Retrieve the captured output from the corresponding item
            captured_output = ""
            for item in session.items:
                if item.nodeid == nodeid:
                    captured_output = getattr(item, '_captured_output', '')
                    break
            entry['captured_output'] = captured_output

    with open(JSON_REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not hasattr(config, '_custom_json_report'):
        config._custom_json_report = {}


def main():
    try:
        # some code that might fail
        pass
    except Exception as e:  # consider using a more specific exception
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
