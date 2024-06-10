import pytest
import json
import io
import sys

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
        rep.capstdout = capture.buffer.getvalue()
        item._report_sections.append(('Captured stdout', 'call', rep.capstdout))

@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    json_report_file = session.config.option.json_report_file
    if not json_report_file:
        return

    with open(json_report_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    for entry in report_data.get('tests', []):
        nodeid = entry['nodeid']
        report = session.config._json_report.report.get(nodeid)
        if report:
            entry['captured_output'] = getattr(report, 'capstdout', '')

    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config._json_report = type('', (), {'report': {}})()

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    config = item.config
    if not hasattr(config._json_report, 'report'):
        config._json_report.report = {}
    config._json_report.report[item.nodeid] = rep
