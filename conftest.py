import pytest
import json

# Hook to capture output
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == 'call':
        capstdout = item.config.pluginmanager.getplugin('capturemanager').read_global_capture()
        rep.capstdout = capstdout

# Hook to modify the JSON report
@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    json_report_file = session.config.option.json_report_file
    if not json_report_file:
        return
    
    with open(json_report_file, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    for entry in report_data.get('tests', []):
        nodeid = entry['nodeid']
        report = session.config._json_report.report[nodeid]
        entry['captured_output'] = report.capstdout

    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)
