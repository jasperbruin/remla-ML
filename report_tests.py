import json
import anybadge

with open('report.json', 'r') as f:
    report = json.load(f)

total_tests = report['summary']['total']
passed_tests = report['summary']['passed']
failed_tests = report['summary']['failed']

badge = anybadge.Badge('tests', f'Passed: {passed_tests}, Failed: {failed_tests}', thresholds={})
badge.write_badge('badge.svg')
