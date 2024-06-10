"""
This module processes the test report and generates a badge.
"""

import json
import anybadge

# Load the test report
with open('report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

# Print the structure of the report
print(json.dumps(report, indent=4))

# Extract necessary information
total_tests = report['summary']['total']
passed_tests = report['summary']['passed']
failed_tests = report['summary']['failed']

# Generate the test summary
badge = anybadge.Badge(
    'tests',
    f'Passed: {passed_tests}, Failed: {failed_tests}',
    thresholds={}
    )
badge.write_badge('badge.svg')
