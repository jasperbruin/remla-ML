# pylint: skip-file

"""
This module processes the test report and generates a badge.
"""

import json
import anybadge
import os

# Path to the report file
report_file = 'report.json'

if not os.path.exists(report_file):
    print(f"Report file '{report_file}' does not exist.")
    exit(1)

# Load the test report
try:
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from report file: {e}")
    exit(1)

# Print the structure of the report
print(json.dumps(report, indent=4))

# Ensure the report has the expected structure
if 'summary' not in report or 'passed' not in report['summary']:
    print("The report does not contain 'summary' or 'passed' key.")
    exit(1)

# Extract the number of tests passed
num_tests_passed = report['summary']['passed']

# Generate the badge
badge = anybadge.Badge('tests passed', str(num_tests_passed))

# Write the badge to a file
badge_output_file = 'badge.svg'
try:
    badge.write_badge(badge_output_file)
    print(f"Badge written to '{badge_output_file}'")
except Exception as e:
    print(f"Error writing badge to file: {e}")
    exit(1)
