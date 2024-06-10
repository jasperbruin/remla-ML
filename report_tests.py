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
if 'tests' not in report:
    print("The report does not contain 'tests' key.")
    exit(1)

# Extract necessary information
test_outputs = [test.get('captured_output', '') for test in report['tests']]

# For demonstration, we can just concatenate all outputs
badge_text = ' | '.join(test_outputs)[:30]  # Limit the length of badge text

# Ensure the badge text is not empty
if not badge_text:
    badge_text = 'No output'

# Generate the badge
badge = anybadge.Badge('tests', badge_text)

# Write the badge to a file
badge_output_file = 'badge.svg'
try:
    badge.write_badge(badge_output_file)
    print(f"Badge written to '{badge_output_file}'")
except Exception as e:
    print(f"Error writing badge to file: {e}")
    exit(1)
