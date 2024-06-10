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
test_outputs = [test.get('captured_output', '') for test in report['tests']]

# For demonstration, we can just concatenate all outputs
badge_text = ' | '.join(test_outputs)[:30]  # Limit the length of badge text

# Generate the badge
badge = anybadge.Badge('tests', badge_text)
badge.write_badge('badge.svg')