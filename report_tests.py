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
test_outputs = [test['captured_output'] for test in report]

# For demonstration, we can just concatenate all outputs
badge_text = ' | '.join(test_outputs)

# Generate the badge
badge = anybadge.Badge('tests', badge_text[:30])  # Limit the length of badge text
badge.write_badge('badge.svg')