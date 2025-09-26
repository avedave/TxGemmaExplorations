import json
import re
import csv

# Input and output file paths
json_file_path = 'tdc_prompts_formatted.json'
csv_file_path = 'tdc_prompts.csv'

# Read the JSON data
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare data for CSV
csv_data = [['Task Name', 'Task Prompt', 'Parameters']]

# Regex to find all {parameter} occurrences
param_regex = re.compile(r'\{([^}]+)\}')

for task_name, task_prompt in data.items():
    parameters = param_regex.findall(task_prompt)
    params_str = ', '.join(parameters)
    csv_data.append([task_name, task_prompt, params_str])

# Write to the CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"Successfully converted {json_file_path} to {csv_file_path}")
