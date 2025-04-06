import csv
import json

def csv_to_json(csv_path, json_path):
    checklist = {}
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['Field']
            value = row['Data']

            # Auto-type conversion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # leave it as string

            checklist[key] = value

    # Write to JSON file
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(checklist, jf, indent=2)
    print(f"✅ JSON saved to: {json_path}")

# Example usage
csv_to_json('./data/company_data.csv', './data/checklist.json')
