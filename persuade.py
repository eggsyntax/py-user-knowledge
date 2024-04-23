import csv
import json

import utils

"""Data handling for the persuade 2.0 dataset"""

RESULTS_DIR = utils.results_dir()

def load_data(categories, filename='/Users/egg/datasets/persuade-2/persuade_2.0_shuffled_12thgrade_remapped_to_match_okcupid.csv', filter=None, NUM_PROFILES=20):
    """
    Load persuade 2.0 profile data from a CSV file, filtering rows based on optional criteria.

    :param filename: Path to the CSV file containing profile data.
    :param filter: Optional dictionary specifying filtering criteria, where keys are column names.
    :return: A list of dictionaries, each containing profile data for a single user.
    """
    # Columns to extract from each row
    columns = ["essay", "sex", "grade_level", "ethnicity"]

    profiles = []  # List to hold profile data dictionaries
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(profiles) >= NUM_PROFILES:
                break   
            # Check if the row matches the filter criteria, if any
            if filter:
                match = True
                for key, value in filter.items():
                    if row.get(key) != value:
                        match = False
                        break
                if not match:
                    continue  # Skip this row as it doesn't match the filter
            # Extract the desired data from the row
            profile = {col: row[col] for col in columns if col in row}
            profiles.append(profile)
    with open(f'{RESULTS_DIR}/valid_profiles.json', 'w') as f:
        f.write(json.dumps(profiles))
    return profiles

