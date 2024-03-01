import csv 

### OKCupid data handling

essay_prompts = {
    'essay0': 'Self Summary',
    'essay1': 'What I\'m doing with my life',
    'essay2': 'I’m really good at',
    'essay3': 'The first thing people usually notice about me',
    'essay4': 'Favorite books, movies, show music, and food',
    'essay5': 'The six things I could never do without',
    'essay6': 'I spend a lot of time thinking about',
    'essay7': 'On a typical Friday night I am',
    'essay8': 'The most private thing I am willing to admit',
    'essay9': 'You should message me if...',
}

def load_okcupid(filename='/Users/egg/datasets/okcupid/okcupid_profiles_shuffled.csv', filter=None):
    """
    Load OKCupid profile data from a CSV file, filtering rows based on optional criteria.

    :param filename: Path to the CSV file containing profile data.
    :param filter: Optional dictionary specifying filtering criteria, where keys are column names.
    :return: A list of dictionaries, each containing profile data for a single user.
    """
    # Columns to extract from each row
    columns = [
        'age', 'status', 'sex', 'orientation', 'education', 'ethnicity',
        'income', 'job', 'location', 'essay0', 'essay1', 'essay2',
        'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9'
    ]

    profiles = []  # List to hold profile data dictionaries

    with open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
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

    return profiles
