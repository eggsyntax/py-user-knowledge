import csv 
import json
import re

import okcupid
import utils

### OKCupid data handling

RESULTS_DIR = utils.results_dir()

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

gender_synonyms = set([
    r'\bhe\b', r'\bhim\b', r'\bhis\b', r'\bhimself\b', r'\bshe\b', r'\bher\b', 
    r'\bhers\b', r'\bherself\b', r'\bman\b', r'\bwoman\b', r'\bmale\b', 
    r'\bfemale\b', r'\bgirl\b', r'\bboy\b', r'\blady\b', r'\bdude\b'
    ])

def load_data(categories, tokens, filename='/Users/egg/datasets/okcupid/okcupid_profiles_shuffled_edu_age.csv', filter=None, NUM_PROFILES=20, offset=0):
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

    # Prune

    valid_profiles, invalid_profiles = prune_profiles(profiles, categories, tokens, NUM_PROFILES, offset)
    with open(f'{RESULTS_DIR}/valid_profiles.json', 'w') as f:
        f.write(json.dumps(valid_profiles))
    with open(f'{RESULTS_DIR}/invalid_profiles.json', 'w') as f:
        f.write(json.dumps(invalid_profiles))
    return valid_profiles

def profile_essays(profile):
    """Return the essay text from a profile"""
    essay_input = ""
    for essay in essay_prompts:
        if essay in profile:
            essay_input += profile[essay] + "\n"
    return essay_input

def prune_profiles(profiles, categories, tokens, NUM_PROFILES=20, offset=0):
    """Remove profiles that include any of various synonyms for gender, or that just 
    have too little essay text to use."""
    print(f'Pruning profiles, looking for {NUM_PROFILES} valid ones...')
    validish_profiles = []
    invalidish_profiles = []
    for profile in profiles[offset:]:
        # if the profile contains any of the words in gender_synonyms, print it and the relevant words:
        # NOTE that this could be better -- we only want to check the essay* fields, not the whole profile
        is_invalid = False
        for key, value in profile.items():
            if any(re.search(word, str(value)) for word in gender_synonyms):
                is_invalid = True
                continue
        # Some profiles have no (or very little) essay text
        combined_essays = profile_essays(profile)
        essays_len = len(combined_essays)
        # TODO measure accuracy as a function of essays_len
        profile['essay'] = combined_essays
        profile['essays_len'] = essays_len
        if essays_len < 400:
            # print(f"Profile has too little text: {profile}")
            # print(f"Profile has too little text")
            is_invalid = True
        # Some profiles (notably ethnicity) have a wide variety of values, some of which are 
        # not included in our guesses, so we skip those. eg it'd be silly to have a category
        # for 'hispanic / latin, white, other' even though that shows up in a profile
        for category in categories:
            known_values = tokens[category].get('okc_vals').values()
            okc_name = tokens[category].get('okc_name')
            profile_value = profile.get(okc_name)
            if profile_value is None:
                is_invalid = True
                continue
            if profile_value not in known_values:
                is_invalid = True
                continue
        if is_invalid:
            invalidish_profiles.append(profile)
        else:
            validish_profiles.append(profile)
        if len(validish_profiles) >= NUM_PROFILES:
            break

    print("Pruning complete.")
    print(f"  Validish profiles: {len(validish_profiles)}")
    print(f"  Invalidish profiles: {len(invalidish_profiles)}")
    return validish_profiles, invalidish_profiles
