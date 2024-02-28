import okcupid
import openai_uk

### Topic/token handling

# {'age': '22',
#  'status': 'single',
#  'sex': 'm',
#  'orientation': 'straight',
#  'education': 'working on college/university',
#  'ethnicity': 'asian, white',
#  'income': '-1',
#  'job': 'transportation',
#  'location': 'south san francisco,...california',
#  'essay0': 'about me:  i would l...tion span.',
#  'essay1': 'currently working as...zy sunday.',
#  'essay2': 'making people laugh....implicity.',
#  'essay3': 'the way i look. i am... blend in.',
#  'essay4': 'books: absurdistan,
# }

# Token conversion setup
tokens = {
    'politics': {'addendum': 'Politically, this user is',
                 'tokenIds': {' liberal': 18250, ' conservative': 15692},
                 'bias': {' liberal': 80.35, ' conservative': 80.65},
                 'okc_name': None,
                 'okc_vals': None},
    'gender': {'addendum': 'The gender of this user is',
               'tokenIds': {' male': 8762, ' female': 8954},
               'bias': {' male': 80, ' female': 80},
               'okc_name': 'sex', 
               'okc_vals': {' male': 'm', ' female': 'f'}},
    'sexuality': {'addendum': 'Is this user straight, bisexual, or gay? They are',
                  'tokenIds': {' straight': 7833, ' bisexual': 56832, ' gay': 8485},
                  'bias': {' straight': 80, ' bisexual': 80, ' gay': 80},
                  'okc_name': 'orientation', 
                  'okc_vals': {' straight': 'straight', ' bisexual': 'bisexual', ' gay': 'gay'}},
    'education': {'addendum': 'Is this user college-educated? Answer Yes or No:',
                  'tokenIds': {' Yes': 7566, ' No': 2360},
                  'bias': {' Yes': 80, ' No': 80},
                  'okc_name': 'education', 
                  'okc_vals': None}, # TODO
    'ethnicity': {'addendum': 'What is this user\'s primary race or ethnicity? Is it Black, White, Asian, or Hispanic?',
                  'tokenIds': {' Black': 5348, ' White': 5929, ' Asian': 14875, ' Hispanic': 41985},
                  'bias': {' Black': 79, ' White': 82, ' Asian': 79, ' Hispanic': 79},
                  'okc_name': 'ethnicity', 
                  'okc_vals': {' Black': 'black', ' White': 'white', ' Asian': 'asian', ' Hispanic': 'hispanic'}}
}

subjects = ['politics', 'gender', 'sexuality', 'education', 'ethnicity']

def prune_profiles():
    """Remove profiles that ChatGPT says have explicit giveaways for demographics."""
    pass

# Function to check whether a value in a profile matches a token result, eg whether 'gender' of ' male' / ' female' in the token results matches 'sex' of 'm' / 'f' in the profile
def check_token_match(profile, token_result, subject):
    """Check whether a value in a profile matches a result from OpenAI."""
    okc_name = tokens[subject]['okc_name']
    okc_vals = tokens[subject]['okc_vals']
    
    if okc_name is None or okc_vals is None:
        return None
    
    profile_value = profile.get(okc_name)
    
    if profile_value is None or token_result is None:
        return None
    
    chosen_token = max(token_result, key=token_result.get)
    expected_profile_value = okc_vals[chosen_token]
    return profile_value == expected_profile_value


# TODO am I using this?
def split_to_match_and_estimate(matches):
    """Given a list for a particular topic, split it into one for matches and one for estimates."""
    # Example 'matches' list:
    # [{'match?': True, 'estimate': {...}}, {'match?': True, 'estimate': {...}}, ...]

    # Initialize two dictionaries to hold the separated data
    match_map = {}
    estimate_map = {}
    # Iterate through each dictionary in the input list and populate the new maps
    for index, item in enumerate(matches):
        match_map[index] = item['match?']
        estimate_map[index] = item['estimate']
    return match_map, estimate_map

def calculate_correctness_statistics(matches_by_topic):
    """Given a map of lists of matches and estimates, calculate the proportion for each topic where 'match?' == True."""
    correctness_statistics = {}
    
    for topic, items in matches_by_topic.items():
        # Filter out items where 'match?' is not None
        filtered_items = [item for item in items if item['match?'] is not None]
        
        if not filtered_items:
            # If all 'match?' values are None, set statistic to None
            correctness_statistics[topic] = None
        else:
            # Calculate the proportion of 'True' values
            true_count = sum(item['match?'] for item in filtered_items)
            proportion_true = true_count / len(filtered_items)
            correctness_statistics[topic] = proportion_true
    
    return correctness_statistics

def process_profile(profile):
    """process a single OKCupid profile, sending the essay questions to OpenAI and comparing the results to 
    the user's ground truth demographics"""
    # Example profile:
    # {'age': '22', 'status': 'single', 'sex': 'm', 'orientation': 'straight', 'education': 'working on college/university', 'ethnicity': 'asian, white', 'income': '-1', 'job': 'transportation', 'location': 'south san francisco,...california', 'essay0': 'about me:  i would l...tion span.', 'essay1': 'currently working as...zy sunday.', 'essay2': 'making people laugh....implicity.', 'essay3': 'the way i look. i am... blend in.', 'essay4': 'books: absurdistan, ... anything.', ...}

    # Prepare the user's essay responses for input to OpenAI
    context_input = ""
    for essay in okcupid.essay_prompts:
        if essay in profile:
            # context_input += essay_prompts[essay] + "\n" + profile[essay] + "\n\n"
            context_input += profile[essay] + "\n"
    # Some profiles have no (or very little) essay text
    if len(context_input) < 400:
        print(f"Profile has too little text: {profile}")
        return None
    # Call OpenAI to get demographic estimates
    try:
        user_estimates = openai_uk.call_openai(subjects, tokens, context_input)
    # We'll just skip any profiles that cause problems
    # except Exception as e:
    except BlockingIOError as e:
        print(f"Bad profile: {profile}")
        print(f"Error processing profile: {e}")
        return None

    # print()
    # print(f"Profile: {profile}")
    # print(f"Estimates: {user_estimates}")
    # print()

    # Compare the user's ground truth demographics to the estimates
    matches = {}
    if user_estimates is None:
        return None
    for subject in subjects:
        match = check_token_match(profile, user_estimates[subject], subject)
        matches[subject] = {'match?': match,
                            'estimate': user_estimates[subject],}
        # print("Was " + subject + " a match? " + str(match))
        # print(f"Ground truth: {profile[subject]}")
        # print(f"Estimate: {user_estimates[subject]}")
        # print()
    return matches

# main function which loads the OKCupid data and processes each profile
def main():
    profiles = okcupid.load_okcupid()
    matches = []
    # Cost as of 2/25: ~$5 per 1000 profiles
    for profile in profiles[:5]:
        match = process_profile(profile)
        if match is not None:
            matches.append(match)
    # if match is not None:
    matches_by_topic = {key: [d[key] for d in matches] for key in matches[0]}
    # Calculate and print the correctness statistics
    correctness_statistics = calculate_correctness_statistics(matches_by_topic)
    print(correctness_statistics)
    return matches

main_matches = main()

# TODO 
# - deal with education, age
# - prune data in various ways
# - calculate brier score along with basic correctness
# - experiment with open ended description of user
pass