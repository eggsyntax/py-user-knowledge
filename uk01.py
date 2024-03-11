import json
import os

import okcupid
import openai_uk
import persuade
import roc
import utils

import plotly.express as px
import plotly.graph_objects as go

### Topic/token handling

RESULTS_DIR = utils.results_dir()
# RESULTS_DIR = 'results/2024-03-03_250'
RESPONSES_FILE = f'{RESULTS_DIR}/openai_responses.json'

# TODO 
# - change addendum to a question, see gender
# - change okc_vals
# - check for token ids, bias -- note that the tokens shouldn't be space-prefaced.
#   - TODO if I want to be cool I should just combine eg ' male' 'male' ' Male' 'Male'.
# 

# Token conversion setup
tokens = {
    'politics': {'addendum': 'Politically, this user is',
                 'tokenIds': {' liberal': 18250, ' conservative': 15692},
                 'bias': {' liberal': 80.35, ' conservative': 80.65},
                 'okc_name': None,
                 'okc_vals': None},
    'gender': {'addendum': 'Is the author of the preceding text male or female?',
               'tokenIds': {' male': 8762, ' female': 8954},
               'bias': {' male': 80, ' female': 80},
               'okc_name': 'sex', 
               'okc_vals': {'male': 'm', 'female': 'f'}},
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

# subjects = ['politics', 'gender', 'sexuality', 'education', 'ethnicity']
subjects = ['gender']

### Helper functions for matching

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
    
    chosen_token = max(token_result, key=lambda k: int((token_result.get(k)[:-1]))) # strip trailing '%' & intify
    try:
        expected_profile_value = okc_vals[chosen_token]
        return profile_value.lower() == expected_profile_value.lower()
    except KeyError as e:
        print(f"Ended up with a weird token result: {e}")
        return None


# TODO am I using this?
def split_to_match_and_estimate(matches):
    """Given a list for a particular topic, split it into one for matches and one for estimates."""
    # Example 'matches' list:
    # [{'match?': True, 'estimate': {...}}, {'match?': True, 'estimate': {...}}, ...]
    #
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

### Graphing 

def summarize_matches(matches, category):
    # Extracting data and converting it to the desired format for plotting
    # Sloppy-ass GPT code
    reduced_data = []
    # TODO -- this whole first_category thing isn't gonna work properly in cases with > 2 options (like ethnicity)
    # need to rethink.
    # first_category = next(iter(matches[0][category]['estimate']))
    # print("<summarize_matches> base category: " + first_category + "; num matches: " + str(len(matches)))
    for item in matches:
        # instead I need max -- do it the same way I do it here on chosen token:
        # chosen_token = max(token_result, key=lambda k: int((token_result.get(k)[:-1]))) # strip trailing '%' & intify
        # first_category_percentage = int(item[category]['estimate'][first_category].replace('%', ''))
        items = item[category]['estimate'].items()
        max_item = max(items, key=lambda item: int(item[1].replace('%', '')))
        top_category, top_category_percentage = max_item
        print(max_item)
        # reduced_data.append({first_category: first_category_percentage, 'Match?': item[category]['match?']})
        reduced_data.append({'max_percent': top_category_percentage, 'Match?': item[category]['match?']})
    print('Overall match percentage on ' + top_category + ': ' + str( len([x for x in reduced_data if x['Match?']]) / len(reduced_data) ))
    # Initialize an empty dictionary to count occurrences and matches in buckets
    percentage_counts = {}
    # Populate the dictionary with data, creating buckets as needed
    for entry in reduced_data:
        percentage = int(entry['max_percent'].replace('%', ''))
        bucket_key = f'{(percentage // 5) * 5:03d}-{((percentage // 5) * 5 + 4):03d}'
        # Adjust the bucket_key for the range 95-100 so it's inclusive.
        if percentage > 94:  # This covers 95 to 100
            bucket_key = '095-100'
        if bucket_key not in percentage_counts:
            percentage_counts[bucket_key] = {'total': 0, 'matches': 0}
        percentage_counts[bucket_key]['total'] += 1
        if entry['Match?']:
            percentage_counts[bucket_key]['matches'] += 1
    print(f"<summarize_matches> reduced_data: {len(reduced_data)}, percentage_counts: {len(percentage_counts)}")
    # Transform the dictionary into the desired list of maps format
    with open(f'{RESULTS_DIR}/reduced_data.json', 'w') as f:
        f.write(json.dumps(reduced_data))
    plot_data = [{'x': key, 'total': value['total'], 'percent correct': (value['matches'] / value['total']) * 100 if value['total'] > 0 else 0} for key, value in percentage_counts.items()]
    plot_data = sorted(plot_data, key=lambda m: m['x'])
    with open(f'{RESULTS_DIR}/plot_data.json', 'w') as f:
        f.write(json.dumps(plot_data))
    return plot_data

def graph_matches(matches, correctness_statistics):
    """Starts with the structure of matches, which is a list of maps of this form:
    `{'politics': {'match?': None, 'estimate': {' liberal': '100%', ' conservative': '0%'}}, 'gender': {'match?': True, 'estimate': {' female': '93%', ' male': '7%'}}, 'sexuality': {'match?': True, 'estimate': {' straight': '95%', ' gay': '3%', ' bisexual': '3%'}}, 'education': {'match?': None, 'estimate': {' Yes': '58%', ' No': '42%'}}, 'ethnicity': {'match?': True, 'estimate': {' White': '94%', ' Hispanic': '3%', ' Asian': '2%', ' Black': '0%'}}}`
    and summarizes them into a relation between the percent confidence of the estimate and the match status"""
    for subject in subjects:
        summary_data = summarize_matches(matches, subject)
        # fig = px.bar(summary_data, x='x', y='percent correct', hover_data='total', title='Percentage Confidence vs. Match Status')

        # Hmm, let's do a different kind of bar chart. NB if I'm happy with this I should just prep the data 
        # this way to begin with in summarize_matches.

        # Preparing data for the stacked bar chart
        correct_data = []
        incorrect_data = []
        for item in summary_data:
            correct_count = round((item['percent correct'] / 100) * item['total'])
            incorrect_count = item['total'] - correct_count
            correct_data.append(correct_count)
            incorrect_data.append(incorrect_count)

        # Create the figure
        fig = go.Figure(data=[
            # go.Bar(name='Correct', x=[item['x'] for item in summary_data], y=correct_data, marker_color='#134e6d'),
            # go.Bar(name='Incorrect', x=[item['x'] for item in summary_data], y=incorrect_data, marker_color='#7f2322')
            go.Bar(name='Correct', x=[item['x'] for item in summary_data], y=correct_data, marker_color='#5A67A4'),
            go.Bar(name='Incorrect', x=[item['x'] for item in summary_data], y=incorrect_data, marker_color='#A45A67')
        ])

        # Update the layout
        # fig.update_layout(barmode='stack', title='Total vs. Correct/Incorrect Distribution',
        #                 xaxis_title='Buckets', yaxis_title='Total Count', hovermode='x')

        # # Add the line chart for 'percent correct'
        # fig.add_trace(go.Scatter(x=[item['x'] for item in summary_data], y=[item['percent correct'] for item in summary_data], 
        #                          mode='lines+markers', name='Percent Correct', line=dict(color='green')))

        # # Update the layout
        # fig.update_layout(barmode='stack', title='Total vs. Correct/Incorrect Distribution and Percent Correct',
        #                 xaxis_title='Buckets', yaxis_title='Total Count', hovermode='x')
        
        # Update layout to add a secondary y-axis
        fig.update_layout(
            barmode='stack',
            # TODO ! Add overall category accuracy to the title -- should be available already somewhere
            title=f'Total vs. Correct/Incorrect Distribution and Percent Correct: {subject}',
            xaxis_title='Buckets',
            yaxis=dict(title='Total Count', side='left', showgrid=False),
            yaxis2=dict(title='Percent Correct', side='right', overlaying='y', showgrid=False),
            hovermode='x'
            )

        # Add the line chart for 'percent correct' with the secondary y-axis
        fig.add_trace(go.Scatter(x=[item['x'] for item in summary_data], y=[item['percent correct'] for item in summary_data],
                                mode='lines+markers', name='Percent Correct', yaxis='y2', line=dict(color='#a59a52')))

        fig.show()
    return summary_data # NB summary data is just the one from the last category
    
### Main profile processing

def process_profile(profile):
    """process a single profile, sending the essay questions to OpenAI and 
    comparing the results to the user's ground truth demographics"""
    # Example profile (from okcupid):
    # {'age': '22', 'status': 'single', 'sex': 'm', 'orientation': 'straight', 'education': 'working on college/university', 'ethnicity': 'asian, white', 'income': '-1', 'job': 'transportation', 'location': 'south san francisco,...california', 'essay0': 'about me:  i would l...tion span.', 'essay1': 'currently working as...zy sunday.', 'essay2': 'making people laugh....implicity.', 'essay3': 'the way i look. i am... blend in.', 'essay4': 'books: absurdistan, ... anything.', ...}

    # Call OpenAI to get demographic estimates
    try:
        # TODO maybe don't call openai if there's no ground truth! That'll save substantial money.

        # TODO Maddeningly, the logit_bias on OpenAI's (legacy) completions endpoint seems to 
        # have stopped working. I spent a ton of time debugging and eventually checked out a
        # version of the code from when I know it was working, and as far as I can tell it
        # honors a single logit_bias and ignores the rest. So...good thing I have some good 
        # results already I guess? 
        #
        #### example of the problem:
        #
        # I send with logit_bias = {18250: 80.35, 15692: 80.65} (18250 = ' liberal', 15692 = ' conservative')
        #
        # I get back:
        # {'text': ' liberal', 'index': 0, 'logprobs': {'tokens': [' liberal'], 
        #  'token_logprobs': [-3.5714898], 'top_logprobs': [{' a': -1.7806993,
        #  ' likely': -2.09508}], 'text_offset': [780]}, 'finish_reason': 'length'}
        #
        # Which is to say: it *seems* to be applying one logit_bias (for liberal) and ignoring the rest.
        # This is not particular to the `politics subject.`
        #
        ### end example
        #
        # TODO and then when I get that figured out, I was just adding roc.py stuff to calculate
        # ROC curve and AUC and such. The GPT-4 code is in roc.py, but I don't think it's been tested.y
        #
        # If I have to be away from this for long enough that I forget what's going on, it's fine 
        # to stash/discard the outstanding changes and go with the last committed version.

        user_estimates = openai_uk.call_openai(subjects, tokens, profile['essay'])
        with open(RESPONSES_FILE, 'a') as f: 
            f.write(json.dumps(profile) + '\n')
            f.write(json.dumps(user_estimates) + '\n\n')
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
    # TODO 'matches' is used elsewhere to mean 'successfull matches' -- rename this
    matches = {}
    if user_estimates is None:
        return None
    print(f"User estimates: {user_estimates}") # XXX
    for subject in subjects:
        match = check_token_match(profile, user_estimates[subject], subject)
        matches[subject] = {'match?': match,
                            'estimate': user_estimates[subject],}
        # print("Was " + subject + " a match? " + str(match))
        # print(f"Ground truth: {profile[subject]}")
        # print(f"Estimate: {user_estimates[subject]}")
        # print()
    # print("matches for profile " + json.dumps(profile))
    # print(matches)
    return matches

def process_profiles(profiles):
    print('Getting estimates from OpenAI...')
    matches = []
    # Cost as of 2/25: ~$5 per 1000 profiles
    for profile in profiles[:NUM_PROFILES]:
        match = process_profile(profile)
        if match is not None:
            matches.append(match)
            if len(matches) % 20 == 0:
                print(f"Processed {len(matches)} profiles")
    if matches:
        matches_by_topic = {key: [d[key] for d in matches] for key in matches[0]}
    else:
        matches_by_topic = {}
    with open(f'{RESULTS_DIR}/matches.json', 'w') as f:
        f.write(json.dumps(matches))
    with open(f'{RESULTS_DIR}/matches_by_topic.json', 'w') as f:
        f.write(json.dumps(matches_by_topic))
    # Calculate and print the correctness statistics
    correctness_statistics = calculate_correctness_statistics(matches_by_topic)
    print(correctness_statistics)
    return matches, correctness_statistics

def main(ask_openai=False, dataset_module=okcupid):
    profiles = dataset_module.load_data(NUM_PROFILES=NUM_PROFILES)
    # Clear previous responses if they exist
    try:
        os.remove(RESPONSES_FILE)
    except FileNotFoundError:
        pass

    if ask_openai:
        main_matches, correctness_statistics = process_profiles(profiles)
    else:
        with open(f'{RESULTS_DIR}/matches.json', 'r') as f:
            main_matches = json.loads(f.read())
    # calculate_roc = roc.generate_roc_and_analyze_skewness(main_matches, "gender")
    summary_data = graph_matches(main_matches, correctness_statistics)
    return summary_data # NB summary data is just the one from the last category, not actually meaningful overall

NUM_PROFILES = 250
main(ask_openai=True, dataset_module=okcupid) # persuade, okcupid

# TODO 
# - push some sharegpt data through
# - measure accuracy as a function of essays_len
# - try having gpt-4 create essays as a persona with specific demographics
# - deal with education, age
# - prune data in various ways (gender: done)
# - calculate brier score along with basic correctness
# - maybe experiment with open ended description of user
pass