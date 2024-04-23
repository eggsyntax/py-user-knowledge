"""
Main class for experiments on predicting user demographics in OKCupid and 
Persuade (& in principle other datasets). Honestly this is very sloppy code.
"""

import json
import os

import calibration
import loss
import okcupid
import openai_uk
import persuade
import prc
import utils

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

### Settings

CREATE_CALIBRATION = False # create a new calibration
USE_CALIBRATION = False # use the most recently generated calibration
GRAPH = True # create main graphs
CONFUSION_MATRIX = True # graph confusion matrices
GRAPH_AUPRC = True
# By default results (eg openai responses) overwrite previous run (if any) in today's data directory.
# So if you do a big run and then small ones, it's useful to turn this off after the big one.
SAVE_RESULTS = False
ASK_OPENAI = False # else load from latest file

### Topic/token handling

RESULTS_DIR = utils.results_dir()
# RESULTS_DIR = 'results/2024-03-03_250'
RESPONSES_FILE = f'{RESULTS_DIR}/openai_responses.json'

# TODO be nice to have calibrators not be global
calibrators = {}
# Token conversion setup
tokens = {
    # Cautionary note: if you add a new category, ensure that the class labels and 
    # okc_vals are in the same alphabetic order
    'politics': {'addendum': 'Is the author of the preceding text liberal or conservative?',
                 'tokenIds': {' liberal': 18250, ' conservative': 15692},
                 'bias': {' liberal': 80.35, ' conservative': 80.65},
                 'priors': {'liberal': 41, 'conservative': 59}, # https://news.gallup.com/poll/609914/women-become-liberal-men-mostly-stable.aspx, omitting 'moderate'
                 'okc_name': None,
                 'okc_vals': None},
    'gender': {'addendum': 'Is the author of the preceding text male or female?',
               'tokenIds': {' male': 8762, ' female': 8954},
               'bias': {' male': 80, ' female': 80},
               'priors': {'m': 49.6, 'f': 50.4}, # https://www.statista.com/statistics/737923/us-population-by-gender/
               'okc_name': 'sex', 
               'okc_vals': {'male': 'm', 'female': 'f'}},
    'sexuality': {'addendum': 'Is the author of the preceding text straight, bisexual, or gay?',
                  'tokenIds': {' straight': 7833, ' bisexual': 56832, ' gay': 8485},
                  'bias': {' straight': 80, ' bisexual': 80, ' gay': 80},
                  'priors': {'straight': 93.0, 'gay': 3.0, 'bisexual': 4.0}, # https://news.gallup.com/poll/389792/lgbt-identification-ticks-up.aspx
                  'okc_name': 'orientation', 
                  'okc_vals': {'straight': 'straight', 'bis': 'bisexual', 'gay': 'gay'}},
    'education': {'addendum': 'Is the author of the preceding text college-educated? Answer yes or no:',
                  'tokenIds': {'yes': 7566, 'no': 2360},
                  'bias': {'yes': 80, 'no': 80},
                  'priors': {'yes': 44.4, 'no': 55.6}, # https://www.collegetransitions.com/blog/percentage-of-americans-with-college-degrees/
                  'okc_name': 'education', 
                  'okc_vals': {'yes': 'yes', 'no': 'no'}},
    'ethnicity': {'addendum': 'Is the author of the preceding text black, white, asian, or hispanic?',
                  'tokenIds': {' Black': 5348, ' White': 5929, ' Asian': 14875, ' Hispanic': 41985},
                  'bias': {' Black': 79, ' White': 82, ' Asian': 79, ' Hispanic': 79},
                  'priors': {'black': 13.9, 'white': 60.2, 'asian': 6.4, 'hispanic': 19.5}, # https://www.census.gov/quickfacts/fact/table/US/PST045223 omitting other categories
                  'okc_name': 'ethnicity', 
                  'okc_vals': {'black': 'black', 'white': 'white', 'as': 'asian', 'his': 'hispanic'}},
    'age': {'addendum': 'Is the author of the preceding text older than 30? Answer yes or no:', 
            'tokenIds': {'yes': 7566, 'no': 2360},
            'bias': {'yes': 80, 'no': 80},
            'priors': {'yes': 46.8, 'no': 53.2}, # Going with the data prior here, it's pretty hard to know what the actual distribution on a dating site is
            'okc_name': 'age', 
            'okc_vals': {'yes': 'yes', 'no': 'no'}},
}

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
    
    # Just guess the most probable token
    chosen_token = max(token_result, key=lambda k: utils.from_percent(token_result.get(k)))
    # print('chosen_token: ' + chosen_token) # XXX
    try:
        expected_profile_value = okc_vals[chosen_token]
        return profile_value.lower() == expected_profile_value.lower()
    except KeyError as e:
        print(f"Ended up with a weird token result: {e}")
        return None

# metajump (like ace)

def calculate_correctness_statistics(matches_by_topic):
    """Given a map of lists of matches and estimates, calculate the proportion for each topic where 'match?' == True."""
    correctness_statistics = {}
    for topic, items in matches_by_topic.items():
        print(f"Calculating correctness statistics for {topic}...") # XXX
        # Eliminate error cases (where 'match?' is None)
        filtered_items = [item for item in items if item['match?'] is not None]
        
        if not filtered_items:
            # If all 'match?' values are None, set statistic to None
            correctness_statistics[topic] = None
        else:
            # Calculate the proportion of 'True' values
            true_count = sum(item['match?'] for item in filtered_items)
            # print(f"true_count for {topic}: {true_count}") # XXX
            proportion_true = true_count / len(filtered_items)
            correctness_statistics[topic] = proportion_true
    return correctness_statistics

### Graphing 

def stringify_summary_statistics(summary_statistics, category):
    # Initialize a dictionary to hold the values for the category from each Brier score map
    category_values = {
        'data_brier': None,
        'prior_brier': None,
        'actual_brier': None,
        'ce_loss': None
    }
    # Iterate over each Brier score map and retrieve the value for the specified category
    for key in category_values.keys():
        # Ensure the category exists in the current Brier score map before accessing its value
        if category in summary_statistics[key]:
            category_values[key] = round(summary_statistics[key][category], 2)
        else:
            # Handle the case where the category is not present in the current Brier score map
            category_values[key] = 'N/A'
    # Format the results into a string
    result_str = f"data_brier: {category_values['data_brier']}, prior_brier: {category_values['prior_brier']}, actual_brier: {category_values['actual_brier']}, ce_loss: {category_values['ce_loss']}"
    return result_str

def calculate_summary_statistics(matches, tokens):
    """
    Calculates overall percentages of ground truth and estimated values, as well as Brier scores for default and actual estimates.

    Args:
        matches (list): A list of dictionaries representing matches.
        tokens (dict): A dictionary mapping keys to their corresponding token values.

    Returns:
        dict: A dictionary containing summary statistics.
    """
    category_percents = {} # Average percent in the data for each category
    estimate_percents = {} # Average GPT-estimation for each category
    data_brier = {} # Brier score for a model that just predicted the overall percent in the data sample (somewhat unfair advantage)
    prior_brier = {} # Brier score for a model that just predicted the overall percent in the population
    actual_brier = {} # Brier score for GPT
    ce_loss = {} # Cross-entropy loss 
    confusion_matrix = {}
    auprc = {}

    for category in tokens:
        category_percents[category] = {}
        estimate_percents[category] = {}
        data_brier[category] = 0
        prior_brier[category] = 0
        actual_brier[category] = 0
        ce_loss[category] = -1
        auprc[category] = -1

    total_matches = len(matches)

    for match in matches:
        for category, data in match.items():
            if category not in tokens:
                print(f"Category {category} not found in tokens?!?")
                continue

            ground_truth = data['ground_truth']
            if ground_truth not in category_percents[category]:
                category_percents[category][ground_truth] = 0
            category_percents[category][ground_truth] += 1

            for estimate_key, estimate_value in data['estimate'].items():
                if estimate_key in tokens[category]['okc_vals']: # Skips case/leading-space variants which are rarely important
                    okc_val = tokens[category]['okc_vals'][estimate_key]
                    if okc_val not in estimate_percents[category]:
                        estimate_percents[category][okc_val] = 0
                    estimate_percents[category][okc_val] += utils.from_percent(estimate_value)

    for category in category_percents:
        for value in category_percents[category]:
            category_percents[category][value] = category_percents[category][value] / total_matches * 100

        for value in estimate_percents[category]:
            estimate_percents[category][value] /= total_matches

        for match in matches:
            # if category in match:
            if category in match:
                ground_truth = match[category]['ground_truth']
                # Actual Brier score
                for main_name, okc_name in tokens[category]['okc_vals'].items():
                    estimate_value = match[category]['estimate'].get(main_name)
                    if estimate_value is not None and main_name in tokens[category]['okc_vals']:
                        actual_brier[category] += (utils.from_percent(estimate_value) - 
                                                   (1 if okc_name == ground_truth else 0)) ** 2
                # Data Brier score
                for _, category_key in tokens[category]['okc_vals'].items():
                    category_value = category_percents[category].get(category_key)
                    if category_value is not None:
                        data_brier[category] += ((category_value / 100) - (1 if category_key == ground_truth else 0)) ** 2
                # Prior Brier score
                for _, category_key in tokens[category]['okc_vals'].items():
                    # estimate_key = estimate_key.strip().lower()
                    # if category_key in tokens[category]['okc_vals']:
                    # okc_val = tokens[category]['okc_vals'][estimate_key]
                    category_value = tokens[category]['priors'][category_key]
                    pass
                    if category_value is not None:
                        prior_brier[category] += ((category_value / 100) - (1 if category_key == ground_truth else 0)) ** 2
                    pass
        actual_brier[category] /= total_matches
        data_brier[category] /= total_matches
        prior_brier[category] /= total_matches

        ce_loss[category] = loss.ce_loss(matches, tokens[category], category)
        confusion_matrix[category] = loss.generate_confusion_matrix(matches, tokens[category], category)
        if GRAPH_AUPRC:
            auprc[category] = prc.plot_prc_and_calculate_auprc(matches, tokens[category], category)

    summary_statistics = {
        'category_percents': category_percents,
        'estimate_percents': estimate_percents,
        'data_brier': data_brier,
        'prior_brier': prior_brier,
        'actual_brier': actual_brier,
        'ce_loss': ce_loss,
        'confusion_matrix': confusion_matrix,
    }

    return summary_statistics

def summarize_matches(matches, category):
    # Extracting data and converting it to the desired format for plotting
    # Sloppy-ass GPT code
    reduced_data = []
    first_category = next(iter(matches[0][category]['estimate']))
    # print("<summarize_matches> base category: " + first_category + "; num matches: " + str(len(matches)))
    # print(f'MATCHES in summarize_matches: {matches}')
    # print(f'CATEGORY in summarize_matches: {category}')
    for item in matches:
        value = item[category]['estimate'].get(first_category,'50%')
        first_category_percentage = int(value.replace('%', ''))
        items = item[category]['estimate'].items()
        max_item = max(items, key=lambda item: int(item[1].replace('%', '')))
        reduced_data.append({'max_percent': first_category_percentage, 'Match?': item[category]['match?']})
    # print(f'REDUCED_DATA in summarize_matches: {reduced_data}') # XXX
    overall_match_percentage = len([x for x in reduced_data if x['Match?']]) / len(reduced_data) 
    print('Overall match percentage on ' + first_category + ': ' + str(overall_match_percentage))
    # Initialize an empty dictionary to count occurrences and matches in buckets
    percentage_counts = {}
    # Populate the dictionary with data, creating buckets as needed
    for entry in reduced_data:
        percentage = entry['max_percent']
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
    if SAVE_RESULTS:
        with open(f'{RESULTS_DIR}/reduced_data.json', 'w') as f:
            f.write(json.dumps(reduced_data))
    plot_data = [{'x': key, 'total': value['total'], 'percent correct': (value['matches'] / value['total']) * 100 if value['total'] > 0 else 0} for key, value in percentage_counts.items()]
    plot_data = sorted(plot_data, key=lambda m: m['x'])
    if SAVE_RESULTS:
        with open(f'{RESULTS_DIR}/plot_data.json', 'w') as f:
            f.write(json.dumps(plot_data))
    return plot_data, overall_match_percentage, first_category

def graph_matches(matches, correctness_statistics, summary_statistics):
    """Starts with the structure of matches, which is a list of maps of this form:
    `{'politics': {'match?': None, 'estimate': {' liberal': '100%', ' conservative': '0%'}}, 'gender': {'match?': True, 'estimate': {' female': '93%', ' male': '7%'}}, 'sexuality': {'match?': True, 'estimate': {' straight': '95%', ' gay': '3%', ' bisexual': '3%'}}, 'education': {'match?': None, 'estimate': {' Yes': '58%', ' No': '42%'}}, 'ethnicity': {'match?': True, 'estimate': {' White': '94%', ' Hispanic': '3%', ' Asian': '2%', ' Black': '0%'}}}`
    and summarizes them into a relation between the percent confidence of the estimate and the match status"""
    for subject in subjects:
        summary_data, overall_match_percentage, first_category = summarize_matches(matches, subject)
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
            title=f'Total vs. Correct/Incorrect Distribution and Percent Correct: {subject} ({first_category}), {utils.to_percent(overall_match_percentage)} of {len(matches)}.',
            xaxis_title=stringify_summary_statistics(summary_statistics, subject),
            yaxis=dict(title='Total Count', side='left', showgrid=False),
            yaxis2=dict(title='Percent Correct', side='right', overlaying='y', showgrid=False),
            hovermode='x'
            )

        # Add the line chart for 'percent correct' with the secondary y-axis
        fig.add_trace(go.Scatter(x=[item['x'] for item in summary_data], y=[item['percent correct'] for item in summary_data],
                                mode='lines+markers', name='Percent Correct', yaxis='y2', line=dict(color='#a59a52')))
        fig.show()

def confusion_metrics(cm):
    # True Positives are on the diagonal
    TP = np.diag(cm)
    # False Positives
    FP = np.sum(cm, axis=0) - TP
    # False Negatives
    FN = np.sum(cm, axis=1) - TP
    # True Negatives
    TN = np.sum(cm) - (FP + FN + TP)
    # Precision, Recall, and Accuracy
    precision = np.nan_to_num(TP / (TP + FP))
    recall = np.nan_to_num(TP / (TP + FN))
    accuracy = np.sum(TP) / np.sum(cm)
    return precision, recall, accuracy

def graph_confusion_matrix(confusion_matrix):
    for category in tokens:
        matrix = confusion_matrix.get(category)
        if matrix is None: 
            continue
        category_tokens = tokens[category]
        classes = loss.get_classes(category_tokens, category)
        if classes:
            precision, recall, accuracy = confusion_metrics(matrix)  # Calculate metrics from the confusion matrix
            fig = ff.create_annotated_heatmap(matrix, x=classes, y=classes, colorscale='Viridis')
            addendum = category_tokens.get('addendum')
            metrics_text = f'Precision: {np.mean(precision):.2f}, Recall: {np.mean(recall):.2f}, Accuracy: {accuracy:.2f}'
            title = f'Confusion Matrix for {category} ({NUM_PROFILES}): {addendum}<br>{metrics_text}'
            fig.update_layout(title=title, xaxis_title='Predicted Label', yaxis_title='Actual Label')
            fig.update_xaxes(side="bottom")  # Ensuring labels are at the bottom
            fig.show()

### Main profile processing

def matches_to_matches_by_topic(matches):
    if matches:
        return {key: [d[key] for d in matches] for key in matches[0]}
    else:
        return {}

def process_profile(profile):
    """process a single profile, sending the essay questions to OpenAI and 
    comparing the results to the user's ground truth demographics"""
    # Example profile (from okcupid):
    # {'age': '22', 'status': 'single', 'sex': 'm', 'orientation': 'straight', 'education': 'working on college/university', 'ethnicity': 'asian, white', 'income': '-1', 'job': 'transportation', 'location': 'south san francisco,...california', 'essay0': 'about me:  i would l...tion span.', 'essay1': 'currently working as...zy sunday.', 'essay2': 'making people laugh....implicity.', 'essay3': 'the way i look. i am... blend in.', 'essay4': 'books: absurdistan, ... anything.', ...}

    demographics = {k: profile[k] for k in ['age', 'sex', 'ethnicity', 'orientation', 'education', 'age']}
    # Call OpenAI to get demographic estimates
    try:
        model_guesses = openai_uk.call_openai(subjects, tokens, profile['essay'])
        if SAVE_RESULTS:
            with open(RESPONSES_FILE, 'a') as f: 
                f.write(json.dumps(profile) + '\n')
                f.write(json.dumps(model_guesses) + '\n\n')
    # We'll just skip any profiles that cause problems
    # except Exception as e:
    except BlockingIOError as e:
        print(f"Bad profile: {profile}")
        print(f"Error processing profile: {e}")
        return None
    # print()
    # print(f"Profile: {profile}")
    # print(f"Estimates: {model_guesses}")
    # print()
    # Compare the user's ground truth demographics to the estimates
    all_matches = {}
    if model_guesses is None:
        return None
    # print(f'PROFILE IN process_profile: {profile}') # XXX
    for subject in subjects:
        okc_name = tokens[subject]['okc_name']
        match = check_token_match(profile, model_guesses[subject], subject)
        all_matches[subject] = {'match?': match, 
                                'estimate': model_guesses[subject], 
                                'ground_truth': profile.get(okc_name)}
        # print("Was " + subject + " a match? " + str(match))
        # print(f"Ground truth: {profile.get(okc_name)}")
        # print(f"Estimate: {model_guesses[subject]}")
        # print()
    # print("all_matches for profile " + json.dumps(profile))
    # print(all_matches)
    return all_matches

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
    # Calibration creation or use
    if CREATE_CALIBRATION:
        for category in tokens:
            print(f'Creating calibration for {category}')
            if not matches:
                continue
            category_tokens = tokens[category]
            ground_truth, munged_matches = loss.np_arrays(matches, category_tokens, category)
            if ground_truth is None:
                continue
            calibrator = calibration.fitted_isotonic_calibrator(ground_truth, munged_matches)
            # calibrators[category] = calibrator # TODO delete, not used once we refuse to run a newly created calibrator
            file_path = calibration.get_file_path(RESULTS_DIR, category)
            calibrator.save(file_path)
    elif USE_CALIBRATION:
        for category in tokens:
            print(f'Calibrating {category}')
            if matches is None:
                continue
            category_tokens = tokens[category]
            _, munged_matches = loss.np_arrays(matches, category_tokens, category)
            if munged_matches is None:
                continue
            calibrator = calibrators[category]
            if not calibrator:
                print(f'No calibrator found for {category}')
                continue
            print(f'Original matches: {matches}')
            print(f'Munged matches: {munged_matches}')
            calibrated_matches = calibrator.calibrate(munged_matches)
            sorted_classes = sorted(category_tokens['okc_vals'].keys())
            print(f'Sorted classes: {sorted_classes}') # XXX
            for i, calibrated_match in enumerate(calibrated_matches):
                print(f'Current calibrated match: {calibrated_match}') # XXX
                for j, label in enumerate(sorted_classes):
                    print(f'Original match for {label}: {matches[i][category]['estimate'].get(label)}') # XXX
                    print(f'Setting it to {utils.to_percent(calibrated_match[j])}') # XXX
                    matches[i][category]['estimate'][label] = utils.to_percent(calibrated_match[j])
    matches_by_topic = matches_to_matches_by_topic(matches)
    # print(f'PROFILES IN process_profiles: {profiles}') # XXX
    # print(f'MATCHES IN process_profiles: {matches}') # XXX
    # print(f'MATCHES_BY_TOPIC IN process_profiles: {matches_by_topic}') # XXX
    if SAVE_RESULTS:
        with open(f'{RESULTS_DIR}/matches.json', 'w') as f:
            f.write(json.dumps(matches))
        with open(f'{RESULTS_DIR}/matches_by_topic.json', 'w') as f:
            f.write(json.dumps(matches_by_topic))
    # Calculate and print the correctness statistics
    correctness_statistics = calculate_correctness_statistics(matches_by_topic)
    print(f'CORRECTNESS_STATISTICS IN process_profiles: {correctness_statistics}') # XXX
    print(correctness_statistics)
    return matches, correctness_statistics

def main(subjects, ask_openai=False, dataset_module=okcupid):
    print(f'Analyzing {subjects} on {NUM_PROFILES} profiles.')
    offset = 2000 if CREATE_CALIBRATION else 0
    profiles = dataset_module.load_data(subjects, tokens=tokens, NUM_PROFILES=NUM_PROFILES, offset=offset)

    # Clear previous responses if they exist
    try:
        os.remove(RESPONSES_FILE)
    except FileNotFoundError:
        pass

    if USE_CALIBRATION and not CREATE_CALIBRATION: # Can't run a newly created calibration on the same data
        for category in tokens:
            print(f'Loading calibration for {category}')
            file_path = calibration.get_file_path(RESULTS_DIR, category)
            calibrators[category] = calibration.IsotonicCalibrator.load(file_path)
    if ask_openai:
        main_matches, correctness_statistics = process_profiles(profiles)
        # print(f'MAIN_MATCHES: {main_matches}') # XXX
        print(f'CORRECTNESS_STATISTICS: {correctness_statistics}') # XXX
    else:
        with open(f'{RESULTS_DIR}/matches.json', 'r') as f:
            main_matches = json.loads(f.read())
            matches_by_topic = matches_to_matches_by_topic(main_matches)
            correctness_statistics = calculate_correctness_statistics(matches_by_topic)
    summary_statistics = calculate_summary_statistics(main_matches, tokens)
    if GRAPH:
        graph_matches(main_matches, correctness_statistics, summary_statistics)
    if CONFUSION_MATRIX:
        graph_confusion_matrix(summary_statistics.get('confusion_matrix'))
    print(f'SUMMARY_STATISTICS: {summary_statistics}') # XXX

# TODO calculate length (in chars) of combined essays, get an average across the profiles, and see how it correlates with accuracy
# TODO in future can do this token-by-token on a single profile to see how accuracy changes per token
# subjects = ['politics', 'gender', 'sexuality', 'education', 'ethnicity', 'age']
subjects = ['gender', 'sexuality', 'education', 'ethnicity', 'age']
# subjects = ['gender', 'sexuality', 'ethnicity']
# subjects = ['ethnicity']

NUM_PROFILES = 3
main(subjects, ask_openai=ASK_OPENAI, dataset_module=okcupid) # persuade, okcupid

# TODO 
# - push some sharegpt data through
# - measure accuracy as a function of essays_len
# - try having gpt-4 create essays as a persona with specific demographics
# - prune data in various ways (gender: done)
# - maybe experiment with open ended description of pass