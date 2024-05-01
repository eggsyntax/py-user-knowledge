from uk01 import tokens
import openai_uk

from pprint import pprint

subjects = ['gender', 'sexuality', 'education', 'ethnicity', 'age']

def rename_key(dictionary, old_key, new_key):
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)  # Move value to new key and remove old key

def get_guesses(text):
    return openai_uk.call_openai(subjects, tokens, text)

def get_valid_classes(subject):
    return tokens[subject]['okc_vals']

def format_sorted_map(data_map, subject):
    valid_classes = get_valid_classes(subject)
    # Parse percentages and filter out '0%' values
    filtered_data = {key: int(value[:-1]) for key, value in data_map.items() if value != '0%'}
    # Drop random keys that don't correspond to valid classes
    valid_data = {key: value for key, value in filtered_data.items() if key in valid_classes}
    rename_key(valid_data, 'bis', 'bisexual')
    sorted_data = sorted(valid_data.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{key}: {value}%" for key, value in sorted_data)

def pretty_print(guesses):
    for subject, guess in guesses.items():
        pretty_guess = format_sorted_map(guess, subject)
        if subject == 'age':
            subject = 'age > 30'
        if subject == 'education':
            subject = 'college-educated'
        print(f"{subject:16s} -> {pretty_guess}")

def main():
    while True:
        user_input = input("\n\nPLEASE ENTER THE TEXT TO EVALUATE\n> ")
        result = get_guesses(user_input)
        pretty_print(result)

if __name__ == "__main__":
    main()
