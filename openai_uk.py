import math
import requests

### OpenAI requests & response handling

def openai(context, tokenIds={}): 
    numTokens = len(tokenIds) if tokenIds else 5
    response = requests.post('https://api.openai.com/v1/completions', json={
        'model': "gpt-3.5-turbo-instruct",
        'prompt': context,
        'max_tokens': 1,
        'logprobs': numTokens,
        'temperature': 0,
        'logit_bias': tokenIds
    }, headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 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'
    })
    data = response.json()
    return data

def call_openai(subjects, tokens, context_input):
    user_estimates = {}
    for subject in subjects:
        # print("SUBJECT:")
        # print(subject)
        topic_info = tokens[subject]
        addendum = topic_info['addendum']
        # print(addendum)
        modified_context = f"```{context_input}``` \n{addendum}"
        tokenIds = map_token_id_to_bias(topic_info)
        data = openai(modified_context, tokenIds)
        try:
            logprobs = data['choices'][0]['logprobs']['top_logprobs'][-1]
        except KeyError as e:
            print(f"Error processing response: {e}")
            print(f"Context: {modified_context}")
            print(f"JSON data: {data}")
            return None
        # print('LOGPROBS') # TEMP
        # print(logprobs) # TEMP
        # display_logprobs(subject, data['choices'][0]['logprobs'])
        probs = convert_log_probs_to_percentages(logprobs)
        user_estimates[subject] = probs
        # print(user_estimates)
        # print(probs) # TEMP
        # print('probs') # TEMP
        # display_logprobs(subject, data['choices'][0]['logprobs'])
        user_estimates[subject] = probs
    # print('user_estimates')
    # print(user_estimates)
    return user_estimates

def convert_log_probs_to_percentages(log_probs):
    converted = {}
    for key, value in log_probs.items():
        # print("key:") # TEMP
        # print(key) # TEMP
        prob = math.exp(value)
        percentage = f"{prob * 100:.0f}%"
        converted[key] = percentage
    return converted

def map_token_id_to_bias(topic_info):
    """`topics` contains a map from option name to token ID, and another from option name to desired bias. 
    This is more readable, but we actually need to *pass* a map from token ID to bias; this function creates 
    that map."""
    tokenIdToBiasMap = {}
    tokenIds = topic_info['tokenIds']
    biases = topic_info['bias']
    for token, tokenId in tokenIds.items():
        bias = biases[token]
        tokenIdToBiasMap[tokenId] = bias
    return tokenIdToBiasMap

