import math
import requests

### OpenAI requests & response handling

def openai(message, addendum, tokenIds={}): 
    numTokens = len(tokenIds) if tokenIds else 5
    messages = [
      {
        "role": "system",
        "content": "You are a helpful assistant who helps determine information about the author of texts. You only ever answer with a single word: one of the exact choices the user provides (eg 'male' or 'female')."
      },
      {
        "role": "user",
        "content": """'''adventurous. ambitious. amusing. dreamer. sarcastic. stubborn.	racing. against. time. walking into trees learning	boiling water	my smile.	i. love. ramen. and south park	family hope love spicy peppers laughter chocolate	being a better person.	lounging on a hammock	i am chronically diagnosed as a terrible driver.	curiosity does not kill the cat''' \n{addendum}"""
      },
      {
        "role": "assistant",
        "content": "female"
      },
      {
        "role": "user",
        "content": """'''bay area native, recent ma resident, college grad, nerd, former copy editor, feminist, poor typist, dangerous amateur rapper	working, furnishing my apartment, settling in	making cereal	my sweet dance moves	books: opened ground -- seamus heaney; a farewell to arms -- ernest hemingway; the hawk in the rain -- ted hughes; buffett: the making of an american capitalist -- roger lowenstein  movies & tv shows, but i don't watch much...: cairo time, layer cake, parks & rec  music: beatles, daft punk, hird, michael jackson, plej, royksopp, sufjan stevens, stevie wonder, jay-z. i'm also listening to belle & sebastian and shad. i listen to one atb song -- the one with the lyric "i've been running like a sentence never begun." isn't that line amusing? let me know if you agree.  food: thai curry, burritos, rogan josh, chicken korma, shrimp and grits, avocado and whole-grain mustard open-faced sandwiches, ben & jerry's chunky monkey	1. my glasses 2. my contacts 3. sleep 4. tea -- darjeeling, english breakfast, and green; not so much a fan of the earl grey 5. (value) investing 6. excel shortcuts	running and how i'd like to do it more, philosophy (majored in it), investing (accurately representing returns on capital investments), obsolescence, written correspondence (style in letters), communication, people, should i go to grad school? (for what? what for?), "and me" vs. "and i" grammar	perhaps: sleeping reading with friends (maybe a party) at a bar	i've been to a small-town rodeo.	you're interested you're smart you wouldn't be more than friends without being friends also''' \n{addendum}"""
      },
      {
        "role": "assistant",
        "content": "male"
      },
      {
        "role": "user",
        "content": """'''talking about myself does not come that easily to me (at least, not in these sorts of open forums.) however, i do have a lot to say (including plenty about myself, once i get started) but it's much easier to do in person.  i'm originally from ireland, but i've been living san francisco long enough now to not imagine having a home anywhere else.  i am looking for someone who is funny and conversational, witty and unexpected, attractive (did i say that already?) & all with her own mind and opinion...  i'm someone who spends a lot of time at home, but who's also a lover of outdoors, and a pursuer of the goal of great personal fitness (to the great pain of those around me!)  i'm opinionated & occasionally cranky, but invariably, i'm in the wrong & what i need is someone to tell me how things are, no matter what an ass i am about taking advice...	ah!!	organizing, getting to places on time, navigating the muni	uh, probably my intensity level, which can be either a) a very good thing (i'm a very good listener, and i always at least appear to be completely interested in who i'm talking to) or b) well, intense...	i go through tv-obsessive phases -- i really enjoyed the recent shows, "the killing" & "game of thrones". and... needless to say, deadwood is an all-time favorite.	my tivo milano cookies golden gate park indulgent restaurant dinners	my plight and circumstances. (i s'pose i do a fair bit of navel-gazing) software development problems (it's what i do... not necessarily for fun!) arsenal fc	dining and drinking!		you should message me! (i'm new on this, and i want lots of messages - otherwise... what's the point?)''' \n{addendum}"""
      },
      {
        "role": "assistant",
        "content": "male"
      },
      {
        "role": "user",
        "content": f"""'''{message}''' \n{addendum}"""
      },
    ]
    try:
        params = {
            'model': "gpt-3.5-turbo",
            'messages': messages,
            # 'max_tokens': 1,
            'logprobs': True,
            # 'logprobs': numTokens,
            'top_logprobs': 5,
            'temperature': 0,
            # 'logit_bias': tokenIds
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', json=params, headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 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'
        })
        data = response.json()
    except Exception as e:
        print(f"Error in request or response: {e}")
        if response:
            print(f"Response: {response}")
        return None
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
            logprobs = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        except KeyError as e:
            print(f"Error processing response: {e}")
            print(f"Context: {modified_context}")
            print(f"JSON data: {data}")
            return None
        # print('LOGPROBS') # TEMP
        # print(logprobs) # TEMP
        # display_logprobs(subject, data['choices'][0]['logprobs'])
        probs = convert_log_probs_to_percentages(logprobs)
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
    for logprob in log_probs:
        token = logprob['token']
        value = logprob['logprob']
        # print("key:") # TEMP
        # print(key) # TEMP
        prob = math.exp(value)
        percentage = f"{prob * 100:.0f}%"
        converted[token] = percentage
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

