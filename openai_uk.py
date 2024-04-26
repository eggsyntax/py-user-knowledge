import math
import requests

### OpenAI requests & response handling

def get_prompt(message, subject, addendum):
  prompt_messages = {
      'gender': [
        {
          "role": "system",
          "content": "You are a helpful assistant who helps determine information about the author of texts. You only ever answer with a single word: one of the exact choices the user provides."
        },
        {
          "role": "user",
          "content": f"""'''adventurous. ambitious. amusing. dreamer. sarcastic. stubborn.	racing. against. time. walking into trees learning	boiling water	my smile.	i. love. ramen. and south park	family hope love spicy peppers laughter chocolate	being a better person.	lounging on a hammock	i am chronically diagnosed as a terrible driver.	curiosity does not kill the cat''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "female"
        },
        {
          "role": "user",
          "content": f"""'''bay area native, recent ma resident, college grad, nerd, former copy editor, feminist, poor typist, dangerous amateur rapper	working, furnishing my apartment, settling in	making cereal	my sweet dance moves	books: opened ground -- seamus heaney; a farewell to arms -- ernest hemingway; the hawk in the rain -- ted hughes; buffett: the making of an american capitalist -- roger lowenstein  movies & tv shows, but i don't watch much...: cairo time, layer cake, parks & rec  music: beatles, daft punk, hird, michael jackson, plej, royksopp, sufjan stevens, stevie wonder, jay-z. i'm also listening to belle & sebastian and shad. i listen to one atb song -- the one with the lyric "i've been running like a sentence never begun." isn't that line amusing? let me know if you agree.  food: thai curry, burritos, rogan josh, chicken korma, shrimp and grits, avocado and whole-grain mustard open-faced sandwiches, ben & jerry's chunky monkey	1. my glasses 2. my contacts 3. sleep 4. tea -- darjeeling, english breakfast, and green; not so much a fan of the earl grey 5. (value) investing 6. excel shortcuts	running and how i'd like to do it more, philosophy (majored in it), investing (accurately representing returns on capital investments), obsolescence, written correspondence (style in letters), communication, people, should i go to grad school? (for what? what for?), "and me" vs. "and i" grammar	perhaps: sleeping reading with friends (maybe a party) at a bar	i've been to a small-town rodeo.	you're interested you're smart you wouldn't be more than friends without being friends also''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "male"
        },
        {
          "role": "user",
          "content": f"""'''talking about myself does not come that easily to me (at least, not in these sorts of open forums.) however, i do have a lot to say (including plenty about myself, once i get started) but it's much easier to do in person.  i'm originally from ireland, but i've been living san francisco long enough now to not imagine having a home anywhere else.  i am looking for someone who is funny and conversational, witty and unexpected, attractive (did i say that already?) & all with her own mind and opinion...  i'm someone who spends a lot of time at home, but who's also a lover of outdoors, and a pursuer of the goal of great personal fitness (to the great pain of those around me!)  i'm opinionated & occasionally cranky, but invariably, i'm in the wrong & what i need is someone to tell me how things are, no matter what an ass i am about taking advice...	ah!!	organizing, getting to places on time, navigating the muni	uh, probably my intensity level, which can be either a) a very good thing (i'm a very good listener, and i always at least appear to be completely interested in who i'm talking to) or b) well, intense...	i go through tv-obsessive phases -- i really enjoyed the recent shows, "the killing" & "game of thrones". and... needless to say, deadwood is an all-time favorite.	my tivo milano cookies golden gate park indulgent restaurant dinners	my plight and circumstances. (i s'pose i do a fair bit of navel-gazing) software development problems (it's what i do... not necessarily for fun!) arsenal fc	dining and drinking!		you should message me! (i'm new on this, and i want lots of messages - otherwise... what's the point?)''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "male"
        },
        {
          "role": "user",
          "content": f"""'''{message}''' \n{addendum}"""
        },
      ],
      'ethnicity': [
        {
          "role": "system",
          "content": "You are a helpful assistant who helps determine information about the author of texts. You only ever answer with a single word: one of the exact choices the user provides."
        },
        {
          "role": "user",
          "content": f"""'''the consensus is i am a very laid back, friendly, happy person. i studied marine biology in college. i love to travel. over the last few years, i have spent significant amounts of time in japan, new orleans, los angeles, and mexico. i like experiencing new things. even though i was brought up in the bay area, i feel like there is still a lot to discover here.  places you may find me: the beach- bonus points if there is a tidepooling area. the tennis court- i am a bit rusty, but it is my sport of choice. the wilderness- camping is so much fun. my backyard- playing bocce ball and grillin' like a villain. the bowling alley- we may be the worst team in the league, but it's all about having fun right? san francisco: so many museums, parks, aquariums, etc. local sporting event: go warriors/giants/niners/sharks! a concert: nothing like live music. my couch: beating the hell out of someone at mario kart.	i work in the environmental field, which i love. for the past year i have spent about five months in new orleans doing studies on the oil spill. most of my free time is spent with friends and family, having as much fun as possible.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "white"
        },
        {
          "role": "user",
          "content": f"""'''i'm a student, a son, a soccer player, salsa dancer, photographer. the list goes on and on. if you want to be my friend, buy me a veggie burrito :d i love going on spontaneous walks, especially taking bart to san fransisco to meet new and interesting characters that compose the beautiful city of sf. i completely love being active and try to stay active 24/7. but, i dont mind staying in and watching a good zombie flick or comedy. i think coffee shops were the best things ever invented. tea=mouthful of bliss. i'm a karaoke champ- at least i'd like to think so. one day i plan to make a difference. let's get this show on the road.	i am a student at ucdavis. when i'm not busy with school, i play soccer or take my cool eos rebel k2 camera to take pictures of eccentric monuments that please my eyes.	i'm good at editing and critiquing papers. i am a closet math genius. oh yea, you know that s-curve on the bay bridge you take just before hitting the tunnel? well i can take that curve at 50 miles per hour like a pro.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "hispanic"
        },
        {
          "role": "user",
          "content": f"""'''when i was a kid, - i thought cartoons were real people and places covered in tin foil and painted - i had a donut conveyor belt for my personal use after hours, and - i got the bait and switch where art camp turned out to be math camp.  when i got older, - i quit 8th grade, like it was a job i could opt out of  these days, - i stick with hbo - i don't know when to quit - i play with robots for science - and, i pay too much money for donuts.	i'm an engineer @ a medical devices company. i'm an amateur cook & avid baker. i camp, glamp, hike and cycle. i'll try anything once.	not knowing how to swim properly.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "asian"
        },
        {
          "role": "user",
          "content": f"""'''{message}''' \n{addendum}"""
        },
      ],
      'sexuality': [
        {
          "role": "system",
          "content": "You are a helpful assistant who helps determine information about the author of texts. You only ever answer with a single word: one of the exact choices the user provides."
        },
        {
          "role": "user",
          "content": f"""'''i was born in kiev, ukraine. i am very lucky that my family decided to get out of the former soviet union, as we had little opportunities there. in 1989, we immigrated to us and san francisco became my second home. i went to school in dallas, tx where i spent 5 years studying music and finance. i now work as an accountant during the day and as a music teacher/performer during evenings and weekends. i love life and open to any new experiences. i rarely get a chance to sit home because i am always rehearsing, walking the dog, working, going to art shows. i cook occasionally but what fun is it to cook for yourself? i tend to pick up on the go and move on to a more intellectual activity. i used to take dancing classes at the city college and i miss going to lindy in the park. i know there are tons of interesting people in san francisco and i am hoping to meet some of them.	living it the best i can.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "bisexual"
        },
        {
          "role": "user",
          "content": f"""'''bay area native, recent ma resident, college grad, nerd, former copy editor, feminist, poor typist, dangerous amateur rapper	working, furnishing my apartment, settling in	making cereal	my sweet dance moves	books: opened ground -- seamus heaney; a farewell to arms -- ernest hemingway; the hawk in the rain -- ted hughes; buffett: the making of an american capitalist -- roger lowenstein  movies & tv shows, but i don't watch much...: cairo time, layer cake, parks & rec  music: beatles, daft punk, hird, michael jackson, plej, royksopp, sufjan stevens, stevie wonder, jay-z. i'm also listening to belle & sebastian and shad. i listen to one atb song -- the one with the lyric "i've been running like a sentence never begun." isn't that line amusing? let me know if you agree.  food: thai curry, burritos, rogan josh, chicken korma, shrimp and grits, avocado and whole-grain mustard open-faced sandwiches, ben & jerry's chunky monkey	1. my glasses 2. my contacts 3. sleep 4. tea -- darjeeling, english breakfast, and green; not so much a fan of the earl grey 5. (value) investing 6. excel shortcuts	running and how i'd like to do it more, philosophy (majored in it), investing (accurately representing returns on capital investments), obsolescence, written correspondence (style in letters), communication, people, should i go to grad school? (for what? what for?), "and me" vs. "and i" grammar	perhaps: sleeping reading with friends (maybe a party) at a bar	i've been to a small-town rodeo.	you're interested you're smart you wouldn't be more than friends without being friends also''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "straight"
        },
        {
          "role": "user",
          "content": f"""'''i should probably change some aspects of this profile, now that i've moved from the west coast... midwesterner to the core, though sometimes mistaken for european because of my bass mumbling. raised catholic, but can pass as jewish. according to my tests, i'll die at 79 and i'm a democrat; i am a democrat, but i think i'll live forever.	must...finish...dissertation...	morning phone conversations and late-night tv marathons. dry delivery.	eyes, voice, walk.	pride and prejudice, brideshead revisited, angels in america. recently saw the secret of kells and it is awesome. currently following american horror story on tv and american office and charmed (yes, charmed) on netflix. scarlatti, kronos quartet, lady gaga. food-wise, i am really missing california.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "gay"
        },
        {
          "role": "user",
          "content": f"""'''{message}''' \n{addendum}"""
        },
      ],
'education': [
        {
          "role": "system",
          "content": "You are a helpful assistant who helps determine information about the author of texts. You only ever answer with a single word: one of the exact choices the user provides."
        },
        {
          "role": "user",
          "content": f"""'''adventurous. ambitious. amusing. dreamer. sarcastic. stubborn.	racing. against. time. walking into trees learning	boiling water	my smile.	i. love. ramen. and south park	family hope love spicy peppers laughter chocolate	being a better person.	lounging on a hammock	i am chronically diagnosed as a terrible driver.	curiosity does not kill the cat''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "yes"
        },
        {
          "role": "user",
          "content": f"""'''so, i've lived in the bay for a couple years..2 to be exact, but i love so many things about it and really feel at home. i love children. i'm a part time nanny and a full time student. kinda busy but definitely looking for people to hang out with in the area. i live by lake merritt and do see a little 'lady of the lake' in me. i'm majoring in psychology and am interested in cognitve behavioral therapy for children and adolescence. i'm around kids a lot and love to be as creative, fun, playful, and caring as them all the time. i did a lot of youth organizing in my old home town and belive without youth empowerment theres hardly anywhere to go. go youth!  anyways message me and lets talk :)	a lot? i guess i'm trying to figure it out still. i have many goals and destinations and now i'm figuring out how to create my path to get to them.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "no"
        },
        {
          "role": "user",
          "content": f"""'''i'm originally from nyc, moved here 3+ years ago and so far so good.  i'd rather meet in person than trade endless emails. if we are a good match on paper, let's just meet for a quick drink... no pressure. i'm a big believer in physical chemistry and there's only one way to find out.	livin' it up	eating, drinking, smiling, laughing. turning you on to really good music you haven't heard before. finding patterns, seeing the bigger picture.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "yes"
        },
        {
          "role": "user",
          "content": f"""'''{message}''' \n{addendum}"""
        },
      ],
      'age': [
        {
          "role": "system",
          "content": "You are a helpful assistant who helps determine information about the author of texts. You only ever answer with a single word: one of the exact choices the user provides."
        },
        {
          "role": "user",
          "content": f"""'''i was born in kiev, ukraine. i am very lucky that my family decided to get out of the former soviet union, as we had little opportunities there. in 1989, we immigrated to us and san francisco became my second home. i went to school in dallas, tx where i spent 5 years studying music and finance. i now work as an accountant during the day and as a music teacher/performer during evenings and weekends. i love life and open to any new experiences. i rarely get a chance to sit home because i am always rehearsing, walking the dog, working, going to art shows. i cook occasionally but what fun is it to cook for yourself? i tend to pick up on the go and move on to a more intellectual activity. i used to take dancing classes at the city college and i miss going to lindy in the park. i know there are tons of interesting people in san francisco and i am hoping to meet some of them.	living it the best i can.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "yes"
        },
        {
          "role": "user",
          "content": f"""'''bay area native, recent ma resident, college grad, nerd, former copy editor, feminist, poor typist, dangerous amateur rapper	working, furnishing my apartment, settling in	making cereal	my sweet dance moves	books: opened ground -- seamus heaney; a farewell to arms -- ernest hemingway; the hawk in the rain -- ted hughes; buffett: the making of an american capitalist -- roger lowenstein  movies & tv shows, but i don't watch much...: cairo time, layer cake, parks & rec  music: beatles, daft punk, hird, michael jackson, plej, royksopp, sufjan stevens, stevie wonder, jay-z. i'm also listening to belle & sebastian and shad. i listen to one atb song -- the one with the lyric "i've been running like a sentence never begun." isn't that line amusing? let me know if you agree.  food: thai curry, burritos, rogan josh, chicken korma, shrimp and grits, avocado and whole-grain mustard open-faced sandwiches, ben & jerry's chunky monkey	1. my glasses 2. my contacts 3. sleep 4. tea -- darjeeling, english breakfast, and green; not so much a fan of the earl grey 5. (value) investing 6. excel shortcuts	running and how i'd like to do it more, philosophy (majored in it), investing (accurately representing returns on capital investments), obsolescence, written correspondence (style in letters), communication, people, should i go to grad school? (for what? what for?), "and me" vs. "and i" grammar	perhaps: sleeping reading with friends (maybe a party) at a bar	i've been to a small-town rodeo.	you're interested you're smart you wouldn't be more than friends without being friends also''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "no"
        },
        {
          "role": "user",
          "content": f"""'''i should probably change some aspects of this profile, now that i've moved from the west coast... midwesterner to the core, though sometimes mistaken for european because of my bass mumbling. raised catholic, but can pass as jewish. according to my tests, i'll die at 79 and i'm a democrat; i am a democrat, but i think i'll live forever.	must...finish...dissertation...	morning phone conversations and late-night tv marathons. dry delivery.	eyes, voice, walk.	pride and prejudice, brideshead revisited, angels in america. recently saw the secret of kells and it is awesome. currently following american horror story on tv and american office and charmed (yes, charmed) on netflix. scarlatti, kronos quartet, lady gaga. food-wise, i am really missing california.''' \n{addendum}"""
        },
        {
          "role": "assistant",
          "content": "yes"
        },
        {
          "role": "user",
          "content": f"""'''{message}''' \n{addendum}"""
        },
      ],

  }
  return prompt_messages[subject]

def openai(message, addendum, subject, tokenIds={}): 
    numTokens = len(tokenIds) if tokenIds else 5
    messages = get_prompt(message, subject, addendum)

    try:
        params = {
            'model': "gpt-3.5-turbo",
            # 'model': "gpt-4-turbo-2024-04-09",
            'messages': messages,
            'max_tokens': 1,
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
    # print(f'Context input: {context_input}') # XXX
    user_estimates = {}
    for subject in subjects:
        # print("SUBJECT:")
        # print(subject)
        topic_info = tokens[subject]
        addendum = topic_info['addendum']
        # print(addendum)
        tokenIds = map_token_id_to_bias(topic_info)
        data = openai(context_input, addendum, subject, tokenIds)
        try:
            logprobs = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        except KeyError as e:
            print(f"Error processing response: {e}")
            print(f"Context: {modified_context}")
            print(f"JSON data: {data}")
            return None
        # print('LOGPROBS') # TEMP
        # print(logprobs) # TEMP
        probs = convert_log_probs_to_percentages(logprobs)
        # print(user_estimates)
        # print(probs) # TEMP
        # print('probs') # TEMP
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
