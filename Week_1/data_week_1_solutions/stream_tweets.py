import twitter
import os
import json

consumer_key = os.environ['twitter_consumer_key']
consumer_secret = os.environ['twitter_consumer_secret']
access_token = os.environ['twitter_access_token']
access_token_secret = os.environ['twitter_access_secret']


api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token,
                  access_token_secret=access_token_secret)


FILTER = ['#datascience']
LANGUAGES = ['en']


def main(path, FILTER, LANGUAGES):
    with open(path + 'output.txt', 'a') as f:
        # api.GetStreamFilter will return a generator that yields one status
        # message (i.e., Tweet) at a time as a JSON dictionary.
        for line in api.GetStreamFilter(track=FILTER, languages=LANGUAGES):
            f.write(json.dumps(line))
            f.write('\n')
            
main(path='', FILTER=FILTER, LANGUAGES=LANGUAGES)