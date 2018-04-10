import base64
import json
import globals
from urllib import urlencode
from urllib2 import urlopen, Request

consumer_key = "hDhDiDyA7J5g36Qw9eFPnEnlS"
consumer_secret = "OofP95eIwpKxC4BV6NI24HiTPS1ScwsRxYtyV3y1dwFBUBpWfA"

API_ENDPOINT = 'https://api.twitter.com'
API_VERSION = '1.1'

["Michigan", "goblue"]

def calcRetweets(data):
    retweets = data['retweet_count']
    return retweets

def calcFavorites(data):

    pass

def calcMichiganMentions(data):
    pass

def standardRequest(url, access_token):
    request = Request(url)
    request.add_header('Authorization', 'Bearer %s' % access_token)
    response = urlopen(request)
    raw_data = response.read().decode('utf-8')
    data = json.loads(raw_data)
    return data

def authorize(url, consumer_key, consumer_secret):
    bearer_token = '%s:%s' % (consumer_key, consumer_secret)
    encoded_bearer_token = base64.b64encode(bearer_token.encode('ascii'))

    # if no data and application/x-www-form-urlencoded, then post
    # if data or no data and no stipulation, then get
    request = Request(url)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded') # must be in for post
    request.add_header('Authorization', 'Basic %s' % encoded_bearer_token.decode('utf-8'))
    request_data = 'grant_type=client_credentials'.encode('ascii')
    request.add_data(request_data)

    response = urlopen(request)
    raw_data = response.read().decode('utf-8')
    data = json.loads(raw_data)

    access_token = data['access_token']

    return access_token


def batcher(token, recruits):
    base_url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    recruitsListTweets = {}
    for recruit in recruits:
        # find last tweet since national signing day
        params = {"screen_name": recruit, "count": str(200), "include_rts": 1, "max_id": str(globals.nsd2016maxID)}
        timeline_data = []

        while True:
            url = (base_url + "?" + urlencode(params))

            try:
                tweet_data = standardRequest(url, token)
            except:
                break

            if not tweet_data:
                break

            timeline_data.append(tweet_data)

            last_element = tweet_data[-1]
            amount_tweets = len(tweet_data)

            newMaxId = last_element['id']
            if (last_element['id'] < globals.nsd2016lowerID) or amount_tweets < params["count"]:
                break

            params["max_id"] = str(newMaxId)

        recruitsListTweets[params["screen_name"]] = timeline_data

    return recruitsListTweets

def main():
    REQUEST_TOKEN_URL = '%s/oauth2/token' % (API_ENDPOINT)
    token = authorize(REQUEST_TOKEN_URL, consumer_key, consumer_secret)

    recruitsListTweets = batcher(token, globals.recruits2016)
    print recruitsListTweets

if __name__ == '__main__':
    main()