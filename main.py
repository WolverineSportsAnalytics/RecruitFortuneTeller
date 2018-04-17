import base64
import json
import globals
from urllib import urlencode
from urllib2 import urlopen, Request
import pandas as pd
import csv

consumer_key = "hDhDiDyA7J5g36Qw9eFPnEnlS"
consumer_secret = "OofP95eIwpKxC4BV6NI24HiTPS1ScwsRxYtyV3y1dwFBUBpWfA"

API_ENDPOINT = 'https://api.twitter.com'
API_VERSION = '1.1'

hashTags = ["mich",
             "uofm",
             "goblue",
             "go blue",
             "umich",
             "wolv",
             "wolverines",
             "wolverine",
             "michigan",
             "hail",
             "victor",
             "victors",
             "blueblood",
             "harbaugh",
             "bighouse",
             "re2spect",
             "umfootball",
             "michfootball",
             "uofmfootball",
             "theteam",
             "thosewhostay",
             "hailtothevictors",
             "303D FE0F"]

negativeWords = ["state", "st.", "st", "central", "eastern", "gogreen", "green", "eastern", "western", "central", "emu",
                 "swoop", "getup", "wmu", "bronco", "eagle", "rowtheboat", "cmu", "chipp", "fireup", "northern", "tech"]

# container class for twitter information
class Twitter:
    def __init__(self, screenName, michFavToAllTweetRatio, michTweetToAllTweetRatio, michOverallTweetRatio,
                 michNativeRTweetRatio, michNativeTweetRatio):
        self.screenName = screenName
        self.michFavToAllTweetRatio = michFavToAllTweetRatio
        self.michTweetToAllTweetRatio = michTweetToAllTweetRatio
        self.michOverallTweetRatio = michOverallTweetRatio
        self.michNativeRTweetRatio = michNativeRTweetRatio
        self.michNativeTweetRatio = michNativeTweetRatio

    def to_dict(self):
        return {
            'screenName': self.screenName,
            'michFavToAllTweetRatio': self.michFavToAllTweetRatio,
            'michTweetToAllTweetRatio': self.michTweetToAllTweetRatio,
            'michOverallTweetRatio': self.michOverallTweetRatio,
            'michNativeRTweetRatio': self.michNativeRTweetRatio,
            'michNativeTweetRatio': self.michNativeTweetRatio
        }

def calcRetweets(tweet):
    retweets = tweet['retweet_count']
    return retweets


def calcFavorites(tweet):
    favorites = tweet['favorite_count']
    return favorites


def calcMichiganMentions(data):
    # check each word in data and if it matches
    twitterData = []
    for screenName, tweetsObject in data.iteritems():
        for tweets in tweetsObject:
            numTweetsAnalyzing = len(tweets)
            nativeTweets = 0
            nativeRetweets = 0
            nativeMichiganTweets = 0
            nativeMichiganRetweets = 0
            michiganTweets = 0

            michFavorites = 0
            michRetweets = 0

            totFavorites = 0
            totRetweets = 0

            screenNameTwitterData = {}
            screenNameTwitterData['screen_name'] = screenName
            screenNameTwitterData['tweet_metrics'] = {}

            print "Analyzing Tweets for: " + screenName

            for tweet in tweets:
                if "retweeted_status" in tweet:
                    retweeted = True
                else:
                    retweeted = False

                if retweeted:
                    nativeRetweets += 1
                else:
                    nativeTweets += 1
                    favorites = calcFavorites(tweet)
                    totFavorites += favorites

                    retweets = calcRetweets(tweet)
                    totRetweets += retweets

                words = []
                tweetText = tweet['text']
                words = tweetText.split(" ")
                hashtags = tweet['entities']['hashtags']
                for taghash in hashtags:
                    words.append(str(taghash['text']))

                if isMichigan(words):
                    michiganTweets += 1
                    if retweeted:
                        print "RT: " + str(words)
                        nativeMichiganRetweets += 1
                    else:
                        print "Native Tweet: " + str(words)
                        nativeMichiganTweets += 1
                        michFavorites += calcFavorites(tweet)
                        michRetweets += calcRetweets(tweet)

            michRetweetsRatio = 0 if nativeMichiganTweets == 0 else float(michRetweets / nativeMichiganTweets)
            michFavoritesRatio = 0 if nativeMichiganTweets == 0 else float(michFavorites / nativeMichiganTweets)

            tweetRatio = - 0 if nativeTweets == 0 else float(totRetweets / nativeTweets)
            favoriteRatio = 0 if nativeTweets == 0 else float(totFavorites / nativeTweets)

            michFavToAllTweetRatio = 0 if favoriteRatio == 0 else float(michFavoritesRatio / favoriteRatio)
            michTweetToAllTweetRatio = 0 if tweetRatio == 0 else float(michRetweetsRatio / tweetRatio)

            michOverallTweetRatio = 0 if numTweetsAnalyzing == 0 else float(michiganTweets / numTweetsAnalyzing)
            michNativeRTweetRatio = 0 if nativeRetweets == 0 else float(nativeMichiganRetweets / nativeRetweets)
            michNativeTweetRatio = 0 if nativeMichiganTweets == 0 else float(nativeMichiganTweets / nativeMichiganTweets)

            screenNameTwitterData['tweet_metrics']['michFavToAllTweetRatio'] = michFavToAllTweetRatio
            screenNameTwitterData['tweet_metrics']['michTweetToAllTweetRatio'] = michTweetToAllTweetRatio

            screenNameTwitterData['tweet_metrics']['michOverallTweetRatio'] = michOverallTweetRatio
            screenNameTwitterData['tweet_metrics']['michNativeRTweetRatio'] = michNativeRTweetRatio
            screenNameTwitterData['tweet_metrics']['michNativeTweetRatio'] = michNativeTweetRatio

            twitterData.append(screenNameTwitterData)

    return twitterData

def isMichigan(word_list):
    tagString = " ".join(
        word_list)  # join into one string
    tagString = tagString.lower()
    for tag in negativeWords:
        if tagString.find(tag) != -1:
            return 0

    for tag in hashTags:  # for word in hashTags
        if tagString.find(
                tag) != -1:  # if word is in tagString
            return 1  # return 1

    return 0  # else return 0

def toPandas(screen_name_to_twitter_data):
    twitterObjects = []

    for tweet_summary in screen_name_to_twitter_data:
        screenName = tweet_summary['screen_name']

        twitter_data = tweet_summary['tweet_metrics']
        michFavToAllTweetRatio = twitter_data['michFavToAllTweetRatio']
        michTweetToAllTweetRatio = twitter_data['michTweetToAllTweetRatio']
        michOverallTweetRatio = twitter_data['michOverallTweetRatio']
        michNativeRTweetRatio = twitter_data['michNativeRTweetRatio']
        michNativeTweetRatio = twitter_data['michNativeTweetRatio']
        tweet = Twitter(screenName, michFavToAllTweetRatio, michTweetToAllTweetRatio, michOverallTweetRatio,
                        michNativeRTweetRatio, michNativeTweetRatio)
        twitterObjects.append(tweet)

    # https://stackoverflow.com/questions/34997174/how-to-convert-list-of-model-objects-to-pandas-dataframe
    twitter_panda = pd.DataFrame.from_records([t.to_dict() for t in twitterObjects])

    return twitter_panda


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
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')  # must be in for post
    request.add_header('Authorization', 'Basic %s' % encoded_bearer_token.decode('utf-8'))
    request_data = 'grant_type=client_credentials'.encode('ascii')
    request.add_data(request_data)

    response = urlopen(request)
    raw_data = response.read().decode('utf-8')
    data = json.loads(raw_data)

    access_token = data['access_token']

    return access_token

def followers(token):
    baseURL = "https://api.twitter.com/1.1/friends/ids.json"
    pass

def batcher(token, recruits):
    base_url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    recruitsListTweets = {}

    recruitsTwitterData = []
    recruitsNonTwitterData = []

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

        # check to see if any tweet data on them
        if not timeline_data:
            recruitsNonTwitterData.append(params["screen_name"])
        else:
            recruitsTwitterData.append(params["screen_name"])

        recruitsListTweets[params["screen_name"]] = timeline_data

    return recruitsListTweets, recruitsNonTwitterData, recruitsTwitterData


def read_in_csv(recruitsFile):
    recruitsFile = "Twitter_Model_Data_2016.csv"

    data = []
    with open(recruitsFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                pass
            else:
                sub = []
                sub.append(row[0])
                sub.append(row[1])
                sub.append(row[4])
                sub.append(row[5])
                sub.append(row[6])
                sub.append(row[7])
                sub.append(row[8])
                sub.append(row[9])
                sub.append(row[10])
                data.append(sub)

    headers = ['Name', 'Twitter Handle', 'Miles from AA', 'First Offer', 'Last Offer', 'Official Visit',
               'Last Official Visit', 'Attended Michigan', 'In-State']
    df = pd.DataFrame(data, columns=headers)
    return df

def main():
    REQUEST_TOKEN_URL = '%s/oauth2/token' % (API_ENDPOINT)
    token = authorize(REQUEST_TOKEN_URL, consumer_key, consumer_secret)

    recruitsListTweets, recruitsNonTwitterData, recruitsTwitterData = batcher(token, globals.recruits2016)

    print "Recruits without Twitter Data: "
    for name in recruitsNonTwitterData:
        print str(name)

    print "\n"

    print "Recruits with Twitter Data: "
    for name in recruitsTwitterData:
        print str(name)

    twitterData = calcMichiganMentions(recruitsListTweets)

    df_twitter = toPandas(twitterData)
    df_twitter.head()

    df_signing = read_in_csv('Twitter_Model_Data_2016.csv')
    df_signing.head()

    df_features = pd.merge(left=df_signing[['Name', 'Twitter Handle', 'Miles from AA', 'First Offer', 'Last Offer', 'Official Visit', 'Last Official Visit', 'Attended Michigan', 'In-State']],
                           right=df_twitter[['screenName', 'michFavToAllTweetRatio', 'michTweetToAllTweetRatio', 'michOverallTweetRatio', 'michNativeRTweetRatio', 'michNativeTweetRatio']],
                           how='inner', left_on='Twitter Handle', right_on='screenName')

    print(str(df_features.head()))

    '''
    sentence1 = "State is better than Michigan. I am not going to play at #umich."
    sentence2 = "I am going to play at #Umich."
    sentence3 = "#michiganst over #Umich."
    sentence4 = "Go Blue. I am commiting to Central Michigan"
    sentence5 = "Harbaugh sucks. Go green"
    sentence6 = "#HailToTheVictors. Go green"
    sentence7 = "I am not going to state. #Umich"
    sentence8 = "Hail to the victors"
    sentence9 = "#Hail #Blue #Go #theteam #state"
    '''


if __name__ == '__main__':
    main()