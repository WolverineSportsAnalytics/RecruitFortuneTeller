from __future__ import print_function

import base64
import json
import globals
import time
from urllib import urlencode
from urllib2 import urlopen, Request
import pandas as pd
import csv
import mysql.connector
import datetime as dt
import pytz

consumer_key = "hDhDiDyA7J5g36Qw9eFPnEnlS"
consumer_secret = "OofP95eIwpKxC4BV6NI24HiTPS1ScwsRxYtyV3y1dwFBUBpWfA"

API_ENDPOINT = 'https://api.twitter.com'
API_VERSION = '1.1'

hashTags = ["michdet",
             "uofm",
             "goblue",
             "go blue",
             "umich",
             "wolv",
             "wolverines",
             "wolverine",
             "michigan",
             "hail",
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
             "\u303d\ufe0f",
            "\u2744\ufe0f"]

negativeWords = ["state", " st.", "central", "eastern", "gogreen", "green", "eastern", "western", "central", "emu",
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

def isMichigan(word_string):
    word_string = word_string.lower().encode('utf8')
    for tag in hashTags:  # for word in hashTags
        if word_string.find(
                tag) != -1:  # if word is in tagString
            return 1  # return 1

    return 0  # else return 0

def calcMichiganMentions(data):
    # check each word in data and if it matches
    twitterData = []
    for screenName, tweetsObject in data.iteritems():
        numTweetsAnalyzing = len(tweetsObject)
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

        print("Analyzing Tweets for: " + screenName)

        for tweet in tweetsObject:
            if tweet[4]:
                retweeted = True
            else:
                retweeted = False

            if retweeted:
                nativeRetweets += 1
            else:
                nativeTweets += 1
                favorites = tweet[6]
                totFavorites += favorites

                retweets = tweet[7]
                totRetweets += retweets

            tweetText = tweet[3]
            hashtags = tweet[5]

            if isMichigan(tweetText) or isMichigan(hashtags):
                michiganTweets += 1
                if retweeted:
                    print(tweetText, end='')
                    print("")
                    nativeMichiganRetweets += 1
                else:
                    print("Native Tweet: " + tweetText, end='')
                    print("")
                    nativeMichiganTweets += 1
                    favoritesMichFunc = tweet[6]
                    retweetsMichFunc = tweet[7]
                    michFavorites += favoritesMichFunc
                    michRetweets += retweetsMichFunc

        michRetweetsRatio = 0 if nativeMichiganTweets == 0 else (float(michRetweets) / float(nativeMichiganTweets))
        michFavoritesRatio = 0 if nativeMichiganTweets == 0 else (float(michFavorites) / float(nativeMichiganTweets))

        tweetRatio = 0 if nativeTweets == 0 else (float(totRetweets) / float(nativeTweets))
        favoriteRatio = 0 if nativeTweets == 0 else (float(totFavorites) / float(nativeTweets))

        michFavToAllTweetRatio = 0 if favoriteRatio == 0 else (float(michFavoritesRatio) / float(favoriteRatio))
        michTweetToAllTweetRatio = 0 if tweetRatio == 0 else (float(michRetweetsRatio) / float(tweetRatio))

        michOverallTweetRatio = 0 if numTweetsAnalyzing == 0 else (float(michiganTweets) / float(numTweetsAnalyzing))
        michNativeRTweetRatio = 0 if nativeRetweets == 0 else (float(nativeMichiganRetweets) / float(nativeRetweets))
        michNativeTweetRatio = 0 if nativeTweets == 0 else (float(nativeMichiganTweets) / float(nativeTweets))

        screenNameTwitterData['tweet_metrics']['michFavToAllTweetRatio'] = michFavToAllTweetRatio
        screenNameTwitterData['tweet_metrics']['michTweetToAllTweetRatio'] = michTweetToAllTweetRatio

        screenNameTwitterData['tweet_metrics']['michOverallTweetRatio'] = michOverallTweetRatio
        screenNameTwitterData['tweet_metrics']['michNativeRTweetRatio'] = michNativeRTweetRatio
        screenNameTwitterData['tweet_metrics']['michNativeTweetRatio'] = michNativeTweetRatio

        twitterData.append(screenNameTwitterData)

    return twitterData

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

def batcher(token, recruits, year, cursor, cnx):
    base_url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    recruitsListTweets = {}

    recruitsTwitterData = []
    recruitsNonTwitterData = []

    maxID = 0
    lowerID = 0

    if year == 2016:
        maxID = globals.nsd2016maxID
        lowerID = globals.nsd2016lowerID
    if year == 2017:
        maxID = globals.nsd2017maxID
        lowerID = globals.nsd2017lowerID

    requests = 0

    print("Number of recruits: " + str(len(recruits)))

    f = '%Y-%m-%d %H:%M:%S'
    recruit_info_q = "SELECT Date_Committed FROM recruit_info WHERE Twitter_Handle = %s"

    start_time = time.time()
    for counter, recruit in enumerate(recruits):
        # find last tweet since national signing day

        last_first_element = 0
        last_last_element = 0

        params = {"screen_name": recruit, "count": str(200), "include_rts": 1, "max_id": str(maxID), "since_id": str(lowerID)}
        timeline_data = []

        # check if recruit is in database
        recruit_check = "SELECT * FROM recruits WHERE twitterScreenName = %s"
        recruit_check_data = (recruit,)

        cursor.execute(recruit_check, recruit_check_data)
        recruit_check_res = cursor.fetchall()

        get_recruit_tweets = "SELECT * FROM tweets WHERE twitterScreenName = %s"
        get_recruit_tweets_data = (recruit,)

        if cursor.rowcount > 0:
            if recruit_check_res[0][2]:
                cursor.execute(get_recruit_tweets, get_recruit_tweets_data)

                print("Getting tweets from database for: " + str(recruit))

                recruitsListTweets[recruit] = cursor.fetchall()
        else:
            # get recruit info
            recruit_info_q_d = (recruit,)
            cursor.execute(recruit_info_q, recruit_info_q_d)

            recruit_info_d = cursor.fetchall()

            sql_commit_date = recruit_info_d[0][0]

            while True:
                url = (base_url + "?" + urlencode(params))

                try:
                    tweet_data = standardRequest(url, token)
                    requests += 1
                    print("Request Number: " + str(requests) + " for: " + str(recruit) + " number " + str(counter))
                except:
                    break

                if not tweet_data:
                    break

                timeline_data.append(tweet_data)

                first_element = tweet_data[0]['id']
                last_element = tweet_data[-1]['id']

                newMaxId = last_element
                if (last_element < lowerID or (first_element == last_first_element and last_element == last_last_element)):
                    del timeline_data[-1]
                    break

                params["max_id"] = str(newMaxId)
                last_first_element = first_element
                last_last_element = last_element

                if requests == 900:
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    api_sleep_time = 900 - time_elapsed

                    if api_sleep_time > 0:
                        print("Stalling for " + str(api_sleep_time) + " seconds for twitter api rate limit")
                        time.sleep(api_sleep_time)

                    start_time = time.time()
                    requests = 0

            insert_batch_tweets = "INSERT INTO tweets (twitterScreenName, tweetID, tweet, retweet, hashtags, numFavorites, numRetweets) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            insert_recruit = "INSERT INTO recruits (twitterScreenName, tweetData) VALUES (%s, %s)"
            # check to see if any tweet data on them
            # store tweet data
            tweet_list = []

            recruitsListTweets[recruit] = []

            inserted_recruit = False
            if not timeline_data:
                recruitsNonTwitterData.append(params["screen_name"])
                insert_recruit_data = (recruit,0)
                cursor.execute(insert_recruit, insert_recruit_data)
                inserted_recruit = True
            else:
                recruitsTwitterData.append(params["screen_name"])

                for counter, tweet_batch in enumerate(timeline_data):
                    for counter_tweet_batch, tweet in enumerate(tweet_batch):
                        if counter != 0 and counter_tweet_batch == 0:
                            continue
                        else:
                            ts = dt.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

                            if ts <= sql_commit_date:
                                retweeted = 0
                                if "retweeted_status" in tweet:
                                    retweeted = 1

                                tweet_text = tweet['text']
                                hashtags = tweet['entities']['hashtags']
                                hashtag_string = ''
                                for taghash in hashtags:
                                    hashtag_string += taghash['text'].encode('utf-8')
                                    hashtag_string += " "

                                tweet_id = tweet['id_str']

                                num_favorites = -1
                                num_retweets = -1

                                if not retweeted:
                                    num_retweets = calcRetweets(tweet)
                                    num_favorites = calcFavorites(tweet)

                                tweet_database_data = (recruit, tweet_id, tweet_text, retweeted, hashtag_string, num_favorites,
                                                       num_retweets)
                                tweet_list.append(tweet_database_data)

            # have list full of tweeter data, now store in
            if len(tweet_list) == 0 and inserted_recruit is False:
                insert_recruit_data = (recruit,0)
                cursor.execute(insert_recruit, insert_recruit_data)
            elif inserted_recruit is False:
                recruitsListTweets[recruit] = tweet_list
                cursor.executemany(insert_batch_tweets, tweet_list)
                insert_recruit_data = (recruit, 1)
                cursor.execute(insert_recruit, insert_recruit_data)

            cnx.commit()

    return recruitsListTweets, recruitsNonTwitterData, recruitsTwitterData


def read_in_csv(recruitsFile):
    data = []
    with open(recruitsFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            sub = []
            sub.append(row[0])
            sub.append(row[1])
            sub.append(int(row[4]))
            sub.append(int(row[5]))
            sub.append(int(row[6]))
            sub.append(int(row[7]))
            sub.append(int(row[8]))
            sub.append(int(row[9]))
            sub.append(int(row[10]))
            data.append(sub)

    headers = ['Name', 'Twitter Handle', 'Miles from AA', 'First Offer', 'Last Offer', 'OfficialVisit',
               'Last Official Visit', 'Attended Michigan', 'In-State']
    df = pd.DataFrame(data, columns=headers)
    return df

def main(year):
    cnx = mysql.connector.connect(user=globals.databaseUser,
                                  host=globals.databaseHost,
                                  database=globals.databaseName,
                                  password=globals.databasePassword)
    cursor = cnx.cursor()

    REQUEST_TOKEN_URL = '%s/oauth2/token' % (API_ENDPOINT)
    token = authorize(REQUEST_TOKEN_URL, consumer_key, consumer_secret)

    analyzeScreenNames = []

    if year == 2016:
        analyzeScreenNames = globals.recruits2016
    if year == 2017:
        analyzeScreenNames = globals.recruits2017

    recruitsListTweets, recruitsNonTwitterData, recruitsTwitterData = batcher(token, analyzeScreenNames, year, cursor, cnx)

    print("Recruits without Twitter Data: ")
    for name in recruitsNonTwitterData:
        print(str(name))

    print("\n")

    print("Recruits with Twitter Data: ")
    for name in recruitsTwitterData:
        print(str(name))

    twitterData = calcMichiganMentions(recruitsListTweets)

    df_twitter = toPandas(twitterData)
    df_twitter.head()

    twitter_csv_data_filename = "Twitter_Model_Data_" + str(year) + ".csv"

    df_signing = read_in_csv(twitter_csv_data_filename)
    df_signing.head()

    df_features = pd.merge(left=df_signing[['Name', 'Twitter Handle', 'Miles from AA', 'First Offer', 'Last Offer', 'OfficialVisit', 'Last Official Visit', 'Attended Michigan', 'In-State']],
                           right=df_twitter[['screenName', 'michFavToAllTweetRatio', 'michTweetToAllTweetRatio', 'michOverallTweetRatio', 'michNativeRTweetRatio', 'michNativeTweetRatio']],
                           how='inner', left_on='Twitter Handle', right_on='screenName')

    print(str(df_features.head()))

    return df_features

if __name__ == '__main__':
    main(2016)