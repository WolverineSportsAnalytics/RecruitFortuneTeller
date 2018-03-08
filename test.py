import base64
import json
from urllib2 import urlopen, Request

consumer_key = "hDhDiDyA7J5g36Qw9eFPnEnlS"
consumer_secret = "OofP95eIwpKxC4BV6NI24HiTPS1ScwsRxYtyV3y1dwFBUBpWfA"

API_ENDPOINT = 'https://api.twitter.com'
API_VERSION = '1.1'

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

def main():
    REQUEST_TOKEN_URL = '%s/oauth2/token' % (API_ENDPOINT)
    token = authorize(REQUEST_TOKEN_URL, consumer_key, consumer_secret)

    TIMELINE_URL = "%s/statuses/user_timeline.json?screen_name=i_williams11&count=200" % (API_ENDPOINT)
    timelineDic = {"screen_name" : "i_williams11", "count" : "200", "exclude_replies" : "true", "include_rts" : "false"}

    url_timeline = TIMELINE_URL
    for key, value in timelineDic.iteritems():
        url_timeline += key + "=" + value + "&"

    url_timeline = url_timeline[:-1]
    timeLineData = standardRequest("https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=i_williams11&count=200", token)



if __name__ == '__main__':
    main()