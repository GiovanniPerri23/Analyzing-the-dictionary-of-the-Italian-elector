import tweepy
import configparser
from textblob import TextBlob
import pandas as pd
import requests

"""Tweet searcher for SNA project. First step: read configs."""

config=configparser.ConfigParser()
config.read("config.ini")

api_key=config['Twitter']['api_key']
api_key_secret=config['Twitter']['api_key_secret']

access_token=config['Twitter']['access_token']
access_token_secret=config['Twitter']['access_token_secret']
bearer_token=config['Twitter']['bearer_token']

"Authentication for Twitter API"

auth = tweepy.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token, access_token_secret)
client=tweepy.Client(bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_key_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    return_type = requests.Response,wait_on_rate_limit=True)

api = tweepy.API(auth)

"""Search Tweets using keywords and geocode in tweepy.Cursor"""

#geo=" 41.9109,12.4818,300km"#geocode=geo Roma
#geo=" 45.4654219,9.1859243,300km"#geocode=geo Milano
geo=" 38.116667,13.366667,300km"#geocode=geo Palermo

keywords = "elezioni"#-filter:retweets"
limit=10000
tweets= tweepy.Cursor(api.search_tweets, q=keywords,count=200,geocode=geo,tweet_mode='extended').items(limit)

"""Subjectivity and polarity"""

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity



"""" Create DataFrame """

User= []
Id=[]
Tweets=[]
Data=[]
Likes=[]
Retweet=[]
Country=[]
Subjectivity=[]
Polarity=[]

for tweet in tweets:
    User.append(tweet.user.screen_name)
    Id.append(tweet.id_str)
    Tweets.append(tweet.full_text)
    Data.append(tweet.created_at)
    Likes.append(tweet.favorite_count)
    Retweet.append(tweet.retweet_count)
    Country.append(tweet.author.location)


df = pd.DataFrame({'User':User,'Id':Id,'Tweets':Tweets,'Likes':Likes,'Retweet':Retweet,'Country':Country,'Data':Data})
df['Subjectivity']=df['Tweets'].apply(getSubjectivity)
df['Polarity']=df['Tweets'].apply(getPolarity)

"""Save as .csv and .xlsx"""

df.to_csv('tweets.csv')
read_file = pd.read_csv (r'tweets.csv')
read_file.to_excel (r'tweets.xlsx', index = None, header=True)
