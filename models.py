
import os
import json

from datetime import datetime

TWEET_ATTRS = {
    "id_str": "id",
    "created_at": "created_at",
    "text": "text",
    "lang": "lang",
    "retweet_count": "retweet_count",
}
USER_ATTRS = {"id_str": "userid", "name": "username", "screen_name": "userscreenname"}


def _strip_tweet(dict_tweet):
    tweet = {}
    for k, v in TWEET_ATTRS.items():
        if k in dict_tweet:
            tweet[v] = dict_tweet[k]

    for k, v in USER_ATTRS.items():
        if k in dict_tweet["user"]:
            tweet[v] = dict_tweet["user"][k]

    retweet = None
    if "retweeted_status" in dict_tweet:
        retweet, _ = _strip_tweet(dict_tweet["retweeted_status"])

    return tweet, retweet



class Tweet: 

    def __init__(self, id): 
        self._id = id
        self._user = None
        self._created_at = None
        self._text = None
        self._retweet_of = None
        self._retweeted_by = []

    @property
    def id(self):
        return self._id

    @property
    def created_at(self):
        return self._created_at

    @created_at.setter
    def created_at(self, value):
        self._created_at = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    @property
    def is_retweet(self):
        return self._retweet_of is not None

    @property
    def retweet_of(self):
        return self._retweet_of

    @retweet_of.setter
    def retweet_of(self, value):
        self._retweet_of = value

    @property
    def retweeted_by(self):
        return self._retweeted_by

    def __repr__(self):
        return "Tweet(%r)" % self._id


class User: 

    def __init__(self, id):
        self._id = id
        self._followers = set()
        self._following = set()
        self._screenname = None

    @property
    def id(self):
        return self._id
    
    @property
    def followers(self):
        return self._followers

    @property
    def following(self) :
        return self._following

    @property
    def screenname(self):
        return self._screenname

    @screenname.setter
    def screenname(self, value):
        self._screenname = value

    @property
    def popularity(self):
        return len(self._followers)

    def __repr__(self):
        return "User(%r)" % self._id
    



class Dataset:
    """A dataset model.
    """
    def __init__(self):
        self._users_by_id = {}
        self._users_by_username = {}
        self._tweets_by_id = {}

    def load_users_and_followers(self, path): 
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:

                    user_dict = json.load(json_file)
                    user_id = str(user_dict['user_id'])

                    if user_id in self._users_by_id: 
                        user = self._users_by_id[user_id]
                    else: 
                        user = User(user_id)    
                        self._users_by_id[user_id] = user
                    
                    for follower_id in user_dict['followers']:
                        user.followers.add(str(follower_id))


    def load_tweets(self, path):
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    full_tweet = json.load(json_file)
                    tweet_dict, retweet_dict = _strip_tweet(full_tweet)

                    tweet = self._update_tweet(tweet_dict)

                    if retweet_dict is not None:
                        retweet = self._update_tweet(retweet_dict)
                        tweet.retweet_of = retweet
                        retweet.retweeted_by.append(tweet)

    @property
    def users_by_id(self):
        return self._users_by_id

    @property
    def tweets_by_id(self):
        return self._tweets_by_id

    @property
    def users_by_username(self):
        return self._users_by_username

    def _update_tweet(self, tweet_dict):
        tweet_id = str(tweet_dict['id'])
        if tweet_id in self._tweets_by_id: 
            tweet = self._tweets_by_id[tweet_id]
        else:
            tweet = Tweet(tweet_id)
            self._tweets_by_id[tweet_id] = tweet

        created_at_str = tweet_dict['created_at']
        tweet.created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y") 
        tweet.text = tweet_dict['text']

        userid = str(tweet_dict['userid'])
        if userid in self._users_by_id:
            user = self._users_by_id[userid]
        else: 
            user = User(userid)
            self._users_by_id[userid] = user

        tweet.user = user

        screenname = tweet_dict.get('userscreenname', None)
        if screenname is not None: 
            user.screenname = screenname
            self._users_by_username[screenname] = user

        return tweet        

    def __repr__(self):
        return "Dataset(%r, %r, %r)" % (self._users_by_id, self._users_by_username, self._tweets_by_id)