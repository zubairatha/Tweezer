import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import requests
import collections
import matplotlib.pyplot as plt

dfr=pd.read_csv('final.csv')

username="elonmusk"
tweetNum=15
bearerToken='Enter your Bearer Token here'

class Sentiment: # Using Enums
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self,text,score):
        self.text=text
        self.score=score
        self.sentiment=self.get_sentiment()
    def get_sentiment(self):
        if self.score==0:
            return Sentiment.NEGATIVE
        elif self.score==4:
            return Sentiment.POSITIVE

            
class ReviewContainer:
    def __init__(self,reviews):
        self.reviews=reviews
    
    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
  
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)


reviews = []
for ind in dfr.index:
    reviews.append(Review(dfr['tweets'][ind],dfr['sentiment'][ind]))

from sklearn.model_selection import train_test_split
training, test = train_test_split(reviews,test_size=0.33,random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()


vectorizer = TfidfVectorizer()

train_x_vectors = vectorizer.fit_transform(np.array(train_x))
test_x_vectors = vectorizer.transform(np.array(test_x))

with open('./models/tweet sentiment classifier-1.pkl','rb') as f:
    loaded_clf = pickle.load(f)


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


response = requests.get(f'https://api.twitter.com/2/tweets/search/recent?query=from:{username}&max_results={tweetNum}', auth=BearerAuth(bearerToken))
response_json=response.json()
n=len(response_json['data'])
l=[]
for i in range(n):
    a=response_json['data'][i]['text']
    t=" ".join(filter(lambda x:x[0]!='@', a.split()))
    l.append(t)




st_vectors=vectorizer.transform(l)
results=loaded_clf.predict(st_vectors)

frequency = collections.Counter(results)
freq = dict(frequency)


data = [freq['POSITIVE']*100/n,freq['NEGATIVE']*100/n]
plt.pie(data, labels = [f'POSITIVE {data[0]}%', f'NEGATIVE {data[1]}%'])
circle = plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title(f"{username}'s tweet analysis")
plt.show()