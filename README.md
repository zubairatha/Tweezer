# Tweezer
Tweezer is a Sentiment Analyzer of Tweets extracted using Twitter API on python only using 'Essential Access' and WITHOUT tweepy.

## Introduction
This project could have been implemented using the 'Tweepy' module but as per Twitter-API's new rules, this access is only available under the 'Elevated access'. 

For this project, I have used the default 'Essential Access' available to all Twitter Developer accounts to extract the tweets.
'Essential Access' allows access only through Twitter v2 endpoints.

Steps:
1. Create a Twitter developer account.
2. Note down essential credentials twitter prompts you to. (here, I have used only the Bearer Token)

Using the python's 'requests' library we are easily able to make the API calls provided we have proper authentication (Provided with 'Essential Access').

The tweets of a user are passed through the trained sentiment analyzer and it classifies them as positive and negative and returns a donut plot.

## Training

The sentiment analyzer was trained on the [this dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) of 1.6 million tweets using 'bag of words' technique.

## How to run

1. Inside run.py: Change the 'username' to any valid username of a public twitter account and 'tweetNum' to a number between 10 and 100.
2. Inside run.py: Add your bearerToken value to 'bearerToken' variable.
3. Run: ```python run.py```

## Output
![Result](/images/elonmusk.png)

## Limitations

'Essential Access' limits user to maximum 100 tweet extraction in one go.

## Future Plan

1. Training a model to classify the tweets into more than just the two categories used now (Positive and Negative).
2. Creating an interactive web app for ease of use and deploying it on the cloud.
