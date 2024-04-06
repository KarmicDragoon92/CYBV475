"""
Machine Learning Spam Detection Assignment
Kaleb Sundstrom
March 2024
"""

# import libraries

import re
from collections import Counter
import string

import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Sanitize the TEXT
def sanitizeText(TEXT):
    TEXT = re.sub("[^a-zA-Z0-9.?!,]", " ", TEXT)
    return TEXT

def punctuation_ratio(TEXT):
    punctuation_chars = set(string.punctuation)

    punctuation_count = sum(1 for char in TEXT if char in punctuation_chars)

    total_chars = len(TEXT)
    if total_chars == 0:
        return 0
    return punctuation_count / total_chars

def get_sentiment(TEXT):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(TEXT)
    return sentiment_scores['compound']

def average_word_length(TEXT):
    words = TEXT.split()
    total_chars = sum(len(word) for word in words)
    total_words = len(words)
    if total_words == 0:
        return 0
    return total_chars / total_words

def vectorize(TEXT):
    vectorizer = HashingVectorizer(n_features=2)
    TEXT = [TEXT]
    vector = vectorizer.transform(TEXT)
    print(vector.shape)
    print(vector.toarray())
    return vector.toarray()

def getFeatures(TEXT):

    TEXT = sanitizeText(TEXT)

    punct_ratio = punctuation_ratio(TEXT)
    sentiment_score = get_sentiment(TEXT)
    avg_word_length = average_word_length(TEXT)
    otherFeatures = vectorize(TEXT)

    return [punct_ratio, sentiment_score, avg_word_length]


def main():
    SENTIMENT = {1: "ham", 0: 'spam'}

    featureList = []
    mailType = []

    df = pd.read_csv('SMSSpamCollection.csv', encoding='ISO-8859-1')
    dfTrain, dfTest = train_test_split(df, test_size = 0.2)

    for row in dfTrain.itertuples():
        mailType.append(row.TYPE)
        features = getFeatures(row.TEXT)
        featureList.append(features)

    eoncodedMailType = preprocessing.LabelEncoder().fit_transform(mailType)
    print(eoncodedMailType)
    model = KNeighborsClassifier(n_neighbors=5)

    #model = GridSearchCV(estimator=pipe, param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, cv=3)
    model.fit(featureList, eoncodedMailType)

    incorrect = 0
    correct = 0
    totalExamined = 0

    for row in dfTrain.itertuples():
        totalExamined += 1
        tstText = row.TEXT
        known = row.TYPE

        features = getFeatures(tstText)
        prediction = model.predict([features])
        predicted = SENTIMENT[prediction[0]]
        if predicted == known:
            correct +=1
        else:
            incorrect +=1

    print("Total Examined: ", totalExamined)
    print('Correct: ', correct)
    print('Incorrect: ', incorrect)
    print('Accuracy: ', (correct / totalExamined) * 100.0)


main()
