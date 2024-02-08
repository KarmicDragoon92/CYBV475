'''
Simple Classifier Example
Applying KNeighborsClassifier

Using Sentiment Analysis as an example
Use Case

'''
import re
from collections import Counter

print("Simple Classifer Model - Professor Hosmer Feb 2022")
print("Loading ML Libraries ..... Please Wait ...")
import pandas as pd
from nltk import word_tokenize, pos_tag, FreqDist, trigrams

# Setup Prettytable for results
from prettytable import PrettyTable

# Machine Learning Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

le = preprocessing.LabelEncoder()

#Psuedo Constants
DEBUG = False   # set DEBUG = True, for debug messages

#Psuedo Lookup for positive and negative sentiments
SENTIMENT = {1:"Yes", 0:"No"}


# Set Panda Options
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)   
pd.set_option('display.width', 2000)   

# Create simplified dataframe for globalWarming.csv
# Just two columns, existence, tweet
print("\nCreating dataframe from globalWarming.csv ...")
df = pd.read_csv('globalWarming.csv', encoding='cp1252')
df = df.rename(columns={'existence': 'Sentiment', 'tweet': 'Tweet'})
df = df[['Sentiment','Tweet']]

def scrubTweet(twt):
    
    twt = re.sub("[^a-zA-Z.?!,]", ' ', twt)
    return twt

def genPosNegTriGrams(dfCheck):
    '''
    Process each row in the training dataframe
    '''
    posTrigrams = set()
    negTrigrams = set()
    
    for row in dfCheck.itertuples():
        if row.Sentiment != 'Yes' and row.Sentiment != 'No':
            continue        
        twt = twt = scrubTweet(row.Tweet)
        
        triGrams = list(trigrams(twt.split()))
        
        for tri in triGrams:
            if row.Sentiment == 'Yes':
                posTrigrams.add(tri)
            else:
                negTrigrams.add(tri)
    
    posTrigrams -= negTrigrams
    negTrigrams -= posTrigrams
    
    return posTrigrams, negTrigrams

def getFeatures(twt, posG, negG):
    ''' Model for feature extraction
        This is just and example not the real features
        that will be included.
    '''
    twt = scrubTweet(twt)
    
    posTriCount = 0
    negTriCount = 0
    twtGrams = list(trigrams(twt.split()))
    
    for eachGram in twtGrams:
        if eachGram in posG:
            posTriCount += 1
        if eachGram in negG:
            negTriCount += 1
    
    tokenizedWords = word_tokenize(twt)
    tokenWords = [w for w in tokenizedWords]
    wordCnt = len(tokenWords)
    
    pos_tagged = pos_tag(tokenWords)
    counts = Counter(tag for word,tag in pos_tagged)
    
    adjFreq = round(counts['JJ']/wordCnt,  4)    # Adjectives
    cmpFreq = round(counts['JJR']/wordCnt, 4)   # Adjectives Comparative
    supFreq = round(counts['JJS']/wordCnt, 4)   # Adjectives Superlative    

    return [adjFreq, cmpFreq, supFreq, posTriCount, negTriCount]

def main():
    
    featureList   = []  # List of features for each sample
    sentimentList = []  # Corresponding sentiment
    
    dfTrain, dfTest = train_test_split(df, test_size = 0.3)
    
    '''
    Get pos and neg trigrams sets from training dataset
    '''
    posTrigrams, negTrigrams = genPosNegTriGrams(dfTrain)
        
    print("Processing Training Dataframe ...")
    for row in dfTrain.itertuples():
        # only process rows that are either Yes or No Sentiment Values
        if row.Sentiment != 'Yes' and row.Sentiment != 'No':
            continue
        
        sentimentList.append(row.Sentiment)  # update the sentiment list
        features = getFeatures(row.Tweet, posTrigrams, negTrigrams)
        featureList.append(features)  # Update the corresponding feature list    

    # encode the Sentimen either 0 = No or 1 = Yes
    encodedSentiment = le.fit_transform(sentimentList)
    
    # Create a K Nearest Neighbor Classifier
    print("Creating Nearest Neighbor Classifer Model ...")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(featureList, encodedSentiment)
    
    incorrect = 0
    correct   = 0
    totalExamined    = 0
    tbl = PrettyTable(["Known", "Predicted", "Tweet"])

    # Now test the modeul using the dfTrain dataframe
    print("\nApplying the Model to the data ...")
    for row in df.itertuples():
        
        # only process rows that are either Yes or No Sentiment Values
        if row.Sentiment != 'Yes' and row.Sentiment != 'No':
            continue
        
        totalExamined += 1
        # get the tweet and get the features
        tstTweet = row.Tweet
        known    = row.Sentiment
        
        features = getFeatures(tstTweet, posTrigrams, negTrigrams)
        
        # use the features to predict the result
        prediction = model.predict([features])
        predicted  = SENTIMENT[prediction[0]]
        if predicted == known:
            correct   += 1
        else:
            incorrect += 1
            
        tbl.add_row([known, predicted, scrubTweet(tstTweet)])
        
    tbl.align='l'
    print(tbl.get_string(sortby="Known"))
    
    print("\nSummary Results")
     
    print("Total Examined:        ", totalExamined)
    print("Correctly Identified:  ", correct)
    print("Incorrectly Identified ", incorrect)
    print("Overall Accuracy:      ", (correct / totalExamined)* 100.0)

if __name__ == '__main__':
    main()
    print("\nScript End")