from nltk import FreqDist
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltkstopwords
from nltk.corpus import wordnet as wn
import re

def loadtxtfile(filename):
    with open(filename, 'r') as myfile:
        data = myfile.read().replace('\n','')
    return data

def customtokenize(document):
    ret = document.lower() #make lowercase and remove leading and trailing whitespace
    ret = re.sub('[^0-9a-zA-Z ]', ' ', ret) #only keep words and numbers
    ret = re.sub(' +', ' ',ret) #remove extra whitespace
    return ret.strip().split()  #remove leading and trailing whitespace then split on space

def makebagofwords_dicts(document):
    #tokens = word_tokenize(document.lower())   #this one blows
    tokens = customtokenize(document.lower())
    
    stopwords = set(nltkstopwords.words('english'))
    tokens = [i for i in tokens if not i in stopwords]
    return FreqDist(tokens)

def loadReviews():
    from os import listdir
    pospath = 'dataset/txt_sentoken/pos/'
    negpath = 'dataset/txt_sentoken/neg/'
    posfilenames = listdir(pospath)
    negfilenames = listdir(negpath)
    posreviews = [loadtxtfile(pospath+filename) for filename in posfilenames]
    negreviews = [loadtxtfile(negpath+filename) for filename in negfilenames]
    return posreviews, negreviews

def pipeline_comparison():
    posreviews, negreviews = loadReviews()
    
    import random
    import numpy as np
    #random.seed()

    accuracy = 0.0    
    percenttrain = 0.8
    posTrainCount = int(percenttrain*len(posreviews))
    negTrainCount = int(percenttrain*len(negreviews))
    
    for test in range(10):    
        random.shuffle(posreviews)
        random.shuffle(negreviews)                

        train_tups = [(r, 1) for r in posreviews[:posTrainCount]] + [(r, 0) for r in negreviews[:negTrainCount]]
        random.shuffle(train_tups)
        train_data = [tup[0] for tup in train_tups]
        Y_train = np.array([tup[1] for tup in train_tups])
        
        test_tups = [(r, 1) for r in posreviews[posTrainCount:]] + [(r, 0) for r in negreviews[negTrainCount:]]
        random.shuffle(test_tups)
        test_data = [tup[0] for tup in test_tups]
        actual = np.array([tup[1] for tup in test_tups])
    
        #80.70  MULTINOMIAL NAIVE BAYES
        #79.25  BERNOULLI NAIVE BAYES
        #83.80  LOGISTIC REGRESSION          <--- winner weeeee
        #next up try cross validation (split data into 5 chunks and swap around which chunk is test set)
        #use same shuffled list for each different pipeline
    
        #pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
        #pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', BernoulliNB())])
        pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', LogisticRegression())])
        
        pipe = pipe.fit(train_data, Y_train)
        prediction = pipe.predict(test_data)
        
        correct = 0
        for i in range(len(actual)):
            if(actual[i] == prediction[i]):
                correct+=1
        
        print(correct / len(actual))
        accuracy += correct / len(actual)
    
    
    print("final accuracy: " + "{:.4f}".format(accuracy / 10.0))
    



# filth below    
    
    
def bagtest():
    txt = loadtxtfile('dataset/txt_sentoken/neg/cv000_29416.txt')
    #bag = makebagofwords(txt)
    bag = vectorizesparsematrix(txt)
    
    return bag
    
def matrixtest():
    txt = loadtxtfile('dataset/txt_sentoken/neg/cv000_29416.txt')
    counts = vectorizesparsematrix(txt)
    return counts.sum(axis = 1).mean()

# fully confused    
#def fulltest():
    
    
    

def wordnettesting():
    for synset in wn.synsets('dog'):
        for lemma in synset.lemmas():
            print(lemma.name())
            
            
def vectorizesparsematrix(document):
    tokens = customtokenize(document.lower())
    
    #stopwords = set(nltkstopwords.words('english'))
    #tokens = [i for i in tokens if not i in stopwords]
    #return CountVectorizer(tokens)
    
    stopper = CountVectorizer(stop_words=nltkstopwords.words('english'))
    xtrainCounts = stopper.fit_transform(tokens);
    return xtrainCounts