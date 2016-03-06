
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from nltk.corpus import stopwords as nltkstopwords
from nltk.corpus import wordnet as wn
import utils


def pipeline_comparison():
    posReviews = utils.loadAllTextFiles('dataset/txt_sentoken/pos/')
    negReviews = utils.loadAllTextFiles('dataset/txt_sentoken/neg/')
    
    import random
    import numpy as np
    #random.seed()

    accuracy = 0.0    
    percenttrain = 0.8
    posTrainCount = int(percenttrain*len(posReviews))
    negTrainCount = int(percenttrain*len(negReviews))
    
    for test in range(10):    
        random.shuffle(posReviews)
        random.shuffle(negReviews)                

        train_tups = [(r, 1) for r in posReviews[:posTrainCount]] + [(r, 0) for r in negReviews[:negTrainCount]]
        random.shuffle(train_tups)
        train_data = [tup[0] for tup in train_tups]
        Y_train = np.array([tup[1] for tup in train_tups])
        
        test_tups = [(r, 1) for r in posReviews[posTrainCount:]] + [(r, 0) for r in negReviews[negTrainCount:]]
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
# some old tests in previous commits as well

#good example showing what wordnet offers
#more about relations between words
#http://wordnetweb.princeton.edu/perl/webwn
def testWordNet(word):
    for synset in wn.synsets(word):
        print(synset)
        for lemma in synset.lemmas():
            print(lemma.name())
