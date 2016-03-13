
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from nltk.corpus import stopwords as nltkstopwords
import utils


# this function is used to determine the accuracy of the different sklearn pipelines
# 4 of the pipelines are can be tested, one just needs to swap the line comments
# to change between them. At the top of the function is where we switch between
# our two data sets
# num defines how many tests to run and average
def pipeline_test(num):
    #posReviews = utils.loadAllTextFiles('dataset/txt_sentoken/pos/')
    #negReviews = utils.loadAllTextFiles('dataset/txt_sentoken/neg/')
    
    posReviews = utils.loadAllTextFiles('dataset/ebert_reviews/4-0/')
    posReviews += utils.loadAllTextFiles('dataset/ebert_reviews/3-5/')
    negReviews = utils.loadAllTextFiles('dataset/ebert_reviews/0-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/0-5/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-5/')
    posReviews = posReviews[:950]
    negReviews = negReviews[:950]
    
    
    import random
    import numpy as np
    #random.seed()

    accuracy = 0.0    
    percenttrain = 0.8
    posTrainCount = int(percenttrain*len(posReviews))
    negTrainCount = int(percenttrain*len(negReviews))
    
    count = 0
    for test in range(num): 
        count += 1
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
    
    
        #pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
        #pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', BernoulliNB())])
        pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', LogisticRegression())])
        #pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', SVC())])
        
        pipe = pipe.fit(train_data, Y_train)
        prediction = pipe.predict(test_data)
        
        correct = 0
        for i in range(len(actual)):
            if(actual[i] == prediction[i]):
                correct+=1
        
        acc = correct / len(actual)
        accuracy += acc
        print("{:.3f} {:.3f}".format(acc, accuracy / count))
    
    
    print("final accuracy: " + "{:.4f}".format(accuracy / num))
    
