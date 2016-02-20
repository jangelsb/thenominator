from nltk import FreqDist
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
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
    negfilenames = listdir(negpath);
    posreviews = [loadtxtfile(pospath+filename) for filename in posfilenames]
    negreviews = [loadtxtfile(negpath+filename) for filename in negfilenames]
    return posreviews, negreviews

def pipeline_multinomialNB():
    posreviews, negreviews = loadReviews()
    
    print(posreviews[0])
    print(negreviews[1])
    
    return
    
    train_data = 0  # list of document strings in training data
    Y_train = 0     # list of ints as positive or negative review
    test_data = 0   # 
    pipe = Pipeline([('vect', CountVectorizer(stop_words=nltkstopwords.words('english'))),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
                     
    pipe = pipe.fit(train_data, Y_train)
    return pipe.predict(test_data)
    
    
    
    
    
    
    
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