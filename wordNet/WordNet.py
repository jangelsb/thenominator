import nltk
from nltk.corpus import wordnet
from nltk import FreqDist
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

def makebagofwords(document):
    #tokens = word_tokenize(document.lower())   #this one blows
    tokens = customtokenize(document.lower())

    stopwords = set(nltkstopwords.words('english'))
    tokens = [i for i in tokens if not i in stopwords]
    return FreqDist(tokens)



def bagtest():
    txt = loadtxtfile('../dataset/txt_sentoken/neg/cv000_29416.txt')
    bag = makebagofwords(txt)
    return bag

def getBadList():
    with open('../negWords.txt', 'r') as myFile:
        myData = myFile.read().split()
    return myData

def getGoodList():
    with open('../posWords.txt', 'r') as myFile:
        myData = myFile.read().split()
    return myData

def sharedWords():
    pos = getGoodList()
    neg = getBadList()
    bla = [x for x in pos if x in neg]
    return bla

def posminusneg(review):
    badWords = getBadList()
    goodWords = getGoodList()
    sharWar = sharedWords()
    file = loadtxtfile(review)
    wordsInReview = customtokenize(file)

    #get rid of shared words in both list
    badWords = [x for x in badWords if x not in sharWar]
    goodWords = [x for x in goodWords if x not in sharWar]

    posWordsInReview = [x for x in wordsInReview if x in goodWords]
    numPos = len(posWordsInReview)

    negWordsInReview = [x for x in wordsInReview if x in badWords]
    numNeg = len(negWordsInReview)

    score = numPos - numNeg

    print("Words in review")
    print(wordsInReview)

    print("here are the good words")
    print(goodWords)

    print("here are the bad words")
    print(badWords)

    print("here are pos words in review")
    print(posWordsInReview)

    print("here are neg words in review")
    print(negWordsInReview)

    if score < 0:
        print("review is negative")
    else:
        print("review is positive")


