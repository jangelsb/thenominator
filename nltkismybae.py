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
    txt = loadtxtfile('dataset/txt_sentoken/neg/cv000_29416.txt')
    bag = makebagofwords(txt)
    return bag


def wordnettesting():
    for synset in wn.synsets('dog'):
        for lemma in synset.lemmas():
            print(lemma.name())