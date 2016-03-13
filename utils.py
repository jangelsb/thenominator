from nltk import pos_tag
from nltk import FreqDist
from nltk.corpus import stopwords as nltkstopwords
from nltk.tokenize import word_tokenize
import re

# loads a text file from the path provided by filename
# returns it as a big string
def loadtxtfile(filename):
    with open(filename, 'r') as myfile:
        data = myfile.read().replace('\n','')
    return data
    
# custom tokenize function that removes punctuation as well
# string document is returned as list of string tokens
def customtokenize(document):
    ret = document.lower() #make lowercase and remove leading and trailing whitespace
    ret = re.sub('[^0-9a-zA-Z ]', ' ', ret) #only keep words and numbers
    ret = re.sub(' +', ' ',ret) #remove extra whitespace
    return ret.strip().split()  #remove leading and trailing whitespace then split on space
    
# takes a string document and tokenizes and makes a bag of words out of it
def makebagofwords_dicts(document):
    #tokens = word_tokenize(document.lower())   #nltk version not as good for our purposes
    tokens = customtokenize(document.lower())
    
    stopwords = set(nltkstopwords.words('english'))
    tokens = [i for i in tokens if not i in stopwords]
    return FreqDist(tokens)
    
# 'NN' is noun
# 'JJ is adjective
# 'VB' is verb
# 'RB' is adverb
# theres a bunch more, look up nltk.pos_tag documentation
# tokenizes the document and removes part of speech
def tokenizeAndRemovePOS(document, pos):
    tags = pos_tag(customtokenize(document)) # figures out pos for every word
    return [tag[0] for tag in tags if tag[1] != pos] 
    
# loads the part of speech list from the super preprocessed tuple lists
# path is where the list is
# filterPosList is a list of parts of speech (can be empty if no filtering wanted)
# if inclusion then only add words with these pos
#   else only remove these pos instead
def loadPosList(path, filterPosList, inclusion):
    filterPosList = set(filterPosList)
    nofilter = len(filterPosList) == 0
    with open(path) as myFile:
        posLines = myFile.read().split('\n')    
    posList = []    # list of (n reviews) list of words
    curList = []    # current list of words
    for line in posLines:
        if line == '':
            posList.append(curList)
            curList = []
        else:
            parts = line.split(' ')
            if nofilter or (inclusion and parts[1] in filterPosList) \
                or (not inclusion and parts[1] not in filterPosList):
                curList.append(parts[0])
        
    return posList
    

# builds a huge single text document containing all the words in all the documents
# each line is formatted as "[word] [part of speech]\n" 
# each document is seperated by a new line
# will load all docs from path
# make a file named by nameOfDoc
def buildPosWordList(path, nameOfDoc):
    documents = loadAllTextFiles(path)    
    with open(nameOfDoc + ".txt","w") as text_file:
        for document in documents:
            postokens = pos_tag(customtokenize(document.lower()))
            for i in postokens:
                print(i[0] + " " + i[1], file = text_file)
            print(file = text_file) 
    
# loads all txt files in a directory into a list of strings
def loadAllTextFiles(path):
    from os import listdir
    filenames = listdir(path)
    return [loadtxtfile(path + filename) for filename in filenames]


# loads and returns positive and negative word dataset (from Cornell)
# returns two sets of strings
def getUniqueGoodandBadWords():
    with open('dataset/negWords.txt', 'r') as myFile:
        badList = myFile.read().split()

    with open('dataset/posWords.txt', 'r') as myFile:
        goodList = myFile.read().split()
        
    shared = [x for x in goodList if x in badList]
    
    goodWords = set([x for x in goodList if x not in shared]) 
    badWords = set([x for x in badList if x not in shared])
    
    return goodWords, badWords
    
# prints a matrix to the screen for easy viewing
def printMatrix(matrix):
    rows = len(matrix)
    for i in range(rows):
        print(matrix[i])