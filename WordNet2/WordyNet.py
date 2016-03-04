from nltk import pos_tag
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
    
# 'NN' is noun
# 'JJ is adjective
# 'VB' is verb
# 'RB' is adverb
# theres a bunch more, look up nltk.pos_tag documentation
def tokenizeAndRemovePOS(document, pos):
    tags = pos_tag(customtokenize(document)) # figures out pos for every word
    return [tag[0] for tag in tags if tag[1] != pos] 

def loadReviews():
    from os import listdir
    pospath = '../dataset/txt_sentoken/pos/'
    negpath = '../dataset/txt_sentoken/neg/'
    posfilenames = listdir(pospath)
    negfilenames = listdir(negpath)
    posreviews = [loadtxtfile(pospath+filename) for filename in posfilenames]
    negreviews = [loadtxtfile(negpath+filename) for filename in negfilenames]

    return posreviews, negreviews
    
# loads the positive and negative part of speech lists
# into a list of 1000 lists of strings for easy separation into training and testing sets
def loadPosLists():
    with open('../dataset/txt_sentoken/posposlist.txt') as myFile:
        posLines = myFile.read().split('\n')
    
    
    posList = []    # list of 1000 list of words
    curList = []    # current list of words
    for line in posLines:
        if line == '':
            posList.append(curList)
            curList = []
        else:
            parts = line.split(' ')
            if parts[1] == 'JJ' or parts[1] == 'RB':
                curList.append(parts[0])
                


    with open('../dataset/txt_sentoken/negposlist.txt') as myFile:
        negLines = myFile.read().split('\n')
    
    
    negList = []    # list of 1000 list of words
    curList = []    # current list of words
    for line in negLines:
        if line == '':
            negList.append(curList)
            curList = []
        else:
            parts = line.split(' ')
            #if parts[1] == 'JJ' or parts[1] == 'RB':
            if parts[1] == 'JJ':
                curList.append(parts[0])            
        
    return posList, negList
    

def posminusneg(review, goodWords, badWords):
    #wordsInReview = customtokenize(review) #69.90%
    #wordsInReview = tokenizeAndRemovePOS(review, 'JJ') #64.35%
    #wordsInReview = tokenizeAndRemovePOS(review, 'NN') #69.45% was ~81% after positive reviews
    #wordsInReview = tokenizeAndRemovePOS(review, 'VB') #69.85%
    #wordsInReview = tokenizeAndRemovePOS(review, 'RB') #68.75%

    score = 0
    for x in wordsInReview:
        if x in goodWords:
            score += 1
        if x in badWords:
            score -= 1
        
    #print("review is " + ("negative" if score < 0 else "positive"))
    return score >= 0
    
def getUniqueGoodandBadWords():
    with open('../dataset/negWords.txt', 'r') as myFile:
        badList = myFile.read().split()

    with open('../dataset/posWords.txt', 'r') as myFile:
        goodList = myFile.read().split()
        
    shared = [x for x in goodList if x in badList]
    
    goodWords = set([x for x in goodList if x not in shared]) 
    badWords = set([x for x in badList if x not in shared])
    
    return goodWords, badWords
  
  
def fullPosNegTest():
    goodWords, badWords = getUniqueGoodandBadWords();
    
    posReviews, negReviews = loadReviews()

    correct = 0
    count = 0    
    s = ""
    for review in posReviews:
        count += 1
        if posminusneg(review, goodWords, badWords):
            correct += 1
            s = "correct!"
        else:
            s = "wrong :("
        #if count % 10 == 0:
        print(s + "  {:.2f}%  ".format(correct / count * 100) + str(count))
            
    print("halfway there!")
    for review in negReviews:
        count += 1
        if not posminusneg(review, goodWords, badWords):
            correct += 1
            s = "correct!"
        else:
            s = "wrong :("
        #if count % 10 == 0:
        print(s + "  {:.2f}%  ".format(correct / count * 100) + str(count))
            
    return correct / count


def conclusionWeight(review, weight, goodWords, badWords, numSent):
    reviewSent = review.split('.')
    lastPart = reviewSent[-numSent:]
    firstPart = reviewSent[:-numSent]

    lastPart = ''.join(lastPart)
    firstPart = ''.join(firstPart)

    lastPartTokens = customtokenize(lastPart)
    firstPartTokens = customtokenize(firstPart)

    numLastGoodWords = 0;
    numLastBadWords = 0
    for x in lastPartTokens:
        if x in goodWords:
            numLastGoodWords+=1
        if x in badWords:
            numLastBadWords+=1
            
    numFirstGoodWords = 0;
    numFirstBadWords = 0
    for x in firstPartTokens:
        if x in goodWords:
            numFirstGoodWords+=1
        if x in badWords:
            numFirstBadWords+=1


    score = (numFirstGoodWords + (numLastGoodWords * weight)) - \
            (numFirstBadWords + (numLastBadWords * weight))

    #print("review is " + ("negative" if score < 0 else "positive"))

    return score >= 0

def fullConcWeight(weight, numSent):
    goodWords, badWords = getUniqueGoodandBadWords();    
    
    posReviews, negReviews = loadReviews()

    correct = 0
    
    for review in posReviews:
        if conclusionWeight(review, weight, goodWords, badWords, numSent):
            correct += 1
            
    for review in negReviews:
        if not conclusionWeight(review, weight, goodWords, badWords, numSent):
            correct += 1
            
    accuracy = correct / (len(posReviews) + len(negReviews))

    return accuracy
    
    
def printMatrix(matrix):
    rows = len(matrix)
    for i in range(rows):
        print(matrix[i])

def getMatrixMax(matrix):
    rows = len(matrix)
    maximum = []
    indexWeight = []
    for i in range(rows):
        tempMax = max(matrix[i])
        indexWeight.append(matrix[i].index(tempMax))
        maximum.append(tempMax)

    totalMax = max(maximum)
    indexNumSent = maximum.index(totalMax)
    return totalMax, indexNumSent, indexWeight[indexNumSent]

def causeImBored(matrix):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ylabel("Weight")
    plt.xlabel("Sentence Num")

    rows = len(matrix)
    columns = len(matrix[0])

    x = []
    y = []
    z = []

    for i in range(rows):
        for j in range(columns):
            x.append(j)
            y.append(i)
            z.append(matrix[i][j])

    ax.plot(x, y, z)
    plt.show()

def concWeightSim(maxWeight, maxSent):
    percentMatrix = [[0 for x in range(maxWeight)] for x in range(maxSent)]

    for i in range(maxWeight):
        for j in range(maxSent):
            percentMatrix[i][j] = fullConcWeight(i, j)

    print("X-Axis = numSent")
    print("Y-Axis = weight")
    printMatrix(percentMatrix)
    maxPercent, bestWeight, bestNumSent = getMatrixMax(percentMatrix)
    print("\nBest percentage: " + str(maxPercent), end="")
    print("\nBest Sentence num: " + str(bestNumSent), end="")
    print("\nBest Weight: " + str(bestWeight))

    #uncomment for some hawt 3d graph action
    causeImBored(percentMatrix)

    
def buildPosWordList(nameOfDoc, documents):
    
    text_file = open(nameOfDoc + ".txt","w")
    
    for document in documents:
        postokens = pos_tag(customtokenize(document.lower()))
        for i in postokens:
            print(i[0] + " " + i[1], file = text_file)
        print(file = text_file)
        
    text_file.close()    
    

def getSuperGoodBad(topNum):
    import random
    # get shit
    posReviews, negReviews = loadReviews()
    posposList, negposList = loadPosLists()
    
    # choose random 800 for training set
    accuracy = 0.0    
    percenttrain = 0.8
    posTrainCount = int(percenttrain*len(posposList))
    negTrainCount = int(percenttrain*len(negposList)) 
    
    posTuples = list(zip(posReviews, posposList))
    negTuples = list(zip(negReviews, negposList))
    
    random.shuffle(posTuples)
    random.shuffle(negTuples)
    
    superPospos = []
    for tup in posTuples:
        superPospos += tup[1]

    superNegpos = []
    for tup in negTuples:
        superNegpos += tup[1]

    goodFreqDist = FreqDist(superPospos);
    badFreqDict = FreqDist(superNegpos);

    goodWordsDict = FreqDist(superPospos).most_common(topNum);
    badWordsDict = FreqDist(superNegpos).most_common(topNum);
    
    return goodWordsDict, badWordsDict
    
    
    
    
    
    
