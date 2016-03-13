from nltk import FreqDist
import utils
import random
        
# main test function for training version of elementary classifier
# this function loads the current review dataset as well as the preprocessed part of speech lists
# it then tuples the dataset together and calls the main function below to run the test
# tests are repeated for a number defined by iterations
# topNum defines how many top good and bad words to track in the generated word lists
def getSuperGoodBadAvg(iterations, topNum):

    posList = ['JJ','NN','RB']    
    inclusion = True
    
#    posReviews = utils.loadAllTextFiles('dataset/txt_sentoken/pos/')
#    negReviews = utils.loadAllTextFiles('dataset/txt_sentoken/neg/')
#    posposList = utils.loadPosList('dataset/txt_sentoken/negposlist.txt', posList, inclusion)
#    negposList = utils.loadPosList('dataset/txt_sentoken/posposlist.txt', posList, inclusion)

    posReviews = utils.loadAllTextFiles('dataset/ebert_reviews/4-0/')
    posReviews += utils.loadAllTextFiles('dataset/ebert_reviews/3-5/')
    negReviews = utils.loadAllTextFiles('dataset/ebert_reviews/0-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/0-5/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-5/')

    posposList = utils.loadPosList('dataset/ebert_reviews/pos4-0.txt', posList, inclusion)
    posposList += utils.loadPosList('dataset/ebert_reviews/pos3-5.txt', posList, inclusion)
    negposList = utils.loadPosList('dataset/ebert_reviews/pos0-0.txt', posList, inclusion)
    negposList += utils.loadPosList('dataset/ebert_reviews/pos0-5.txt', posList, inclusion)
    negposList += utils.loadPosList('dataset/ebert_reviews/pos1-0.txt', posList, inclusion)
    negposList += utils.loadPosList('dataset/ebert_reviews/pos1-5.txt', posList, inclusion)
    
    numberOfReviews = 950
    posReviews = posReviews[:numberOfReviews]
    negReviews = negReviews[:numberOfReviews]
    posposList = posposList[:numberOfReviews]
    negposList = negposList[:numberOfReviews]
    
    posTuples = list(zip(posReviews, posposList))
    negTuples = list(zip(negReviews, negposList))    
    
    dataSetGoodWords, dataSetBadWords = utils.getUniqueGoodandBadWords()
    
    totalacc = 0.0
    for i in range(iterations):
        totalacc += getSuperGoodBad(topNum, posTuples, negTuples, dataSetGoodWords, dataSetBadWords)
        print("accuracy : " + "{:.4f}".format(totalacc / (i+1)))
        
            
        
# THINGS TO TRY
# print out top pos and neg words for test set and compare with training!
# try with different pos combinations
# only count up to one word per document (make each document a set instead of bow basically)
# swap bad and good weights at each position?
# try all words of each list over 100 appearances in each (bad may be longer and help offset?)
    
# first shuffle the tuple lists and then unzip them into seperate lists
# then build custom top positive and negative word lists for this training set of reviews
# swap the weights of each list to offset frequency difference between top words
# lastly perform a simple scoring test on the review using the generated word lists
# topNum is how many top words to generate in each list
# posTuples and negTuples are tuples with reviews paired with part of speech lists
# the two last ones are Breen's word data sets
def getSuperGoodBad(topNum, posTuples, negTuples, dataSetGoodWords, dataSetBadWords):
    # 80% will be train set and 20% test set
    percenttrain = 0.8
    byWeight = True    # toggles equal weighting of words versus weighting by frequency
    printing = False   # turn this on if you want to see more detailed info
    
    posLen = len(posTuples)
    trainCount = int(percenttrain*posLen)
    testCount = posLen - trainCount
    
    random.shuffle(posTuples)
    random.shuffle(negTuples)
    
    trainPosTuples = posTuples[:trainCount]
    trainNegTuples = negTuples[:trainCount]
    
    testPosReviews = [tup[0] for tup in posTuples[trainCount:]]
    testNegReviews = [tup[0] for tup in negTuples[trainCount:]]
    
    # probly could optimize this part
    superPospos = []
    for tup in trainPosTuples:
        superPospos += list(set(tup[1]))

    superNegpos = []
    for tup in trainNegTuples:
        superNegpos += list(set(tup[1]))

    goodFreqDist = FreqDist(superPospos);
    badFreqDist = FreqDist(superNegpos);

    topGoodWords = []
    topBadWords = []
    topGoodDict = {}
    topBadDict = {}
    
    # build custom top good and bad word lists
    while len(topGoodWords) < topNum or len(topBadWords) < topNum:
        if len(topGoodWords) < topNum:        
            goodTup = goodFreqDist.most_common(1)[0]
            goodWord = goodTup[0]
            goodWeight = goodTup[1]
            topGoodDict[goodWord] = goodWeight
            del goodFreqDist[goodWord]

            if goodWord in dataSetGoodWords:
                if goodWord in topBadWords:
                    topBadWords.remove(goodWord)
                else:
                    topGoodWords.append(goodWord)
            
        if len(topBadWords) < topNum:
            badTup = badFreqDist.most_common(1)[0]
            badWord = badTup[0];
            badWeight = badTup[1]
            topBadDict[badWord] = badWeight
            del badFreqDist[badWord]
              
            if badWord in dataSetBadWords:        
                if badWord in topGoodWords:
                    topGoodWords.remove(badWord)
                else:
                    topBadWords.append(badWord)
    
    topGoodCheck = [(word,topGoodDict[word]) for word in topGoodWords]
    topBadCheck = [(word,topBadDict[word]) for word in topBadWords]
    
    topGoodDict = {}
    topBadDict = {}
    # swap weights of each set to equalize versus appearance frequency
    for i in range(len(topGoodCheck)):
        gtup = topGoodCheck[i]
        btup = topBadCheck[i]
        
        topGoodDict[gtup[0]] = btup[1]
        topBadDict[btup[0]] = gtup[1]
        # alternate strategy
        #topGoodDict[gtup[0]] = trainCount - gtup[1]
        #topBadDict[btup[0]] = trainCount - btup[1]
        

    count = 0
    correct = 0    
    avgposscore = 0
    avgnegscore = 0
    curacc = 0
    for posReview in testPosReviews:
        count+=1
        score = 0    
        reviewTokens = utils.customtokenize(posReview.lower())
        for token in reviewTokens:
            if token in topGoodWords:
                score += topGoodDict[token] if byWeight else 1
            if token in topBadWords:
                score -= topBadDict[token] if byWeight else 1
        avgposscore += score
        if score > 0:
            if printing: print("correct! " + str(score))
            correct+=1
        else:
            if printing: print("wrong " + str(score))
            
    curacc = correct/count
    
    for negReview in testNegReviews:
        count+=1
        score = 0    
        reviewTokens = utils.customtokenize(negReview.lower())
        for token in reviewTokens:
            if token in topGoodWords:
                score += topGoodDict[token] if byWeight else 1
            if token in topBadWords:
                score -= topBadDict[token] if byWeight else 1
        avgnegscore += score
        if score <= 0:
            if printing: print("correct! " + str(score))
            correct+=1
        else:
            if printing: print("wrong " + str(score))
    
    finalacc = correct / count
    if printing:
        print("avg positive score : " + "{:.4f}".format(avgposscore / testCount))
        print("avg negative score : " + "{:.4f}".format(avgnegscore / testCount))
        print("positive accuracy  : " + "{:.4f}".format(curacc))
        #calculate positive accuracy from final and negative
        curacc += (finalacc - curacc)*2
        print("negative accuracy  : " + "{:.4f}".format(curacc))    
        print("final accuracy     : " + "{:.4f}".format(finalacc))
        
    return finalacc
    
    
