import utils


def posminusneg(review, goodWords, badWords):
    wordsInReview = utils.customtokenize(review) #69.90%
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
    

def fullPosNegTest():
    goodWords, badWords = utils.getUniqueGoodandBadWords();
    
#    posReviews = utils.loadAllTextFiles('dataset/txt_sentoken/pos/')
#    negReviews = utils.loadAllTextFiles('dataset/txt_sentoken/neg/')
    posReviews = utils.loadAllTextFiles('dataset/ebert_reviews/4-0/')
    posReviews += utils.loadAllTextFiles('dataset/ebert_reviews/3-5/')
    negReviews = utils.loadAllTextFiles('dataset/ebert_reviews/0-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/0-5/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-5/')
    
    numberOfReviews = 950
    posReviews = posReviews[:numberOfReviews]
    negReviews = negReviews[:numberOfReviews]

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

    lastPartTokens = utils.customtokenize(lastPart)
    firstPartTokens = utils.customtokenize(firstPart)

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

# weight for last sentences
# number of sentences to increase weight
def fullConcWeight(weight, numSent):
    goodWords, badWords = utils.getUniqueGoodandBadWords();    
    
#    posReviews = utils.loadAllTextFiles('dataset/txt_sentoken/pos/')
#    negReviews = utils.loadAllTextFiles('dataset/txt_sentoken/neg/')
    
    posReviews = utils.loadAllTextFiles('dataset/ebert_reviews/4-0/')
    posReviews += utils.loadAllTextFiles('dataset/ebert_reviews/3-5/')
    negReviews = utils.loadAllTextFiles('dataset/ebert_reviews/0-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/0-5/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-0/')
    negReviews += utils.loadAllTextFiles('dataset/ebert_reviews/1-5/')
    
    numberOfReviews = 950
    posReviews = posReviews[:numberOfReviews]
    negReviews = negReviews[:numberOfReviews]

    correct = 0
    
    for review in posReviews:
        if conclusionWeight(review, weight, goodWords, badWords, numSent):
            correct += 1
            
    for review in negReviews:
        if not conclusionWeight(review, weight, goodWords, badWords, numSent):
            correct += 1
            
    accuracy = correct / (len(posReviews) + len(negReviews))

    return accuracy
    


# figure out best weight and sentence numbers for fullConcTest
def concWeightSim(maxWeight, maxSent):
    percentMatrix = [[0 for x in range(maxWeight)] for x in range(maxSent)]

    for i in range(maxWeight):
        for j in range(maxSent):
            percentMatrix[i][j] = fullConcWeight(i, j)

    print("X-Axis = numSent")
    print("Y-Axis = weight")
    utils.printMatrix(percentMatrix)
    maxPercent, bestWeight, bestNumSent = getMatrixMax(percentMatrix)
    print("\nBest percentage: " + str(maxPercent), end="")
    print("\nBest Sentence num: " + str(bestNumSent), end="")
    print("\nBest Weight: " + str(bestWeight))

    #uncomment for some hawt 3d graph action
    drawWeightMatrix(percentMatrix)
    
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
    
    # draws the weight matrix for above test
def drawWeightMatrix(matrix):
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

