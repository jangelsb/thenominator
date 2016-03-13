# Grabs movies from the lists (located in /dataset/movietitles) 
# to search Roger Ebert's website for the review and score. 
# Then stores all the reviews in folders based on their score; 
# This ultimately makes a supervised data set of movie reviews.



import re
from bs4 import BeautifulSoup
import os
import urllib.request

rogerberturl = "http://www.rogerebert.com/reviews/"


# it gets the movie names from the text file and converts it to the format made for roger ebert's site
# e.g. the hunger games -> the-hunger-games-2012
def getMovieNames(year):
     with open('{}.txt'.format(year), 'r', encoding="utf8") as f:
        ret = f.read()
        ret = ret.lower() #make lowercase 
        ret = re.sub('[^0-9a-zA-Z \'\n]', ' ', ret) #only keep words, numbers, spaces, numbers \' and \n
        ret = re.sub(' +', ' ',ret) #remove extra whitespace
        ret = re.sub('\'', '',ret) #remove apostrophes

        ret = ret.replace('\n','-{}\n'.format(year)) #add in '-year' at the end
        ret = ret.replace(' ', '-') #replace ' ' with '-'
        ret = ret.split('\n') #create a list 

        ret = [x[1:] if len(x) > 0 and x[0] == '-' else x for x in ret] #removes a '-' in front of each value in the list, if it exists
        return ret #list of the movies in roger ebert's url format


# given a movie, e.g., 'the-hunger-games-2012'
# this function retures the score and the review from roger ebert's site, if it exists 
def grabReview(movie):

    score = None
    review = None

    print("visiting {}".format(movie), end="")
    try:
        html = urllib.request.urlopen(rogerberturl+movie).read()

        soup = BeautifulSoup(html, 'html.parser')
        s = soup.find("meta", {"itemprop": "ratingValue"})
        score = str(s).split("\"")[1] #this gives the value of the review
        # print(score)

        review=soup.find("div", {"itemprop": "reviewBody"})
        review=review.getText()
        # now clean up the garabage left over by the review
        review=review.replace("Watch Now"," ")
        review=review.replace("\xa0"," ")
        review=review.replace("\n"," ")
        review = re.sub(' +', ' ',review) #remove extra whitespace
        review=review.strip() 

        score=score.replace('.','-') #to use it as a folder name

        print("")
    except:
        print(" --- NOT FOUND")

    return score, review


def makeSurePathExists(dir):
    newpath = './{}'.format(dir)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    

# grabs the review for the movie, and prints it to the file in a folder based on review score
def printReviewToFile(movie):

    score, review = grabReview(movie)

    # if the review exists, save it
    if score is not None:
        makeSurePathExists(score)

        with open("./{}/{}-{}.txt".format(score,movie,score), "w") as f:
            print("{}".format(review), file=f)
            

if __name__ == "__main__":

    # this assumes you have movie names in the same directory as this file of the style: year.txt
    # e.g., 2014.txt

    for year in range(2016,2017):
        movies =  getMovieNames(year)
        for i, movie in enumerate(movies):
            if i > 5:
                break
            printReviewToFile(movie)
