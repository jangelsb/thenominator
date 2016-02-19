# this assumes python 2.7 is installed


from imdb import IMDb
ia = IMDb()

movie = imdb.get_movie('0816692')
print "Name of the movie: ", movie
for i in movie['director']:
    print "Director: ", i
    director = ia.search_person(i["name"])[0]
    ia.update(director)
    print "Movies directed by %s:" % director
    for movie_name in director["director movie"]:
        print movie_name





# movie_name = name of movie
# movieid = movie_name.id 
# dir = movie.director
# count = 0, rt_total_score = 0
# movives = dir.movies
# for m in movies:
# 	rt_total_score += m.rt_score
# 	count += 1
# rt_avg = rt_total_score/count

# return "good" if rt_avg > 60 else "shit"